#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used-Car Scout（Discord通知版 / ルールベース + 分位回帰ハイブリッド）
- JSなしで取得できる範囲のリスト/リンク部から 年式/距離/価格/タイトル文字 を抽出
- 相場1: 同回収データの価格中央値から price_ratio を算出（常時）
- 相場2: 収集件数が十分な回は、分位回帰(Quantile GB)のOOFで p50(中央値) / p20(下位20%) を推定
    * お得度 = p50 - 実売価格 でスコア/緊急度を後押し
    * 実売価格 ≤ p20 なら強い割安としてさらにブースト
- 緊急度4以上だけ Discord に通知。上位を results.csv に保存
環境変数:
  DISCORD_WEBHOOK_URL  … Discord Incoming Webhook URL（必須/通知）
  TARGETS_JSON         … 監視ターゲット上書き（任意のJSON文字列）
"""
from __future__ import annotations
import os, re, csv, json, time, math
from datetime import datetime
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept-Language": "ja,en;q=0.9"}

# --- 監視ターゲット（必要に応じて TARGETS_JSON で上書き） ---
DEFAULT_TARGETS = [
    {
        "name": "ハリアー",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/bTO/index.html?CARC=TO_S114%2ATO_S115&SORT=22",
        "price_max": 3000000,
        "year_min": 2014,
        "mileage_max": 100000,
        "pages": 1,
    },
    {
        "name": "フォレスター",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/bSU/index.html?CARC=SU_S042%2ASU_S043&SORT=22",
        "price_max": 2800000,
        "year_min": 2014,
        "mileage_max": 120000,
        "pages": 1,
    },
    {
        "name": "CX-5",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/bMA/index.html?CARC=MA_S009&SORT=22",
        "price_max": 2500000,
        "year_min": 2015,
        "mileage_max": 100000,
        "pages": 1,
    },
    {
        "name": "（参考）グーネット",
        "site": "goonet",
        "url": "https://www.goo-net.com/cgi-bin/fsearch/goo_used_search.cgi?category=USDN",
        "price_max": 99999999,
        "year_min": 2010,
        "mileage_max": 999999,
        "pages": 1,
    },
]

# --- 正規表現ヘルパ ---
_price_ja = re.compile(r"([0-9,.]+)\s*万円|([0-9,]+)\s*円")
_km_ja    = re.compile(r"([0-9.]+)\s*万?km")
_year_ja  = re.compile(r"(\d{4})年")


def textnum_to_int(val: str) -> int:
    m = _price_ja.search(val)
    if not m:
        return 0
    if m.group(1):  # 万円
        n = float(m.group(1).replace(",", "")) * 10000
        return int(n)
    return int(m.group(2).replace(",", ""))


def km_to_int(val: str) -> int:
    m = _km_ja.search(val)
    if not m:
        return 0
    s = m.group(1)
    if "万" in val:
        return int(float(s) * 10000)
    return int(float(s))


def year_from_text(val: str) -> int:
    m = _year_ja.search(val)
    if not m:
        return 0
    return int(m.group(1))


# --- polite fetch ---
def fetch(url: str) -> str:
    time.sleep(1.2)  # 負荷配慮
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


# --- サイト別パーサ ---
def parse_carsensor_list(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    items: List[Dict[str, Any]] = []
    cards = soup.select(".cassette, .list, .cl-list-item, .cl-list-content") or []
    if not cards:
        cards = soup.select("a")
    for c in cards:
        try:
            a = c.select_one("h3 a") or c.select_one("a")
            if not a or not a.get("href"):
                continue
            title = a.get_text(strip=True)
            url = a["href"]
            if url.startswith("/"):
                url = "https://www.carsensor.net" + url
            t = c.get_text(" ", strip=True)
            price = textnum_to_int(t)
            year = year_from_text(t)
            mileage = km_to_int(t)
            if price == 0 and ("価格" in t or "万円" in t):
                continue
            items.append({
                "title": title,
                "url": url,
                "year": year,
                "mileage": mileage,
                "price": price,
                "site": "carsensor",
            })
        except Exception:
            continue
    return items


def parse_goonet_list(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    items: List[Dict[str, Any]] = []
    rows = soup.select("a")
    for a in rows:
        href = a.get("href", "")
        if "/usedcar/" in href or "goo-net.com/usedcar" in href:
            title = a.get_text(strip=True)
            url = href if href.startswith("http") else "https://www.goo-net.com" + href
            t = a.get_text(" ", strip=True)
            items.append({
                "title": title,
                "url": url,
                "year": year_from_text(t),
                "mileage": km_to_int(t),
                "price": textnum_to_int(t),
                "site": "goonet",
            })
    return items


SITE_PARSERS = {
    "carsensor": parse_carsensor_list,
    "goonet":    parse_goonet_list,
}

# --- 相場/判定（ルール） ---
def _percentile(sorted_list, p: float):
    if not sorted_list:
        return None
    k = (len(sorted_list) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_list[int(k)]
    return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)


def compute_price_stats(items):
    prices = [it.get("price") for it in items if it.get("price")]
    if len(prices) < 4:
        return {"median": None, "q25": None}
    s = sorted(prices)
    return {"median": _percentile(s, 0.5), "q25": _percentile(s, 0.25)}


# --- 分位回帰 OOF: p50 / p20 予測 ---
def _feat(it: Dict[str, Any]):
    title = (it.get("title") or "")
    return [
        it.get("year") or 0,
        it.get("mileage") or 0,
        1 if "サンルーフ" in title else 0,
        1 if "レザー" in title else 0,
        1 if "BOSE" in title else 0,
        1 if "JBL"  in title else 0,
    ]


def oof_quantile_preds(collected: List[Dict[str, Any]], alphas=(0.5, 0.2)):
    X = np.array([_feat(it) for it in collected], dtype=float)
    y = np.array([it.get("price") or 0 for it in collected], dtype=float)
    n = len(collected)
    if n < 12 or y.sum() == 0:
        return {a: [None]*n for a in alphas}
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    out = {a: np.zeros(n) for a in alphas}
    for a in alphas:
        preds = np.zeros(n)
        for tr, te in kf.split(X):
            mdl = GradientBoostingRegressor(loss="quantile", alpha=a, random_state=42)
            mdl.fit(X[tr], y[tr])
            preds[te] = mdl.predict(X[te])
        out[a] = preds
    return {a: v.tolist() for a, v in out.items()}


def assess_deal(it, stats, cfg):
    now = datetime.now().year
    price = it.get("price") or 0
    year  = it.get("year") or 0
    km    = it.get("mileage") or 0
    age   = max(0, now - year) if year else 15

    median = (stats or {}).get("median") or 0
    price_ratio = (price / median) if (price and median) else 1.0

    # ベース（価格比重視）
    if price_ratio <= 0.6:   base = 95
    elif price_ratio <= 0.7: base = 85
    elif price_ratio <= 0.8: base = 75
    elif price_ratio <= 0.9: base = 60
    elif price_ratio <= 1.0: base = 45
    else:                    base = 30

    # 微調整（年式/距離/文字）
    adj = 0
    if age <= 8: adj += 5
    if km and km <= 0.7 * cfg.get("mileage_max", 150_000): adj += 5
    title = (it.get("title") or "")
    if any(k in title for k in ["サンルーフ", "レザー", "BOSE", "JBL"]): adj += 3
    if any(k in title for k in ["修復歴", "事故", "冠水"]):               adj -= 20

    score = max(0, min(100, base + adj))

    # 予測ベース（OOF）ブースト
    pred50 = it.get("pred_p50")
    pred20 = it.get("pred_p20")
    gap = None
    if isinstance(pred50, (int, float)) and pred50 > 0:
        gap = pred50 - price
        if gap > 500_000:
            score = min(100, score + 8)
        elif gap > 300_000:
            score = min(100, score + 5)
    if isinstance(pred20, (int, float)) and pred20 > 0 and price <= pred20:
        score = min(100, score + 5)

    # 緊急度（1-5）
    if score >= 90:   urgency = 5
    elif score >= 80: urgency = 4
    elif score >= 70: urgency = 3
    elif score >= 60: urgency = 2
    else:             urgency = 1
    if price_ratio <= 0.6:
        urgency = min(5, urgency + 1)
    if gap is not None and gap > 500_000:
        urgency = min(5, urgency + 1)

    # 相場不明フォールバック（中央値が無い回）
    if not median:
        if (price and price <= 0.8 * cfg.get("price_max", 9e9)
            and year and year >= cfg.get("year_min", 0) + 2
            and km and km <= 0.7 * cfg.get("mileage_max", 9e9)):
            urgency = max(urgency, 4)
            score   = max(score, 80)

    it["price_ratio"] = round(price_ratio, 2)
    it["score"]       = score
    it["urgency"]     = int(urgency)
    it["deal_gap"]    = int(gap) if gap is not None else None


# --- Discord通知 ---
def discord_notify(items: List[Dict[str, Any]]):
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("[INFO] DISCORD_WEBHOOK_URL 未設定。Discord通知をスキップ")
        return
    cands = [x for x in items if x.get("urgency", 1) >= 4][:5]
    if not cands:
        print("[INFO] Discord通知対象なし（L4+なし）")
        return
    embeds = []
    for it in cands:
        price = f"{it.get('price',0):,}円" if it.get('price') else "—"
        year  = it.get("year") or "—"
        km    = it.get("mileage") or 0
        km_s  = f"{km:,}km" if km else "—"
        gap   = it.get("deal_gap")
        gap_s = (f"+{gap:,}円" if isinstance(gap, int) and gap is not None and gap >= 0
                 else (f"{gap:,}円" if gap is not None else "—"))
        p50 = it.get("pred_p50"); p20 = it.get("pred_p20")
        p50s = f"{int(p50):,}円" if isinstance(p50, (int,float)) else "—"
        p20s = f"{int(p20):,}円" if isinstance(p20, (int,float)) else "—"
        embeds.append({
            "title": (it.get("title") or "")[:256],
            "url": it.get("url"),
            "description": f"{it.get('site','')} | {year}年 | {km_s} | {price}",
            "fields": [
                {"name": "Score",       "value": str(it.get("score", 0)),             "inline": True},
                {"name": "Price Ratio", "value": str(it.get("price_ratio", '-')),     "inline": True},
                {"name": "Urgency",     "value": "🔥" * it.get("urgency", 1),         "inline": True},
                {"name": "Deal Gap",    "value": gap_s,                                "inline": True},
                {"name": "p50(pred)",   "value": p50s,                                 "inline": True},
                {"name": "p20(pred)",   "value": p20s,                                 "inline": True},
            ],
        })
    payload = {
        "content": f"🚗 **おすすめ中古車ピックアップ（即買い候補）**\n{datetime.now():%Y-%m-%d %H:%M}",
        "embeds": embeds,
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        print("[OK] Discord 通知完了")
    except Exception as e:
        print(f"[WARN] Discord 通知失敗: {e}")


# --- メイン ---
def main():
    targets = DEFAULT_TARGETS
    tj = os.getenv("TARGETS_JSON")
    if tj:
        try:
            targets = json.loads(tj)
        except Exception as e:
            print(f"[WARN] TARGETS_JSON の読み込み失敗: {e}")

    all_picks: List[Dict[str, Any]] = []

    for cfg in targets:
        site = cfg.get("site")
        url  = cfg.get("url")
        pages = int(cfg.get("pages", 1))
        parser = SITE_PARSERS.get(site)
        if not parser:
            print(f"[SKIP] 未対応サイト: {site}")
            continue

        collected: List[Dict[str, Any]] = []
        for page in range(1, pages + 1):
            u = url + (f"&page={page}" if page > 1 else "")
            try:
                print(f"[GET] {u}")
                html = fetch(u)
                items = parser(html)
                collected.extend(items)
            except requests.HTTPError as e:
                code = getattr(e.response, "status_code", "?")
                print(f"[HTTP {code}] {u}")
            except Exception as e:
                print(f"[ERR] {u}: {e}")

        # 分位回帰 OOF 予測（十分な件数がある回だけ有効）
        qpreds = oof_quantile_preds(collected, alphas=(0.5, 0.2))
        for i, it in enumerate(collected):
            it["pred_p50"] = qpreds.get(0.5, [None]*len(collected))[i] if collected else None
            it["pred_p20"] = qpreds.get(0.2, [None]*len(collected))[i] if collected else None

        # 中央値相場 → 評価
        stats = compute_price_stats(collected)
        for it in collected:
            assess_deal(it, stats, cfg)

        # L4+を優先、足りなければスコア上位で補完
        picks = [x for x in collected if x.get("urgency", 1) >= 4]
        if len(picks) < 8:
            extra = sorted([x for x in collected if x not in picks],
                           key=lambda x: x.get("score", 0), reverse=True)
            picks.extend(extra[:8 - len(picks)])

        print(f"  → 候補 {len(picks)} 件（{cfg.get('name')}）")
        all_picks.extend(picks)

    # 全体から上位10
    all_picks.sort(key=lambda x: x.get("score", 0), reverse=True)
    top = all_picks[:10]

    # CSV 出力
    out_csv = "results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["title","url","site","year","mileage","price",
                    "score","price_ratio","urgency","pred_p50","pred_p20","deal_gap"])
        for it in top:
            p50 = it.get("pred_p50"); p20 = it.get("pred_p20")
            w.writerow([
                it.get("title", ""), it.get("url", ""), it.get("site", ""),
                it.get("year", 0), it.get("mileage", 0), it.get("price", 0),
                it.get("score", 0), it.get("price_ratio", "-"), it.get("urgency", 1),
                int(p50) if isinstance(p50, (int,float)) else "",
                int(p20) if isinstance(p20, (int,float)) else "",
                it.get("deal_gap","")
            ])
    print(f"[OK] CSV 出力: {out_csv}（{len(top)}件）")

    # Discord 通知
    discord_notify(top)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used-Car Scout（Discord通知版 / SUV監視・ハスラー除外 / ルール + 分位回帰ハイブリッド）
- 検索エンドポイント + スマホ版(STID=SMPH0001) + PAGE巡回で、requests+BS4で安定取得
- 一覧で年式/距離/価格が拾えない場合、詳細ページを1回だけ参照して補完（負荷低減）
- 相場1: 同回収データの価格“中央値”から price_ratio を算出（常時）
- 相場2: 件数十分な回は、分位回帰(Quantile GB, OOF)で p50/p20 を推定 → お得度ブースト
- SUVのみ拾う include_keywords / 常時除外に「ハスラー/HUSTLER」
- 上位を results.csv に保存
- Discordは二段通知：🚀即買い（デフォ緊急度≧4）と 🤔ありかも（緊急度3 or スコア70–84.9）

環境変数:
  DISCORD_WEBHOOK_URL_MAIN  … 即買いレベルの通知先（必須推奨）
  DISCORD_WEBHOOK_URL_MAYBE … ありかもレベルの通知先（任意）
  DISCORD_WEBHOOK_URL       … 旧単一Webhook名（MAIN未設定時のフォールバック）
  TARGETS_JSON              … 監視ターゲット上書き（JSON文字列）
  DISCORD_DRY_RUN           … "1" なら通知せず、送信内容をコンソール表示

  IMMEDIATE_URGENCY_MIN     … 即買い緊急度しきい値（既定 4）
  MAYBE_SCORE_MIN           … ありかもスコア下限（既定 70）
  MAYBE_SCORE_MAX           … ありかもスコア上限（既定 84.9）
"""
from __future__ import annotations
import os, re, csv, json, time, math
from datetime import datetime
from typing import List, Dict, Any
from unicodedata import normalize
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

# --------- 通信設定 ---------
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept-Language": "ja,en;q=0.9"}

# --------- 二段階通知のしきい値（環境変数で上書き可） ---------
IMMEDIATE_URGENCY_MIN = int(os.getenv("IMMEDIATE_URGENCY_MIN", "4"))   # 即買い: 緊急度≧4
MAYBE_SCORE_MIN = float(os.getenv("MAYBE_SCORE_MIN", "70"))            # ありかも: スコア下限
MAYBE_SCORE_MAX = float(os.getenv("MAYBE_SCORE_MAX", "84.9"))          # ありかも: 上限(即買い未満)

# Webhook（MAINは旧DISCORD_WEBHOOK_URLをフォールバック）
WEBHOOK_MAIN  = os.getenv("DISCORD_WEBHOOK_URL_MAIN") or os.getenv("DISCORD_WEBHOOK_URL")
WEBHOOK_MAYBE = os.getenv("DISCORD_WEBHOOK_URL_MAYBE")
DRY_RUN = os.getenv("DISCORD_DRY_RUN", "0") == "1"

# --------- 監視ターゲット（必要に応じて TARGETS_JSON で上書き） ---------
# デフォルトはあなたが提示した検索URLをサンプルで1件入れておく（軽めに pages=2）
DEFAULT_TARGETS = [
    {
        "name": "テスト: 北海道SUV 2013年〜 7万km以下 〜120万円",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/search.php?STID=CS210610&YMIN=2013&AR=4&SMAX=70000&PMAX=1200000&SP=D&NOTKEI=1&BT=X",
        "pages": 2,
        "price_max": 1200000,
        "year_min": 2013,
        "mileage_max": 70000,
        "include_keywords": [],  # URL側で条件指定しているので空でOK
        "exclude_keywords": []
    }
]

# --------- 正規表現ヘルパ ---------
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

# --------- URL補正（スマホ版/PAGE付与） ---------
def _ensure_mobile_url(url: str) -> str:
    """Carsensor検索URLをスマホ版に強制(STID=SMPH0001)。AL=1は表記安定化用。"""
    p = urlparse(url)
    q = parse_qs(p.query)
    q['STID'] = ['SMPH0001']  # スマホ版
    q.setdefault('AL', ['1'])
    new_q = urlencode(q, doseq=True)
    return urlunparse(p._replace(query=new_q))

def _with_page(url: str, page: int) -> str:
    """PAGE=n を付与"""
    p = urlparse(url)
    q = parse_qs(p.query)
    q['PAGE'] = [str(page)]
    new_q = urlencode(q, doseq=True)
    return urlunparse(p._replace(query=new_q))

# --------- polite fetch ---------
def fetch(url: str) -> str:
    time.sleep(1.2)  # 負荷配慮（必要に応じてランダム化）
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text

# --------- サイト別パーサ（スマホ版前提） ---------
def parse_carsensor_list(html: str) -> List[Dict[str, Any]]:
    """スマホ版で /usedcar/detail/ へのリンクのみ拾い、親ブロックの文字から数値抽出"""
    soup = BeautifulSoup(html, "lxml")
    items: List[Dict[str, Any]] = []
    seen = set()

    for a in soup.select('a[href*="/usedcar/detail/"]'):
        href = a.get("href", "")
        if not href:
            continue
        url = href if href.startswith("http") else "https://www.carsensor.net" + href
        if url in seen:
            continue
        seen.add(url)

        title = a.get_text(" ", strip=True)

        # 親を数階層たどって、まとまりのあるテキストを取得
        block = a
        for _ in range(4):
            if block and block.parent:
                block = block.parent
            else:
                break
        text = (block.get_text(" ", strip=True) if block else a.get_text(" ", strip=True))

        items.append({
            "title": title,
            "url": url,
            "year": year_from_text(text),
            "mileage": km_to_int(text),
            "price": textnum_to_int(text),
            "site": "carsensor",
        })
    return items

def parse_goonet_list(html: str) -> List[Dict[str, Any]]:
    # 使うなら同様に実装。今回は未使用。
    soup = BeautifulSoup(html, "lxml")
    out: List[Dict[str, Any]] = []
    for a in soup.select('a[href*="/usedcar/"]'):
        href = a.get("href", "")
        if not href:
            continue
        url = href if href.startswith("http") else "https://www.goo-net.com" + href
        t = a.get_text(" ", strip=True)
        out.append({
            "title": a.get_text(strip=True),
            "url": url,
            "year": year_from_text(t),
            "mileage": km_to_int(t),
            "price": textnum_to_int(t),
            "site": "goonet",
        })
    return out

SITE_PARSERS = {
    "carsensor": parse_carsensor_list,
    "goonet":    parse_goonet_list,
}

# --------- 詳細ページ補完 ---------
def enrich_from_detail(it: Dict[str, Any]) -> None:
    """一覧で抜けた year/mileage/price を詳細HTMLから補完"""
    try:
        h = fetch(it["url"])
    except Exception:
        return
    if not it.get("year"):
        it["year"] = year_from_text(h) or it.get("year", 0)
    if not it.get("mileage"):
        it["mileage"] = km_to_int(h) or it.get("mileage", 0)
    if not it.get("price"):
        it["price"] = textnum_to_int(h) or it.get("price", 0)

# --------- タイトル正規化 & キーワードフィルタ（SUVだけ/ハスラー除外） ---------
def _norm(s: str) -> str:
    return normalize("NFKC", (s or "")).upper()

def keyword_filter(items, include_keywords=None, exclude_keywords=None):
    inc = [_norm(k) for k in (include_keywords or [])]
    exc = {_norm(k) for k in ((exclude_keywords or []) + ["ハスラー", "HUSTLER"])}
    out = []
    for it in items:
        title_n = _norm(it.get("title", ""))
        if inc and not any(k in title_n for k in inc):
            continue
        if any(k in title_n for k in exc):
            continue
        out.append(it)
    return out

# --------- 相場/判定（ルール） ---------
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

# --------- 分位回帰 OOF: p50 / p20 予測 ---------
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

    it["price_ratio"] = round(price_ratio, 2) if median else "-"
    it["score"]       = score
    it["urgency"]     = int(urgency)
    it["deal_gap"]    = int(gap) if gap is not None else None

# --------- Discord通知（共通） ---------
def _post_discord(items: List[Dict[str, Any]], webhook_url: str | None, title_prefix: str):
    if not items:
        print(f"[INFO] {title_prefix} 通知対象なし")
        return
    if DRY_RUN:
        preview = [{
            "title": it.get("title"),
            "url": it.get("url"),
            "price": it.get("price"),
            "year": it.get("year"),
            "mileage": it.get("mileage"),
            "score": it.get("score"),
            "urgency": it.get("urgency"),
            "price_ratio": it.get("price_ratio"),
            "deal_gap": it.get("deal_gap"),
        } for it in items]
        print(f"[DRY-RUN] {title_prefix} payload preview:", json.dumps(preview, ensure_ascii=False, indent=2))
        return
    if not webhook_url:
        print(f"[INFO] {title_prefix} 用Webhook未設定。通知スキップ")
        return

    embeds = []
    for it in items[:5]:  # 各カテゴリ最大5件
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
                {"name": "Score",       "value": str(it.get("score", 0)),         "inline": True},
                {"name": "Price Ratio", "value": str(it.get("price_ratio", '-')), "inline": True},
                {"name": "Urgency",     "value": "🔥" * it.get("urgency", 1),     "inline": True},
                {"name": "Deal Gap",    "value": gap_s,                            "inline": True},
                {"name": "p50(pred)",   "value": p50s,                             "inline": True},
                {"name": "p20(pred)",   "value": p20s,                             "inline": True},
            ],
        })
    payload = {
        "content": f"{title_prefix}\n{datetime.now():%Y-%m-%d %H:%M}",
        "embeds": embeds,
    }
    try:
        r = requests.post(webhook_url, json=payload, timeout=12)
        r.raise_for_status()
        print(f"[OK] Discord 通知完了: {title_prefix}")
    except Exception as e:
        print(f"[WARN] Discord 通知失敗({title_prefix}): {e}")

# 後方互換の単一通知（必要なら利用）
def discord_notify(items: List[Dict[str, Any]]):
    url = WEBHOOK_MAIN  # 旧: DISCORD_WEBHOOK_URL を含む
    if DRY_RUN:
        preview = [{
            "title": it.get("title"),
            "url": it.get("url"),
            "price": it.get("price"),
            "year": it.get("year"),
            "mileage": it.get("mileage"),
            "score": it.get("score"),
            "urgency": it.get("urgency"),
            "price_ratio": it.get("price_ratio"),
            "deal_gap": it.get("deal_gap"),
        } for it in items if it.get("urgency",1) >= IMMEDIATE_URGENCY_MIN]
        print("[DRY-RUN] (legacy) Discord payload preview:", json.dumps(preview, ensure_ascii=False, indent=2))
        return
    if not url:
        print("[INFO] DISCORD_WEBHOOK_URL 未設定。Discord通知をスキップ")
        return
    cands = [x for x in items if x.get("urgency", 1) >= IMMEDIATE_URGENCY_MIN][:5]
    if not cands:
        print("[INFO] Discord通知対象なし（即買い該当なし）")
        return
    _post_discord(cands, url, "🚀 即買いレベル（legacy）")

# --------- 候補の分割（即買い / ありかも） ---------
def split_candidates(all_items: List[Dict[str, Any]]):
    immediate = [x for x in all_items if x.get("urgency",1) >= IMMEDIATE_URGENCY_MIN]
    maybe = [x for x in all_items
             if (x.get("urgency",1) == 3) or
                (MAYBE_SCORE_MIN <= x.get("score",0) <= MAYBE_SCORE_MAX)]
    ids = set(id(x) for x in immediate)
    maybe = [x for x in maybe if id(x) not in ids]
    immediate.sort(key=lambda x: x.get("score",0), reverse=True)
    maybe.sort(key=lambda x: x.get("score",0), reverse=True)
    return immediate[:5], maybe[:5]

# --------- メイン ---------
def main():
    targets = DEFAULT_TARGETS
    tj = os.getenv("TARGETS_JSON")
    if tj:
        try:
            targets = json.loads(tj)
        except Exception as e:
            print(f"[WARN] TARGETS_JSON の読み込み失敗: {e}")

    all_picks: List[Dict[str, Any]] = []
    all_items: List[Dict[str, Any]] = []

    for cfg in targets:
        site = cfg.get("site")
        url  = cfg.get("url")
        pages = int(cfg.get("pages", 1))
        parser = SITE_PARSERS.get(site)
        if not parser:
            print(f"[SKIP] 未対応サイト: {site}")
            continue

        collected: List[Dict[str, Any]] = []
        mobile_base = _ensure_mobile_url(url)

        for page in range(1, pages + 1):
            u = _with_page(mobile_base, page)
            try:
                print(f"[GET] {u}")
                html = fetch(u)
                items = parser(html)
                # SUVのみ + ハスラー除外（表記ゆれ吸収）
                items = keyword_filter(
                    items,
                    include_keywords=cfg.get("include_keywords"),
                    exclude_keywords=cfg.get("exclude_keywords")
                )

                # 足りない数値は詳細1回で補完（取りすぎ防止で上位30件まで）
                need_enrich = [x for x in items if (not x.get("price") or not x.get("year") or not x.get("mileage"))]
                for it in need_enrich[:30]:
                    enrich_from_detail(it)

                collected.extend(items)
                print(f"  └ parsed {len(items)} items (page {page})")
            except requests.HTTPError as e:
                code = getattr(e.response, "status_code", "?")
                print(f"[HTTP {code}] {u}")
            except Exception as e:
                print(f"[ERR] {u}: {e}")

        # 分位回帰 OOF 予測（十分な件数がある回だけ）
        qpreds = oof_quantile_preds(collected, alphas=(0.5, 0.2))
        for i, it in enumerate(collected):
            it["pred_p50"] = qpreds.get(0.5, [None]*len(collected))[i] if collected else None
            it["pred_p20"] = qpreds.get(0.2, [None]*len(collected))[i] if collected else None

        # 中央値相場 → 評価
        stats = compute_price_stats(collected)
        for it in collected:
            assess_deal(it, stats, cfg)

        all_items.extend(collected)

        # CSV用の上位候補（L4+優先、足りなければスコアで補完）
        picks = [x for x in collected if x.get("urgency", 1) >= 4]
        if len(picks) < 8:
            extra = sorted([x for x in collected if x not in picks],
                           key=lambda x: x.get("score", 0), reverse=True)
            picks.extend(extra[:8 - len(picks)])
        print(f"  → 候補 {len(picks)} 件（{cfg.get('name')}）")
        all_picks.extend(picks)

    # 全体から上位10を書き出し（互換）
    all_picks.sort(key=lambda x: x.get("score", 0), reverse=True)
    top = all_picks[:10]

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

    # 二段階通知
    immediate, maybe = split_candidates(all_items)
    _post_discord(immediate, WEBHOOK_MAIN,  "🚀 即買いレベル")
    _post_discord(maybe,    WEBHOOK_MAYBE, "🤔 ありかもレベル")

    # 互換の単一通知を使う場合は必要に応じて
    # discord_notify(top)

if __name__ == "__main__":
    main()

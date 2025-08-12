#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used-Car Scout（Discord通知版 / SUV監視・ハスラー除外 / ルール + 分位回帰ハイブリッド）
- JSなしで取得できる範囲のリスト/リンク部から 年式/距離/価格/タイトル文字 を抽出
- 相場1: 同回収データの価格中央値から price_ratio を算出（常時）
- 相場2: 収集件数が十分な回は、分位回帰(Quantile GB)のOOFで p50(中央値) / p20(下位20%) を推定
    * お得度 = p50 - 実売価格 でスコア/緊急度を後押し
    * 実売価格 ≤ p20 なら強い割安としてさらにブースト
- SUVのみ拾うための include_keywords / 除外 exclude_keywords（デフォで「ハスラー/HUSTLER」は常時除外）
- 上位を results.csv に保存
- Discord通知は二段構え：
    * 🚀 即買いレベル … MAIN チャンネルへ（既定：緊急度≧4）
    * 🤔 ありかもレベル … MAYBE チャンネルへ（既定：緊急度=3 または スコア70〜84.9）

環境変数:
  DISCORD_WEBHOOK_URL_MAIN  … 即買いレベルの通知先（必須推奨）
  DISCORD_WEBHOOK_URL_MAYBE … ありかもレベルの通知先（任意）
  DISCORD_WEBHOOK_URL       … 旧単一Webhook名（MAINが未設定の時のフォールバック）
  TARGETS_JSON              … 監視ターゲット上書き（任意のJSON文字列）
  DISCORD_DRY_RUN           … "1" なら通知せず、送信内容をコンソールに出力（テスト用）

  IMMEDIATE_URGENCY_MIN     … 即買いレベルの緊急度しきい値（デフォルト "4"）
  MAYBE_SCORE_MIN           … ありかもレベルのスコア下限（デフォルト "70"）
  MAYBE_SCORE_MAX           … ありかもレベルのスコア上限（デフォルト "84.9"）
"""
from __future__ import annotations
import os, re, csv, json, time, math
from datetime import datetime
from typing import List, Dict, Any
from unicodedata import normalize

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

# --------- SUV監視ターゲット（必要に応じて TARGETS_JSON で上書き推奨） ---------
# include_keywords は SUVモデル名をざっくり網羅（表記ゆれは正規化で吸収）
DEFAULT_TARGETS = [
    {
        "name": "トヨタSUV",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/bTO/index.html?SORT=22",
        "price_max": 4500000,
        "year_min": 2014,
        "mileage_max": 120000,
        "pages": 1,
        "include_keywords": [
            "ハリアー","RAV4","C-HR","カローラクロス","ヤリスクロス",
            "ランドクルーザー","プラド","ライズ"
        ],
        "exclude_keywords": []
    },
    {
        "name": "スバルSUV",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/bSU/index.html?SORT=22",
        "price_max": 3800000,
        "year_min": 2013,
        "mileage_max": 140000,
        "pages": 1,
        "include_keywords": ["フォレスター","アウトバック","レガシィアウトバック","XV","CROSSTREK","クロストレック"],
        "exclude_keywords": []
    },
    {
        "name": "マツダSUV",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/bMA/index.html?SORT=22",
        "price_max": 3600000,
        "year_min": 2015,
        "mileage_max": 120000,
        "pages": 1,
        "include_keywords": ["CX-3","CX-30","CX-5","CX-8","CX-60"],
        "exclude_keywords": []
    },
    {
        "name": "日産SUV",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/",
        "price_max": 3800000,
        "year_min": 2013,
        "mileage_max": 140000,
        "pages": 1,
        "include_keywords": ["エクストレイル","キックス","ジューク","テラノ","ムラーノ"],
        "exclude_keywords": []
    },
    {
        "name": "ホンダSUV",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/",
        "price_max": 4200000,
        "year_min": 2014,
        "mileage_max": 130000,
        "pages": 1,
        "include_keywords": ["ヴェゼル","VEZEL","CR-V"],
        "exclude_keywords": []
    },
    {
        "name": "三菱SUV",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/",
        "price_max": 3800000,
        "year_min": 2013,
        "mileage_max": 140000,
        "pages": 1,
        "include_keywords": ["アウトランダー","RVR","パジェロ"],
        "exclude_keywords": []
    },
    {
        "name": "スズキSUV（ハスラー除外）",
        "site": "carsensor",
        "url": "https://www.carsensor.net/usedcar/",
        "price_max": 3000000,
        "year_min": 2015,
        "mileage_max": 120000,
        "pages": 1,
        "include_keywords": ["ジムニー","ジムニーシエラ","エスクード","クロスビー"],
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

# --------- polite fetch ---------
def fetch(url: str) -> str:
    time.sleep(1.2)  # 負荷配慮（必要に応じてランダム化してもOK）
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text

# --------- サイト別パーサ ---------
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

# --------- タイトル正規化 & キーワードフィルタ（SUVだけ/ハスラー除外） ---------
def _norm(s: str) -> str:
    return normalize("NFKC", (s or "")).upper()  # 全角→半角/記号正規化＋大文字化

def keyword_filter(items, include_keywords=None, exclude_keywords=None):
    inc = [_norm(k) for k in (include_keywords or [])]
    # デフォでハスラー/HUSTLERを除外に追加
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
        score = min(100, scor

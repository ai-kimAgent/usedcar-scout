#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUV特化型 Used-Car Scout（精密判定版 / Discord二段階通知）
改良点:
- SUV判定ロジックを大幅強化（車種DB + タイトル解析 + 詳細ページ確認）
- 軽自動車（ハスラー等）の確実な除外
- 車種情報の詳細取得と検証
- より正確な相場分析

環境変数:
  DISCORD_WEBHOOK_URL_MAIN  … 即買いレベルの通知先（必須推奨）
  DISCORD_WEBHOOK_URL_MAYBE … ありかもレベルの通知先（任意）
  DISCORD_DRY_RUN           … "1" なら通知せず、送信内容をコンソールに出力
  IMMEDIATE_URGENCY_MIN     … 即買いレベルの緊急度しきい値（デフォルト "4"）
  MAYBE_SCORE_MIN           … ありかもレベルのスコア下限（デフォルト "70"）
"""
from __future__ import annotations
import os, re, csv, json, time, math, random
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from unicodedata import normalize
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

# ========== 通信設定 ==========
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
]

def get_headers():
    return {
        "User-Agent": random.choice(UA_LIST),
        "Accept-Language": "ja,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

# ========== SUV車種データベース ==========
@dataclass
class VehicleModel:
    """車種情報"""
    maker: str
    model: str
    name_variants: Set[str] = field(default_factory=set)
    is_suv: bool = True
    is_kei: bool = False  # 軽自動車フラグ
    body_types: Set[str] = field(default_factory=set)
    
SUV_DATABASE = {
    # トヨタ
    "ハリアー": VehicleModel("トヨタ", "ハリアー", {"HARRIER", "harrier"}, True, False, {"SUV", "クロスオーバーSUV"}),
    "RAV4": VehicleModel("トヨタ", "RAV4", {"ラヴフォー", "ラブフォー"}, True, False, {"SUV", "クロスオーバーSUV"}),
    "ランドクルーザー": VehicleModel("トヨタ", "ランドクルーザー", {"LANDCRUISER", "ランクル", "LC"}, True, False, {"SUV", "クロカン"}),
    "ランドクルーザープラド": VehicleModel("トヨタ", "プラド", {"PRADO", "ランクルプラド"}, True, False, {"SUV", "クロカン"}),
    "C-HR": VehicleModel("トヨタ", "C-HR", {"CHR", "chr"}, True, False, {"SUV", "コンパクトSUV"}),
    "カローラクロス": VehicleModel("トヨタ", "カローラクロス", {"COROLLA CROSS"}, True, False, {"SUV"}),
    "ヤリスクロス": VehicleModel("トヨタ", "ヤリスクロス", {"YARIS CROSS"}, True, False, {"SUV", "コンパクトSUV"}),
    "ライズ": VehicleModel("トヨタ", "ライズ", {"RAIZE"}, True, False, {"SUV", "コンパクトSUV"}),
    "ハイラックス": VehicleModel("トヨタ", "ハイラックス", {"HILUX"}, True, False, {"ピックアップトラック", "SUV"}),
    
    # 日産
    "エクストレイル": VehicleModel("日産", "エクストレイル", {"X-TRAIL", "XTRAIL"}, True, False, {"SUV"}),
    "キックス": VehicleModel("日産", "キックス", {"KICKS", "e-POWER"}, True, False, {"SUV", "コンパクトSUV"}),
    "ジューク": VehicleModel("日産", "ジューク", {"JUKE"}, True, False, {"SUV", "コンパクトSUV"}),
    "ムラーノ": VehicleModel("日産", "ムラーノ", {"MURANO"}, True, False, {"SUV"}),
    "テラノ": VehicleModel("日産", "テラノ", {"TERRANO"}, True, False, {"SUV", "クロカン"}),
    "アリア": VehicleModel("日産", "アリア", {"ARIYA"}, True, False, {"SUV", "電気自動車"}),
    
    # ホンダ
    "ヴェゼル": VehicleModel("ホンダ", "ヴェゼル", {"VEZEL", "ベゼル"}, True, False, {"SUV", "コンパクトSUV"}),
    "CR-V": VehicleModel("ホンダ", "CR-V", {"CRV", "シーアールブイ"}, True, False, {"SUV"}),
    "ZR-V": VehicleModel("ホンダ", "ZR-V", {"ZRV"}, True, False, {"SUV"}),
    
    # マツダ
    "CX-3": VehicleModel("マツダ", "CX-3", {"cx3", "シーエックススリー"}, True, False, {"SUV", "コンパクトSUV"}),
    "CX-30": VehicleModel("マツダ", "CX-30", {"cx30", "シーエックスサーティー"}, True, False, {"SUV"}),
    "CX-5": VehicleModel("マツダ", "CX-5", {"cx5", "シーエックスファイブ"}, True, False, {"SUV"}),
    "CX-8": VehicleModel("マツダ", "CX-8", {"cx8", "シーエックスエイト"}, True, False, {"SUV", "3列シート"}),
    "CX-60": VehicleModel("マツダ", "CX-60", {"cx60"}, True, False, {"SUV"}),
    "MX-30": VehicleModel("マツダ", "MX-30", {"mx30"}, True, False, {"SUV", "電動"}),
    
    # スバル
    "フォレスター": VehicleModel("スバル", "フォレスター", {"FORESTER"}, True, False, {"SUV"}),
    "XV": VehicleModel("スバル", "XV", {"CROSSTREK", "クロストレック"}, True, False, {"SUV", "クロスオーバー"}),
    "レガシィアウトバック": VehicleModel("スバル", "アウトバック", {"OUTBACK", "レガシィ"}, True, False, {"SUV", "クロスオーバー"}),
    "アセント": VehicleModel("スバル", "アセント", {"ASCENT"}, True, False, {"SUV", "3列シート"}),
    
    # 三菱
    "アウトランダー": VehicleModel("三菱", "アウトランダー", {"OUTLANDER", "PHEV"}, True, False, {"SUV"}),
    "エクリプスクロス": VehicleModel("三菱", "エクリプスクロス", {"ECLIPSE CROSS"}, True, False, {"SUV"}),
    "RVR": VehicleModel("三菱", "RVR", {"アールブイアール"}, True, False, {"SUV", "コンパクトSUV"}),
    "パジェロ": VehicleModel("三菱", "パジェロ", {"PAJERO"}, True, False, {"SUV", "クロカン"}),
    
    # スズキ（軽自動車は除外対象）
    "ジムニー": VehicleModel("スズキ", "ジムニー", {"JIMNY"}, True, True, {"軽自動車", "クロカン"}),
    "ジムニーシエラ": VehicleModel("スズキ", "ジムニーシエラ", {"JIMNY SIERRA"}, True, False, {"SUV", "クロカン"}),
    "エスクード": VehicleModel("スズキ", "エスクード", {"ESCUDO"}, True, False, {"SUV"}),
    "クロスビー": VehicleModel("スズキ", "クロスビー", {"XBEE", "CROSSBEE"}, True, False, {"SUV", "コンパクトSUV"}),
    "ハスラー": VehicleModel("スズキ", "ハスラー", {"HUSTLER"}, False, True, {"軽自動車", "軽SUV"}),
    "スペーシアギア": VehicleModel("スズキ", "スペーシアギア", {"SPACIA GEAR"}, False, True, {"軽自動車"}),
    
    # ダイハツ（軽自動車）
    "タフト": VehicleModel("ダイハツ", "タフト", {"TAFT"}, False, True, {"軽自動車", "軽SUV"}),
    "ロッキー": VehicleModel("ダイハツ", "ロッキー", {"ROCKY"}, True, False, {"SUV", "コンパクトSUV"}),
    "テリオスキッド": VehicleModel("ダイハツ", "テリオスキッド", {"TERIOS KID"}, False, True, {"軽自動車"}),
}

# 軽自動車の確実な除外リスト
KEI_CAR_KEYWORDS = {
    "ハスラー", "HUSTLER", "タフト", "TAFT", "スペーシアギア", "SPACIA GEAR",
    "テリオスキッド", "TERIOS KID", "キャスト", "CAST", "アクティバ", "ACTIVA",
    "ウェイク", "WAKE", "軽自動車", "軽SUV", "K-CAR", "660cc", "660CC"
}

# ========== 車種判定エンジン ==========
class VehicleClassifier:
    """車種判定クラス"""
    
    def __init__(self):
        self.suv_patterns = self._compile_patterns()
        self.kei_patterns = re.compile(
            r'(軽自動車|軽SUV|660cc|660CC|K-?CAR)', 
            re.IGNORECASE
        )
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """SUVパターンをコンパイル"""
        patterns = {}
        for key, model in SUV_DATABASE.items():
            if model.is_suv and not model.is_kei:
                # メインの車種名と別名をパターン化
                all_names = {key, model.model} | model.name_variants
                pattern_str = '|'.join(re.escape(name) for name in all_names)
                patterns[key] = re.compile(pattern_str, re.IGNORECASE)
        return patterns
    
    def classify(self, text: str, detailed_text: str = "") -> Tuple[bool, str, float]:
        """
        テキストからSUV判定
        Returns: (is_suv, model_name, confidence)
        """
        normalized = normalize("NFKC", text.upper())
        detailed_norm = normalize("NFKC", detailed_text.upper()) if detailed_text else ""
        
        # 軽自動車チェック（除外）
        if self._is_kei_car(normalized + " " + detailed_norm):
            return False, "軽自動車", 0.0
        
        # SUVモデル検出
        for model_key, pattern in self.suv_patterns.items():
            if pattern.search(text) or (detailed_text and pattern.search(detailed_text)):
                model = SUV_DATABASE[model_key]
                confidence = self._calculate_confidence(text, detailed_text, model)
                return True, model_key, confidence
        
        # SUV関連キーワードチェック（弱い判定）
        suv_keywords = {"SUV", "クロスオーバー", "クロカン", "4WD", "AWD", "オフロード"}
        if any(kw in normalized for kw in suv_keywords):
            return True, "不明SUV", 0.3
        
        return False, "", 0.0
    
    def _is_kei_car(self, text: str) -> bool:
        """軽自動車判定"""
        for kw in KEI_CAR_KEYWORDS:
            if kw.upper() in text:
                return True
        if self.kei_patterns.search(text):
            return True
        return False
    
    def _calculate_confidence(self, title: str, detail: str, model: VehicleModel) -> float:
        """信頼度計算"""
        confidence = 0.7  # ベース信頼度
        
        # メーカー名が含まれていれば信頼度UP
        if model.maker in title or model.maker in detail:
            confidence += 0.15
        
        # ボディタイプが一致すれば信頼度UP  
        for body_type in model.body_types:
            if body_type in title or body_type in detail:
                confidence += 0.1
                break
        
        # 複数の別名が含まれていれば信頼度UP
        variant_count = sum(1 for v in model.name_variants if v.upper() in title.upper())
        confidence += min(variant_count * 0.05, 0.15)
        
        return min(confidence, 1.0)

# ========== 詳細ページ取得 ==========
def fetch_with_retry(url: str, max_retries: int = 2) -> Optional[str]:
    """リトライ付きフェッチ"""
    for attempt in range(max_retries + 1):
        try:
            time.sleep(random.uniform(1.0, 2.0))  # ランダム待機
            r = requests.get(url, headers=get_headers(), timeout=15)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] Failed to fetch {url}: {e}")
                return None
            time.sleep(2 ** attempt)  # 指数バックオフ
    return None

def extract_vehicle_details(url: str, site: str) -> Dict[str, Any]:
    """詳細ページから車両情報を抽出"""
    html = fetch_with_retry(url)
    if not html:
        return {}
    
    soup = BeautifulSoup(html, "lxml")
    details = {}
    
    if site == "carsensor":
        # カーセンサーの詳細情報取得
        details["body_type"] = _extract_text(soup, ".specWrap th:contains('ボディタイプ') + td")
        details["model_year"] = _extract_text(soup, ".specWrap th:contains('年式') + td")
        details["grade"] = _extract_text(soup, ".specWrap th:contains('グレード') + td")
        details["engine"] = _extract_text(soup, ".specWrap th:contains('排気量') + td")
        details["drive_type"] = _extract_text(soup, ".specWrap th:contains('駆動方式') + td")
        details["color"] = _extract_text(soup, ".specWrap th:contains('車体色') + td")
        details["equipment"] = _extract_equipment_carsensor(soup)
        
    elif site == "goonet":
        # Goo-netの詳細情報取得
        details["body_type"] = _extract_text(soup, "th:contains('ボディタイプ') + td")
        details["model_year"] = _extract_text(soup, "th:contains('年式') + td")
        details["grade"] = _extract_text(soup, "th:contains('グレード') + td")
        details["engine"] = _extract_text(soup, "th:contains('排気量') + td")
        details["drive_type"] = _extract_text(soup, "th:contains('駆動') + td")
        details["equipment"] = _extract_equipment_goonet(soup)
    
    # 説明文から追加情報
    description = soup.get_text(" ", strip=True)[:2000]  # 最初の2000文字
    details["description"] = description
    
    return details

def _extract_text(soup, selector: str) -> str:
    """セレクタからテキスト抽出"""
    elem = soup.select_one(selector)
    return elem.get_text(strip=True) if elem else ""

def _extract_equipment_carsensor(soup) -> List[str]:
    """カーセンサーから装備抽出"""
    equipment = []
    equip_section = soup.select(".equipment li, .equipmentList li")
    for item in equip_section[:20]:  # 最大20個
        equipment.append(item.get_text(strip=True))
    return equipment

def _extract_equipment_goonet(soup) -> List[str]:
    """Goo-netから装備抽出"""
    equipment = []
    equip_section = soup.select(".equipment span, .icon-list li")
    for item in equip_section[:20]:
        equipment.append(item.get_text(strip=True))
    return equipment

# ========== パーサー改良版 ==========
def parse_carsensor_list_enhanced(html: str) -> List[Dict[str, Any]]:
    """カーセンサーのリストページ解析（強化版）"""
    soup = BeautifulSoup(html, "lxml")
    items = []
    classifier = VehicleClassifier()
    
    # 車両カードを取得
    cards = soup.select(".cassette__inner, .cassetteMain, .js-listTableCassette")
    if not cards:
        cards = soup.select("article, .itemBox")
    
    for card in cards:
        try:
            # 基本情報取得
            title_elem = card.select_one("h3 a, .cassetteMain__title a, h2 a")
            if not title_elem:
                continue
                
            title = title_elem.get_text(strip=True)
            url = title_elem.get("href", "")
            if url.startswith("/"):
                url = "https://www.carsensor.net" + url
            
            # 車種判定（第1段階）
            is_suv, model_name, confidence = classifier.classify(title)
            if not is_suv or confidence < 0.3:
                continue  # SUVでない or 信頼度が低い
            
            # 価格・年式・走行距離の抽出
            text = card.get_text(" ", strip=True)
            price = _extract_price(text)
            year = _extract_year(text)
            mileage = _extract_mileage(text)
            
            # ボディタイプの確認（可能な場合）
            body_type_elem = card.select_one(".cassetteMain__etc span:contains('SUV'), .bodyType")
            body_type = body_type_elem.get_text(strip=True) if body_type_elem else ""
            
            # 修復歴チェック
            has_repair = "修復歴あり" in text or "R" in text
            
            # グレード情報
            grade_elem = card.select_one(".cassetteMain__grade, .grade")
            grade = grade_elem.get_text(strip=True) if grade_elem else ""
            
            items.append({
                "title": title,
                "url": url,
                "year": year,
                "mileage": mileage,
                "price": price,
                "site": "carsensor",
                "model_name": model_name,
                "confidence": confidence,
                "body_type": body_type,
                "grade": grade,
                "has_repair": has_repair,
                "raw_text": text[:500]  # デバッグ用
            })
            
        except Exception as e:
            print(f"[WARN] Parse error: {e}")
            continue
    
    return items

def parse_goonet_list_enhanced(html: str) -> List[Dict[str, Any]]:
    """Goo-netのリストページ解析（強化版）"""
    soup = BeautifulSoup(html, "lxml")
    items = []
    classifier = VehicleClassifier()
    
    # 車両要素を取得
    cars = soup.select(".car-list-unit, .used-car-list-wrap article, .item-wrap")
    
    for car in cars:
        try:
            title_elem = car.select_one("h3 a, h2 a, .item-name a")
            if not title_elem:
                continue
                
            title = title_elem.get_text(strip=True)
            url = title_elem.get("href", "")
            if not url.startswith("http"):
                url = "https://www.goo-net.com" + url
            
            # SUV判定
            is_suv, model_name, confidence = classifier.classify(title)
            if not is_suv or confidence < 0.3:
                continue
            
            text = car.get_text(" ", strip=True)
            price = _extract_price(text)
            year = _extract_year(text)
            mileage = _extract_mileage(text)
            
            # グレード・色情報
            spec_elem = car.select_one(".spec-wrap, .item-spec")
            grade = spec_elem.get_text(strip=True) if spec_elem else ""
            
            items.append({
                "title": title,
                "url": url,
                "year": year,
                "mileage": mileage,
                "price": price,
                "site": "goonet",
                "model_name": model_name,
                "confidence": confidence,
                "grade": grade,
                "raw_text": text[:500]
            })
            
        except Exception as e:
            print(f"[WARN] Parse error: {e}")
            continue
    
    return items

# ========== 数値抽出ヘルパー ==========
def _extract_price(text: str) -> int:
    """価格抽出（改良版）"""
    patterns = [
        r"([0-9,]+(?:\.[0-9]+)?)\s*万円",
        r"￥([0-9,]+)",
        r"([0-9,]+)円"
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            val = m.group(1).replace(",", "")
            if "万円" in m.group(0):
                return int(float(val) * 10000)
            return int(float(val))
    return 0

def _extract_year(text: str) -> int:
    """年式抽出"""
    patterns = [
        r"(\d{4})年式",
        r"(\d{4})年",
        r"H(\d{2})年",  # 平成
        r"R(\d{1,2})年"  # 令和
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            if pattern.startswith("H"):
                return 1988 + int(m.group(1))  # 平成変換
            elif pattern.startswith("R"):
                return 2018 + int(m.group(1))  # 令和変換
            else:
                year = int(m.group(1))
                if 2000 <= year <= 2030:
                    return year
    return 0

def _extract_mileage(text: str) -> int:
    """走行距離抽出"""
    patterns = [
        r"([0-9.]+)\s*万\s*km",
        r"([0-9,]+)\s*km"
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            if "万" in m.group(0):
                return int(float(val) * 10000)
            return int(float(val))
    return 0

# ========== 評価エンジン（改良版） ==========
def compute_enhanced_score(item: Dict[str, Any], market_stats: Dict, config: Dict) -> None:
    """強化版スコアリング"""
    price = item.get("price", 0)
    year = item.get("year", 0)
    mileage = item.get("mileage", 0)
    confidence = item.get("confidence", 0.5)
    
    if not price:
        item["score"] = 0
        item["urgency"] = 0
        return
    
    # 基本スコア計算
    base_score = 50
    
    # 価格評価
    median_price = market_stats.get("median", 0)
    if median_price:
        price_ratio = price / median_price
        if price_ratio <= 0.6:
            base_score = 95
        elif price_ratio <= 0.7:
            base_score = 85
        elif price_ratio <= 0.8:
            base_score = 75
        elif price_ratio <= 0.9:
            base_score = 65
        item["price_ratio"] = round(price_ratio, 2)
    
    # 年式評価
    current_year = datetime.now().year
    age = current_year - year if year else 99
    if age <= 3:
        base_score += 15
    elif age <= 5:
        base_score += 10
    elif age <= 8:
        base_score += 5
    elif age >= 15:
        base_score -= 10
    
    # 走行距離評価
    if mileage:
        annual_mileage = mileage / max(age, 1)
        if annual_mileage <= 5000:
            base_score += 15
        elif annual_mileage <= 8000:
            base_score += 10
        elif annual_mileage <= 12000:
            base_score += 5
        elif annual_mileage >= 20000:
            base_score -= 10
    
    # SUV信頼度による調整
    base_score = base_score * (0.7 + 0.3 * confidence)
    
    # 装備による加点
    title = item.get("title", "")
    premium_keywords = [
        "サンルーフ", "レザー", "革シート", "BOSE", "JBL", 
        "マークレビンソン", "360度", "プロパイロット", "アイサイト",
        "ハンズフリー", "電動リアゲート", "マトリクスLED"
    ]
    for kw in premium_keywords:
        if kw in title:
            base_score += 2
    
    # 修復歴による減点
    if item.get("has_repair"):
        base_score -= 25
    
    # 予測価格との差額評価
    if "pred_p50" in item and item["pred_p50"]:
        gap = item["pred_p50"] - price
        if gap > 500000:
            base_score += 10
        elif gap > 300000:
            base_score += 7
        elif gap > 100000:
            base_score += 4
        item["deal_gap"] = int(gap)
    
    # スコア正規化
    score = max(0, min(100, base_score))
    
    # 緊急度判定
    if score >= 90:
        urgency = 5
    elif score >= 80:
        urgency = 4
    elif score >= 70:
        urgency = 3
    elif score >= 60:
        urgency = 2
    else:
        urgency = 1
    
    # 特別条件による緊急度ブースト
    if median_price and price <= median_price * 0.6:
        urgency = min(5, urgency + 1)
    
    item["score"] = round(score, 1)
    item["urgency"] = urgency

# ========== 分位回帰予測 ==========
def build_features(item: Dict[str, Any]) -> List[float]:
    """特徴量構築（拡張版）"""
    title = item.get("title", "")
    current_year = datetime.now().year
    
    features = [
        item.get("year", 0),
        item.get("mileage", 0),
        current_year - item.get("year", current_year) if item.get("year") else 15,  # 車齢
        item.get("confidence", 0.5),  # SUV判定信頼度
        
        # 装備フラグ
        1 if "サンルーフ" in title else 0,
        1 if any(kw in title for kw in ["レザー", "革シート", "本革"]) else 0,
        1 if any(kw in title for kw in ["BOSE", "JBL", "マークレビンソン"]) else 0,
        1 if any(kw in title for kw in ["4WD", "AWD", "四駆"]) else 0,
        1 if "ハイブリッド" in title else 0,
        1 if "ターボ" in title else 0,
        1 if item.get("has_repair", False) else 0,
        
        # モデル別ダミー変数（主要モデル）
        1 if "ハリアー" in title else 0,
        1 if "RAV4" in title else 0,
        1 if "CX-5" in title else 0,
        1 if "フォレスター" in title else 0,
        1 if "エクストレイル" in title else 0,
    ]
    
    return features

def predict_quantiles(items: List[Dict[str, Any]], quantiles=(0.5, 0.2)) -> Dict[float, np.ndarray]:
    """分位回帰による価格予測"""
    if len(items) < 20:
        return {q: np.array([None] * len(items)) for q in quantiles}
    
    X = np.array([build_features(item) for item in items])
    y = np.array([item.get("price", 0) for item in items])
    
    # 価格が0の項目を除外
    valid_mask = y > 0
    if valid_mask.sum() < 15:
        return {q: np.array([None] * len(items)) for q in quantiles}
    
    predictions = {}
    
    for q in quantiles:
        preds = np.full(len(items), np.nan)
        kf = KFold(n_splits=min(5, valid_mask.sum() // 3), shuffle=True, random_state=42)
        
        try:
            for train_idx, val_idx in kf.split(X[valid_mask]):
                # 有効なデータのインデックスを取得
                valid_indices = np.where(valid_mask)[0]
                train_indices = valid_indices[train_idx]
                val_indices = valid_indices[val_idx]
                
                model = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=q,
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model.fit(X[train_indices], y[train_indices])
                preds[val_indices] = model.predict(X[val_indices])
            
        except Exception as e:
            print(f"[WARN] Quantile regression failed: {e}")
            return {q: np.array([None] * len(items)) for q in quantiles}
        
        predictions[q] = preds
    
    return predictions

# ========== Discord通知（改良版） ==========
def send_discord_notification(items: List[Dict[str, Any]], webhook_url: str, category: str):
    """Discord通知送信"""
    if not items:
        print(f"[INFO] {category}: 通知対象なし")
        return
    
    if os.getenv("DISCORD_DRY_RUN", "0") == "1":
        print(f"[DRY-RUN] {category} 通知:")
        for item in items[:3]:
            print(f"  - {item['title'][:50]}... Score:{item['score']} Price:{item['price']:,}円")
        return
    
    if not webhook_url:
        print(f"[WARN] {category}: Webhook URL未設定")
        return
    
    embeds = []
    for item in items[:5]:
        price = item.get("price", 0)
        year = item.get("year", 0)
        mileage = item.get("mileage", 0)
        
        fields = [
            {"name": "価格", "value": f"{price:,}円", "inline": True},
            {"name": "年式", "value": f"{year}年", "inline": True},
            {"name": "走行距離", "value": f"{mileage:,}km", "inline": True},
            {"name": "スコア", "value": f"{item.get('score', 0):.1f}", "inline": True},
            {"name": "緊急度", "value": "🔥" * item.get("urgency", 1), "inline": True},
            {"name": "SUV判定", "value": f"{item.get('model_name', '不明')} ({item.get('confidence', 0):.0%})", "inline": True},
        ]
        
        if item.get("price_ratio"):
            fields.append({"name": "相場比", "value": f"{item['price_ratio']:.0%}", "inline": True})
        
        if item.get("deal_gap"):
            fields.append({"name": "予測差額", "value": f"+{item['deal_gap']:,}円", "inline": True})
        
        color = 0xFF0000 if item.get("urgency", 1) >= 4 else 0xFFAA00 if item.get("urgency", 1) >= 3 else 0x00AA00
        
        embeds.append({
            "title": item.get("title", "不明")[:256],
            "url": item.get("url", ""),
            "color": color,
            "fields": fields,
            "footer": {"text": f"{item.get('site', '')} | {item.get('grade', '')}"}
        })
    
    payload = {
        "content": f"**{category}** - {datetime.now():%Y/%m/%d %H:%M}",
        "embeds": embeds
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"[SUCCESS] {category}: {len(items)}件通知完了")
    except Exception as e:
        print(f"[ERROR] Discord通知失敗 ({category}): {e}")

# ========== 監視設定 ==========
MONITORING_CONFIGS = [
    {
        "name": "トヨタSUV",
        "site": "carsensor",
        "base_url": "https://www.carsensor.net/usedcar/bTO/index.html",
        "params": {
            "SORT": "22",  # 新着順
            "SHASHU": "ハリアー|RAV4|ランドクルーザー|C-HR|カローラクロス|ヤリスクロス"
        },
        "price_max": 5000000,
        "year_min": 2015,
        "mileage_max": 100000,
        "pages": 2
    },
    {
        "name": "マツダSUV",
        "site": "carsensor",
        "base_url": "https://www.carsensor.net/usedcar/bMA/index.html",
        "params": {
            "SORT": "22",
            "BODY": "SUV"
        },
        "price_max": 4000000,
        "year_min": 2016,
        "mileage_max": 80000,
        "pages": 2
    },
    {
        "name": "スバルSUV",
        "site": "carsensor",
        "base_url": "https://www.carsensor.net/usedcar/bSU/index.html",
        "params": {
            "SORT": "22",
            "SHASHU": "フォレスター|XV|レガシィアウトバック|アウトバック"
        },
        "price_max": 4000000,
        "year_min": 2015,
        "mileage_max": 90000,
        "pages": 2
    },
    {
        "name": "日産・ホンダSUV",
        "site": "carsensor",
        "base_url": "https://www.carsensor.net/usedcar/index.html",
        "params": {
            "MAKER": "NI|HO",  # 日産・ホンダ
            "BODY": "SUV",
            "SORT": "22"
        },
        "price_max": 4500000,
        "year_min": 2015,
        "mileage_max": 100000,
        "pages": 2
    },
    {
        "name": "Goo-net SUV",
        "site": "goonet",
        "base_url": "https://www.goo-net.com/usedcar/",
        "params": {
            "body_type": "suv",
            "sort": "update_desc"
        },
        "price_max": 5000000,
        "year_min": 2015,
        "mileage_max": 100000,
        "pages": 1
    }
]

# ========== メイン処理 ==========
def main():
    """メイン処理"""
    print("=" * 60)
    print("SUV特化型中古車スカウト - 起動")
    print(f"実行時刻: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    
    # 環境変数読み込み
    webhook_main = os.getenv("DISCORD_WEBHOOK_URL_MAIN") or os.getenv("DISCORD_WEBHOOK_URL")
    webhook_maybe = os.getenv("DISCORD_WEBHOOK_URL_MAYBE")
    immediate_min = int(os.getenv("IMMEDIATE_URGENCY_MIN", "4"))
    maybe_score_min = float(os.getenv("MAYBE_SCORE_MIN", "70"))
    maybe_score_max = float(os.getenv("MAYBE_SCORE_MAX", "84.9"))
    
    # カスタム設定があれば読み込み
    custom_config = os.getenv("TARGETS_JSON")
    if custom_config:
        try:
            MONITORING_CONFIGS.clear()
            MONITORING_CONFIGS.extend(json.loads(custom_config))
            print(f"[INFO] カスタム設定を読み込みました")
        except Exception as e:
            print(f"[ERROR] TARGETS_JSON解析エラー: {e}")
    
    all_vehicles = []
    classifier = VehicleClassifier()
    
    # 各設定でスクレイピング
    for config in MONITORING_CONFIGS:
        print(f"\n[処理中] {config['name']}")
        
        site = config["site"]
        if site == "carsensor":
            parser = parse_carsensor_list_enhanced
        elif site == "goonet":
            parser = parse_goonet_list_enhanced
        else:
            print(f"[SKIP] 未対応サイト: {site}")
            continue
        
        vehicles = []
        
        for page in range(1, config.get("pages", 1) + 1):
            # URL構築
            url = config["base_url"]
            if config.get("params"):
                param_str = "&".join([f"{k}={v}" for k, v in config["params"].items()])
                url += ("&" if "?" in url else "?") + param_str
            if page > 1:
                url += f"&page={page}"
            
            print(f"  取得中: {url}")
            html = fetch_with_retry(url)
            if not html:
                continue
            
            # パース
            items = parser(html)
            print(f"  → {len(items)}件取得")
            
            # フィルタリング
            filtered = []
            for item in items:
                # 価格・年式・走行距離フィルタ
                if item.get("price", 0) > config.get("price_max", 9999999):
                    continue
                if item.get("year", 0) < config.get("year_min", 0):
                    continue
                if item.get("mileage", 0) > config.get("mileage_max", 9999999):
                    continue
                
                # SUV確認（詳細チェック）
                if item.get("confidence", 0) < 0.5:
                    # 信頼度が低い場合は詳細ページを確認
                    details = extract_vehicle_details(item["url"], site)
                    if details:
                        detailed_text = details.get("description", "") + " " + details.get("body_type", "")
                        is_suv, model, conf = classifier.classify(item["title"], detailed_text)
                        
                        if is_suv and conf >= 0.5:
                            item["model_name"] = model
                            item["confidence"] = conf
                            item.update(details)
                        else:
                            continue  # SUVでない
                
                filtered.append(item)
            
            vehicles.extend(filtered)
            print(f"  → フィルタ後: {len(filtered)}件")
        
        if not vehicles:
            print(f"  → 該当車両なし")
            continue
        
        # 市場統計計算
        prices = [v["price"] for v in vehicles if v.get("price", 0) > 0]
        if len(prices) >= 5:
            market_stats = {
                "median": np.median(prices),
                "q25": np.percentile(prices, 25),
                "q75": np.percentile(prices, 75),
                "mean": np.mean(prices),
                "std": np.std(prices)
            }
            print(f"  市場統計: 中央値 {market_stats['median']:,.0f}円")
        else:
            market_stats = {}
        
        # 予測価格計算
        if len(vehicles) >= 20:
            predictions = predict_quantiles(vehicles)
            for i, vehicle in enumerate(vehicles):
                vehicle["pred_p50"] = predictions[0.5][i] if not np.isnan(predictions[0.5][i]) else None
                vehicle["pred_p20"] = predictions[0.2][i] if not np.isnan(predictions[0.2][i]) else None
        
        # スコア計算
        for vehicle in vehicles:
            compute_enhanced_score(vehicle, market_stats, config)
        
        all_vehicles.extend(vehicles)
    
    # 全体でソート
    all_vehicles.sort(key=lambda x: (x.get("urgency", 0), x.get("score", 0)), reverse=True)
    
    # CSV出力
    print("\n[CSV出力]")
    with open("suv_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "タイトル", "URL", "サイト", "モデル", "信頼度",
            "価格", "年式", "走行距離", "スコア", "緊急度",
            "相場比", "予測差額", "グレード"
        ])
        
        for v in all_vehicles[:30]:
            writer.writerow([
                v.get("title", ""),
                v.get("url", ""),
                v.get("site", ""),
                v.get("model_name", ""),
                f"{v.get('confidence', 0):.0%}",
                v.get("price", 0),
                v.get("year", 0),
                v.get("mileage", 0),
                v.get("score", 0),
                v.get("urgency", 0),
                f"{v.get('price_ratio', 0):.0%}" if v.get("price_ratio") else "",
                v.get("deal_gap", ""),
                v.get("grade", "")
            ])
    print(f"  → suv_results.csv に上位30件を保存")
    
    # Discord通知準備
    immediate_items = [v for v in all_vehicles if v.get("urgency", 0) >= immediate_min]
    maybe_items = [v for v in all_vehicles 
                   if v.get("urgency", 0) == 3 or 
                   (maybe_score_min <= v.get("score", 0) <= maybe_score_max)]
    
    # 重複除去
    immediate_urls = {v["url"] for v in immediate_items}
    maybe_items = [v for v in maybe_items if v["url"] not in immediate_urls]
    
    # 通知送信
    print("\n[Discord通知]")
    send_discord_notification(immediate_items[:5], webhook_main, "🚀 即買いレベル SUV")
    send_discord_notification(maybe_items[:5], webhook_maybe, "🤔 検討価値あり SUV")
    
    print("\n[完了]")
    print(f"  即買い候補: {len(immediate_items)}件")
    print(f"  検討候補: {len(maybe_items)}件")
    print(f"  総取得数: {len(all_vehicles)}件")

if __name__ == "__main__":
    main()

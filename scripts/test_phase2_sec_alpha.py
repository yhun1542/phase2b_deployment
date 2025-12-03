#!/usr/bin/env python3
"""
Phase 2 SEC-API Fundamental Alpha Generation
=============================================

SEC 공시 데이터를 활용한 펀더멘탈 알파 신호 생성
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SEC_Alpha')

# API 키
SEC_API_KEY = "c2c08a95c67793b5a8bbba1e51611ed466900124e70c0615badefea2c6d429f9"
SHARADAR_API_KEY = "H6zH4Q2CDr9uTFk9koqJ"


class SECFundamentalAlpha:
    """SEC 펀더멘탈 데이터 기반 알파 생성"""
    
    def __init__(self):
        self.sec_api_key = SEC_API_KEY
        self.sharadar_api_key = SHARADAR_API_KEY
        self.base_url = "https://api.sec-api.io"
    
    def get_company_facts(self, ticker: str) -> Dict:
        """회사 재무 사실 조회"""
        
        try:
            # CIK 조회
            cik = self._get_cik(ticker)
            if not cik:
                logger.warning(f"CIK not found for {ticker}")
                return {}
            
            # 재무 데이터 조회
            url = f"{self.base_url}/company/{cik}/facts"
            params = {'api_key': self.sec_api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error fetching SEC data for {ticker}: {e}")
            return {}
    
    def _get_cik(self, ticker: str) -> Optional[str]:
        """Ticker에서 CIK 조회"""
        
        try:
            # SEC EDGAR에서 CIK 조회
            url = f"https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                'company': ticker,
                'owner': 'exclude',
                'action': 'getcompany',
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'cik_lookup' in data and data['cik_lookup']:
                cik = list(data['cik_lookup'].values())[0]
                return str(cik).zfill(10)
        
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
        
        return None
    
    def extract_financial_metrics(self, ticker: str) -> Dict:
        """주요 재무 지표 추출"""
        
        facts = self.get_company_facts(ticker)
        metrics = {}
        
        if not facts:
            return metrics
        
        try:
            # 주요 지표 추출
            us_gaap = facts.get('us-gaap', {})
            
            # 수익 (Revenue)
            if 'Revenues' in us_gaap:
                revenues = us_gaap['Revenues'].get('units', {}).get('USD', [])
                if revenues:
                    latest = revenues[-1]
                    metrics['revenue'] = latest.get('val', 0)
                    metrics['revenue_date'] = latest.get('end', '')
            
            # 순이익 (Net Income)
            if 'NetIncomeLoss' in us_gaap:
                net_income = us_gaap['NetIncomeLoss'].get('units', {}).get('USD', [])
                if net_income:
                    latest = net_income[-1]
                    metrics['net_income'] = latest.get('val', 0)
            
            # 총자산 (Total Assets)
            if 'Assets' in us_gaap:
                assets = us_gaap['Assets'].get('units', {}).get('USD', [])
                if assets:
                    latest = assets[-1]
                    metrics['total_assets'] = latest.get('val', 0)
            
            # 총부채 (Total Liabilities)
            if 'Liabilities' in us_gaap:
                liabilities = us_gaap['Liabilities'].get('units', {}).get('USD', [])
                if liabilities:
                    latest = liabilities[-1]
                    metrics['total_liabilities'] = latest.get('val', 0)
            
            # 주주자본 (Stockholders Equity)
            if 'StockholdersEquity' in us_gaap:
                equity = us_gaap['StockholdersEquity'].get('units', {}).get('USD', [])
                if equity:
                    latest = equity[-1]
                    metrics['stockholders_equity'] = latest.get('val', 0)
            
            # 영업현금흐름 (Operating Cash Flow)
            if 'OperatingActivitiesCashFlowStatement' in us_gaap:
                ocf = us_gaap['OperatingActivitiesCashFlowStatement'].get('units', {}).get('USD', [])
                if ocf:
                    latest = ocf[-1]
                    metrics['operating_cash_flow'] = latest.get('val', 0)
            
            # 자본지출 (Capital Expenditures)
            if 'PaymentsToAcquirePropertyPlantAndEquipment' in us_gaap:
                capex = us_gaap['PaymentsToAcquirePropertyPlantAndEquipment'].get('units', {}).get('USD', [])
                if capex:
                    latest = capex[-1]
                    metrics['capex'] = abs(latest.get('val', 0))
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error extracting metrics for {ticker}: {e}")
            return metrics
    
    def calculate_fundamental_ratios(self, metrics: Dict) -> Dict:
        """재무 비율 계산"""
        
        ratios = {}
        
        try:
            # ROE (Return on Equity)
            if metrics.get('net_income') and metrics.get('stockholders_equity'):
                if metrics['stockholders_equity'] > 0:
                    ratios['roe'] = metrics['net_income'] / metrics['stockholders_equity']
            
            # ROA (Return on Assets)
            if metrics.get('net_income') and metrics.get('total_assets'):
                if metrics['total_assets'] > 0:
                    ratios['roa'] = metrics['net_income'] / metrics['total_assets']
            
            # Debt-to-Equity
            if metrics.get('total_liabilities') and metrics.get('stockholders_equity'):
                if metrics['stockholders_equity'] > 0:
                    ratios['debt_to_equity'] = metrics['total_liabilities'] / metrics['stockholders_equity']
            
            # Operating Margin
            if metrics.get('net_income') and metrics.get('revenue'):
                if metrics['revenue'] > 0:
                    ratios['operating_margin'] = metrics['net_income'] / metrics['revenue']
            
            # Free Cash Flow
            if metrics.get('operating_cash_flow') and metrics.get('capex'):
                ratios['free_cash_flow'] = metrics['operating_cash_flow'] - metrics['capex']
            
            # FCF Margin
            if ratios.get('free_cash_flow') and metrics.get('revenue'):
                if metrics['revenue'] > 0:
                    ratios['fcf_margin'] = ratios['free_cash_flow'] / metrics['revenue']
            
            return ratios
        
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            return ratios
    
    def generate_alpha_signal(self, ticker: str, price: float) -> Dict:
        """펀더멘탈 알파 신호 생성"""
        
        metrics = self.extract_financial_metrics(ticker)
        ratios = self.calculate_fundamental_ratios(metrics)
        
        signal = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'ratios': ratios,
            'alpha_score': 0.0
        }
        
        try:
            # 알파 스코어 계산 (0~1)
            scores = []
            
            # ROE 점수 (높을수록 좋음)
            if 'roe' in ratios:
                roe_score = min(ratios['roe'] / 0.15, 1.0)  # 15% 기준
                scores.append(roe_score * 0.25)
            
            # ROA 점수
            if 'roa' in ratios:
                roa_score = min(ratios['roa'] / 0.10, 1.0)  # 10% 기준
                scores.append(roa_score * 0.25)
            
            # Debt-to-Equity 점수 (낮을수록 좋음)
            if 'debt_to_equity' in ratios:
                dte_score = 1.0 - min(ratios['debt_to_equity'] / 2.0, 1.0)
                scores.append(dte_score * 0.20)
            
            # FCF Margin 점수
            if 'fcf_margin' in ratios:
                fcf_score = min(ratios['fcf_margin'] / 0.10, 1.0)  # 10% 기준
                scores.append(fcf_score * 0.30)
            
            signal['alpha_score'] = sum(scores) if scores else 0.5
            
            logger.info(f"{ticker}: Alpha Score = {signal['alpha_score']:.4f}")
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating alpha signal for {ticker}: {e}")
            return signal


class PolygonMarketAlpha:
    """POLYGON 시장 데이터 기반 기술 알파 생성"""
    
    def __init__(self):
        self.polygon_api_key = "w7KprL4_lK7uutSH0dYGARkucXHOFXCN"
        self.base_url = "https://api.polygon.io"
    
    def get_daily_bars(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """일봉 데이터 조회"""
        
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {
                'apikey': self.polygon_api_key,
                'sort': 'asc'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' in data:
                results = data['results']
                
                dates = []
                opens = []
                highs = []
                lows = []
                closes = []
                volumes = []
                
                for bar in results:
                    dates.append(pd.to_datetime(bar['t'], unit='ms'))
                    opens.append(bar.get('o', 0))
                    highs.append(bar.get('h', 0))
                    lows.append(bar.get('l', 0))
                    closes.append(bar.get('c', 0))
                    volumes.append(bar.get('v', 0))
                
                df = pd.DataFrame({
                    'date': dates,
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                })
                
                logger.info(f"Fetched {len(df)} bars for {symbol}")
                return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """기술 지표 계산"""
        
        indicators = {}
        
        try:
            # RSI (14-day)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.iloc[-1] if len(rsi) > 0 else 50
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1] if len(macd) > 0 else 0
            indicators['macd_signal'] = signal_line.iloc[-1] if len(signal_line) > 0 else 0
            
            # Bollinger Bands
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            bb_position = (df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            indicators['bb_position'] = bb_position
            
            # Volume Trend
            volume_sma = df['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = df['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1.0
            
            return indicators
        
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return indicators
    
    def generate_technical_alpha(self, symbol: str, bars: pd.DataFrame) -> Dict:
        """기술 알파 신호 생성"""
        
        indicators = self.calculate_technical_indicators(bars)
        
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'indicators': indicators,
            'alpha_score': 0.0
        }
        
        try:
            # 알파 스코어 계산
            scores = []
            
            # RSI 점수 (30~70 범위가 최적)
            rsi = indicators.get('rsi', 50)
            rsi_score = 1.0 - abs(rsi - 50) / 50
            scores.append(rsi_score * 0.30)
            
            # MACD 점수
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_score = 1.0 if macd > macd_signal else 0.0
            scores.append(macd_score * 0.30)
            
            # Bollinger Bands 점수 (중간값 근처가 최적)
            bb_pos = indicators.get('bb_position', 0.5)
            bb_score = 1.0 - abs(bb_pos - 0.5)
            scores.append(bb_score * 0.20)
            
            # Volume 점수
            vol_ratio = indicators.get('volume_ratio', 1.0)
            vol_score = min(vol_ratio / 1.5, 1.0)  # 1.5배 이상 거래량이 최적
            scores.append(vol_score * 0.20)
            
            signal['alpha_score'] = sum(scores)
            
            logger.info(f"{symbol}: Technical Alpha Score = {signal['alpha_score']:.4f}")
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating technical alpha for {symbol}: {e}")
            return signal


# 테스트 실행
if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("Phase 2 SEC + POLYGON Alpha Generation Test")
    logger.info("=" * 80)
    
    # 1. SEC 펀더멘탈 알파
    logger.info("\n[1] SEC Fundamental Alpha")
    sec_alpha = SECFundamentalAlpha()
    
    # SPY (S&P 500 ETF) 테스트
    spy_signal = sec_alpha.generate_alpha_signal('SPY', 450.0)
    logger.info(f"SPY Alpha Score: {spy_signal['alpha_score']:.4f}")
    
    # 2. POLYGON 기술 알파
    logger.info("\n[2] POLYGON Technical Alpha")
    polygon_alpha = PolygonMarketAlpha()
    
    # SPY 일봉 데이터
    start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    bars = polygon_alpha.get_daily_bars('SPY', start_date, end_date)
    
    if len(bars) > 0:
        tech_signal = polygon_alpha.generate_technical_alpha('SPY', bars)
        logger.info(f"SPY Technical Alpha Score: {tech_signal['alpha_score']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Phase 2 SEC + POLYGON Alpha Generation Complete!")
    logger.info("=" * 80)

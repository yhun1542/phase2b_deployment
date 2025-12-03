#!/usr/bin/env python3
"""
Alpha Engines 모듈 테스트 (새로운 베이스라인)
============================================

다양한 알파 엔진(모멘텀, 변동성, 평균회귀 등)을 새로운 베이스라인에 적용합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AlphaEnginesTest')

def load_validated_baseline() -> Tuple[pd.Series, Dict]:
    """검증된 베이스라인 로드"""
    baseline_path = '/home/ubuntu/phase2b_deployment/data/validated_baseline_returns.csv'
    metadata_path = '/home/ubuntu/phase2b_deployment/data/validated_baseline_metadata.json'
    
    baseline_df = pd.read_csv(baseline_path, parse_dates=['date'])
    baseline_df.set_index('date', inplace=True)
    baseline_returns = baseline_df['returns']
    
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return baseline_returns, metadata

def calculate_metrics(returns: pd.Series) -> Dict:
    """성과 지표 계산"""
    clean_returns = returns.dropna()
    
    if len(clean_returns) < 10:
        return {'sharpe': 0, 'annual_return': 0, 'annual_volatility': 0, 'max_dd': 0}
    
    cumulative = (1 + clean_returns).cumprod()
    years = len(clean_returns) / 252
    annual_return = (cumulative.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0
    annual_vol = clean_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'max_dd': max_dd
    }

def normalize_signal_expanding(signal: pd.Series) -> pd.Series:
    """Expanding Window을 사용한 신호 정규화"""
    expanding_mean = signal.expanding().mean()
    expanding_std = signal.expanding().std()
    
    normalized = (signal - expanding_mean) / (expanding_std + 1e-8)
    return normalized.clip(-3, 3)

class AlphaEngine:
    """다양한 알파 엔진 구현"""
    
    @staticmethod
    def momentum_alpha(returns: pd.Series, window: int = 20) -> pd.Series:
        """모멘텀 알파"""
        momentum = returns.rolling(window=window).mean()
        return normalize_signal_expanding(momentum)
    
    @staticmethod
    def mean_reversion_alpha(returns: pd.Series, window: int = 20) -> pd.Series:
        """평균회귀 알파"""
        rolling_mean = returns.rolling(window=window).mean()
        deviation = returns - rolling_mean
        return normalize_signal_expanding(-deviation)  # 반대 방향
    
    @staticmethod
    def volatility_alpha(returns: pd.Series, window: int = 20) -> pd.Series:
        """변동성 알파 (저변동성 선호)"""
        volatility = returns.rolling(window=window).std()
        return normalize_signal_expanding(-volatility)  # 낮은 변동성 선호
    
    @staticmethod
    def trend_alpha(returns: pd.Series, short_window: int = 10, long_window: int = 30) -> pd.Series:
        """추세 알파 (이중 이동평균)"""
        short_ma = returns.rolling(window=short_window).mean()
        long_ma = returns.rolling(window=long_window).mean()
        trend = short_ma - long_ma
        return normalize_signal_expanding(trend)
    
    @staticmethod
    def autocorr_alpha(returns: pd.Series, window: int = 20) -> pd.Series:
        """자기상관 알파"""
        autocorr = returns.rolling(window=window).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0
        )
        return normalize_signal_expanding(autocorr)
    
    @staticmethod
    def skewness_alpha(returns: pd.Series, window: int = 20) -> pd.Series:
        """왜도 알파 (음의 왜도 회피)"""
        from scipy.stats import skew
        skewness = returns.rolling(window=window).apply(
            lambda x: skew(x) if len(x) > 2 else 0
        )
        return normalize_signal_expanding(-skewness)

def test_individual_alphas(baseline_returns: pd.Series):
    """개별 알파 엔진 테스트"""
    logger.info("\n" + "="*80)
    logger.info("INDIVIDUAL ALPHA ENGINES TEST")
    logger.info("="*80)
    
    alpha_engines = {
        'Momentum': AlphaEngine.momentum_alpha,
        'Mean Reversion': AlphaEngine.mean_reversion_alpha,
        'Volatility': AlphaEngine.volatility_alpha,
        'Trend': AlphaEngine.trend_alpha,
        'Autocorr': AlphaEngine.autocorr_alpha,
        'Skewness': AlphaEngine.skewness_alpha
    }
    
    results = []
    
    for name, alpha_func in alpha_engines.items():
        try:
            # 알파 신호 생성
            alpha_signal = alpha_func(baseline_returns)
            
            # 레버리지 적용
            leverage = 1.0 + alpha_signal * 0.1
            leverage = leverage.clip(0.5, 2.0)
            
            # 포트폴리오 수익률
            portfolio_returns = baseline_returns * leverage
            
            # 거래비용
            leverage_changes = leverage.diff().fillna(0)
            transaction_costs = leverage_changes.abs() * 0.0001
            net_returns = portfolio_returns - transaction_costs
            
            # 성과 계산
            metrics = calculate_metrics(net_returns)
            
            logger.info(f"\n{name}:")
            logger.info(f"  Sharpe: {metrics['sharpe']:.4f}")
            logger.info(f"  Return: {metrics['annual_return']:.4f}")
            
            results.append({
                'name': name,
                'sharpe': metrics['sharpe'],
                'return': metrics['annual_return']
            })
        
        except Exception as e:
            logger.warning(f"{name} failed: {str(e)}")
    
    # 최고 성과 찾기
    best = max(results, key=lambda x: x['sharpe'])
    logger.info(f"\n✓ Best Alpha: {best['name']}, Sharpe={best['sharpe']:.4f}")
    
    return results

def test_combined_alphas(baseline_returns: pd.Series):
    """결합 알파 테스트"""
    logger.info("\n" + "="*80)
    logger.info("COMBINED ALPHAS TEST")
    logger.info("="*80)
    
    # 다양한 알파 신호 생성
    momentum_signal = AlphaEngine.momentum_alpha(baseline_returns, window=20)
    trend_signal = AlphaEngine.trend_alpha(baseline_returns, short_window=10, long_window=30)
    volatility_signal = AlphaEngine.volatility_alpha(baseline_returns, window=20)
    
    # 가중치 조합 테스트
    combinations = [
        ('Momentum 100%', momentum_signal, 1.0, 0.0, 0.0),
        ('Momentum 60% + Trend 40%', momentum_signal, 0.6, 0.4, 0.0),
        ('Momentum 50% + Trend 30% + Vol 20%', momentum_signal, 0.5, 0.3, 0.2),
        ('Equal Weight (3)', momentum_signal, 1/3, 1/3, 1/3),
    ]
    
    results = []
    
    for combo_name, _, w1, w2, w3 in combinations:
        # 결합 신호
        combined_signal = (w1 * momentum_signal + 
                          w2 * trend_signal + 
                          w3 * volatility_signal)
        
        # 레버리지 적용
        leverage = 1.0 + combined_signal * 0.1
        leverage = leverage.clip(0.5, 2.0)
        
        # 포트폴리오 수익률
        portfolio_returns = baseline_returns * leverage
        
        # 거래비용
        leverage_changes = leverage.diff().fillna(0)
        transaction_costs = leverage_changes.abs() * 0.0001
        net_returns = portfolio_returns - transaction_costs
        
        # 성과 계산
        metrics = calculate_metrics(net_returns)
        
        logger.info(f"\n{combo_name}:")
        logger.info(f"  Sharpe: {metrics['sharpe']:.4f}")
        logger.info(f"  Return: {metrics['annual_return']:.4f}")
        
        results.append({
            'name': combo_name,
            'sharpe': metrics['sharpe'],
            'return': metrics['annual_return']
        })
    
    # 최고 성과 찾기
    best = max(results, key=lambda x: x['sharpe'])
    logger.info(f"\n✓ Best Combination: {best['name']}, Sharpe={best['sharpe']:.4f}")
    
    return results

def main():
    logger.info(f"Starting Alpha Engines Module Test at {datetime.now()}")
    
    # 검증된 베이스라인 로드
    logger.info("\n[1] Loading Validated Baseline")
    baseline_returns, metadata = load_validated_baseline()
    
    logger.info(f"✓ Baseline loaded: {len(baseline_returns)} days")
    logger.info(f"  Sharpe: {metadata['performance']['sharpe_ratio']:.4f}")
    
    # 1. 개별 알파 엔진 테스트
    logger.info("\n[2] Testing Individual Alpha Engines")
    individual_results = test_individual_alphas(baseline_returns)
    
    # 2. 결합 알파 테스트
    logger.info("\n[3] Testing Combined Alphas")
    combined_results = test_combined_alphas(baseline_returns)
    
    # 최종 요약
    logger.info("\n" + "="*80)
    logger.info("ALPHA ENGINES MODULE TEST SUMMARY")
    logger.info("="*80)
    
    baseline_metrics = calculate_metrics(baseline_returns)
    
    logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"Best Individual Alpha Sharpe: {max(r['sharpe'] for r in individual_results):.4f}")
    logger.info(f"Best Combined Alpha Sharpe: {max(r['sharpe'] for r in combined_results):.4f}")
    
    logger.info(f"\n✓ Alpha Engines Module: OK")

if __name__ == '__main__':
    main()

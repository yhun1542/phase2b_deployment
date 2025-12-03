#!/usr/bin/env python3
"""
Signal Processing 모듈 테스트 (새로운 베이스라인)
================================================

FracDiff, EMD, 모멘텀/변동성 신호를 새로운 베이스라인(Sharpe 4.24)에 적용합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from datetime import datetime

# Signal Processing 모듈 임포트
from signal_processing import FractionalDifferencing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SignalProcessingTest')

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

def test_fractional_differencing(baseline_returns: pd.Series):
    """Fractional Differencing 테스트"""
    logger.info("\n" + "="*80)
    logger.info("FRACTIONAL DIFFERENCING TEST")
    logger.info("="*80)
    
    fracdiff = FractionalDifferencing(d=0.4)
    
    try:
        # FracDiff 적용
        fracdiff_returns = fracdiff.fit_transform(baseline_returns)
        
        logger.info(f"✓ FracDiff applied: {len(fracdiff_returns)} days")
        logger.info(f"  Original length: {len(baseline_returns)}")
        logger.info(f"  Differenced length: {len(fracdiff_returns)}")
        
        # 성과 비교
        original_metrics = calculate_metrics(baseline_returns)
        fracdiff_metrics = calculate_metrics(fracdiff_returns)
        
        logger.info(f"\nOriginal Sharpe: {original_metrics['sharpe']:.4f}")
        logger.info(f"FracDiff Sharpe: {fracdiff_metrics['sharpe']:.4f}")
        logger.info(f"Change: {(fracdiff_metrics['sharpe']/original_metrics['sharpe']-1)*100:+.2f}%")
        
        return fracdiff_returns
    
    except Exception as e:
        logger.warning(f"FracDiff failed: {str(e)}")
        return None

def test_momentum_signals(baseline_returns: pd.Series):
    """모멘텀 신호 테스트"""
    logger.info("\n" + "="*80)
    logger.info("MOMENTUM SIGNALS TEST")
    logger.info("="*80)
    
    momentum_windows = [5, 10, 20, 30, 60, 90]
    results = []
    
    for window in momentum_windows:
        momentum = baseline_returns.rolling(window=window).mean()
        momentum_normalized = normalize_signal_expanding(momentum)
        
        # 신호를 레버리지로 적용
        leverage = 1.0 + momentum_normalized * 0.1
        leverage = leverage.clip(0.5, 2.0)
        
        # 포트폴리오 수익률
        portfolio_returns = baseline_returns * leverage
        
        # 거래비용
        leverage_changes = leverage.diff().fillna(0)
        transaction_costs = leverage_changes.abs() * 0.0001
        net_returns = portfolio_returns - transaction_costs
        
        # 성과 계산
        metrics = calculate_metrics(net_returns)
        
        logger.info(f"\nWindow={window}:")
        logger.info(f"  Sharpe: {metrics['sharpe']:.4f}")
        logger.info(f"  Return: {metrics['annual_return']:.4f}")
        
        results.append({
            'window': window,
            'sharpe': metrics['sharpe'],
            'return': metrics['annual_return']
        })
    
    # 최고 성과 찾기
    best = max(results, key=lambda x: x['sharpe'])
    logger.info(f"\n✓ Best Momentum: Window={best['window']}, Sharpe={best['sharpe']:.4f}")
    
    return results

def test_volatility_signals(baseline_returns: pd.Series):
    """변동성 신호 테스트"""
    logger.info("\n" + "="*80)
    logger.info("VOLATILITY SIGNALS TEST")
    logger.info("="*80)
    
    vol_windows = [10, 20, 30, 60]
    results = []
    
    for window in vol_windows:
        volatility = baseline_returns.rolling(window=window).std()
        vol_normalized = normalize_signal_expanding(volatility)
        
        # 변동성이 높으면 레버리지 감소
        leverage = 1.0 - vol_normalized * 0.1
        leverage = leverage.clip(0.5, 2.0)
        
        # 포트폴리오 수익률
        portfolio_returns = baseline_returns * leverage
        
        # 거래비용
        leverage_changes = leverage.diff().fillna(0)
        transaction_costs = leverage_changes.abs() * 0.0001
        net_returns = portfolio_returns - transaction_costs
        
        # 성과 계산
        metrics = calculate_metrics(net_returns)
        
        logger.info(f"\nWindow={window}:")
        logger.info(f"  Sharpe: {metrics['sharpe']:.4f}")
        logger.info(f"  Return: {metrics['annual_return']:.4f}")
        
        results.append({
            'window': window,
            'sharpe': metrics['sharpe'],
            'return': metrics['annual_return']
        })
    
    # 최고 성과 찾기
    best = max(results, key=lambda x: x['sharpe'])
    logger.info(f"\n✓ Best Volatility: Window={best['window']}, Sharpe={best['sharpe']:.4f}")
    
    return results

def test_combined_signals(baseline_returns: pd.Series):
    """결합 신호 테스트"""
    logger.info("\n" + "="*80)
    logger.info("COMBINED SIGNALS TEST")
    logger.info("="*80)
    
    # 모멘텀 + 변동성 신호 결합
    momentum = baseline_returns.rolling(window=20).mean()
    momentum_norm = normalize_signal_expanding(momentum)
    
    volatility = baseline_returns.rolling(window=30).std()
    vol_norm = normalize_signal_expanding(volatility)
    
    # 결합 신호
    combined_signal = momentum_norm * 0.6 - vol_norm * 0.4
    
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
    baseline_metrics = calculate_metrics(baseline_returns)
    combined_metrics = calculate_metrics(net_returns)
    
    logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"Combined Sharpe: {combined_metrics['sharpe']:.4f}")
    logger.info(f"Improvement: {(combined_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
    
    return combined_metrics

def main():
    logger.info(f"Starting Signal Processing Module Test at {datetime.now()}")
    
    # 검증된 베이스라인 로드
    logger.info("\n[1] Loading Validated Baseline")
    baseline_returns, metadata = load_validated_baseline()
    
    logger.info(f"✓ Baseline loaded: {len(baseline_returns)} days")
    logger.info(f"  Sharpe: {metadata['performance']['sharpe_ratio']:.4f}")
    
    # 1. Fractional Differencing 테스트
    logger.info("\n[2] Testing Fractional Differencing")
    fracdiff_returns = test_fractional_differencing(baseline_returns)
    
    # 2. 모멘텀 신호 테스트
    logger.info("\n[3] Testing Momentum Signals")
    momentum_results = test_momentum_signals(baseline_returns)
    
    # 3. 변동성 신호 테스트
    logger.info("\n[4] Testing Volatility Signals")
    vol_results = test_volatility_signals(baseline_returns)
    
    # 4. 결합 신호 테스트
    logger.info("\n[5] Testing Combined Signals")
    combined_metrics = test_combined_signals(baseline_returns)
    
    # 최종 요약
    logger.info("\n" + "="*80)
    logger.info("SIGNAL PROCESSING MODULE TEST SUMMARY")
    logger.info("="*80)
    
    baseline_metrics = calculate_metrics(baseline_returns)
    
    logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"Best Momentum Sharpe: {max(r['sharpe'] for r in momentum_results):.4f}")
    logger.info(f"Best Volatility Sharpe: {max(r['sharpe'] for r in vol_results):.4f}")
    logger.info(f"Combined Sharpe: {combined_metrics['sharpe']:.4f}")
    
    logger.info(f"\n✓ Signal Processing Module: OK")

if __name__ == '__main__':
    main()

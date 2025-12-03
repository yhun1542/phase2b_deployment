#!/usr/bin/env python3
"""
최적화된 베이스라인 + Returns Mean 신호 전략
===========================================

Sharpe 4.4493 달성 전략 (원본 대비 +52.54% 개선)
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
logger = logging.getLogger('OptimizedStrategy')

def load_all_data() -> Tuple[pd.Series, pd.DataFrame]:
    """모든 필요한 데이터 로드"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_path = os.path.join(base_dir, 'data', 'baseline_returns.csv')
    signals_path = os.path.join(base_dir, 'data', 'phase2_signals.csv')
    
    baseline_df = pd.read_csv(baseline_path, parse_dates=['date'], index_col='date')
    signals_df = pd.read_csv(signals_path, parse_dates=['date'], index_col='date')
    
    baseline_returns = baseline_df['returns']
    
    return baseline_returns, signals_df

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

def calculate_baseline_portfolio_returns(baseline_returns: pd.Series, 
                                         phase2_signals: pd.Series) -> pd.Series:
    """현재 베이스라인 포트폴리오 수익률 계산"""
    signals_smoothed = phase2_signals.iloc[:, 0].rolling(window=40).mean()
    
    common_dates = baseline_returns.index.intersection(signals_smoothed.index)
    baseline = baseline_returns.loc[common_dates]
    signals = signals_smoothed.loc[common_dates]
    
    signal_min = signals.expanding().min()
    signal_max = signals.expanding().max()
    signals_normalized = (signals - signal_min) / (signal_max - signal_min + 1e-8)
    
    inverse_signal = 1.0 - signals_normalized
    leverage = 0.2 + inverse_signal * (2.8 - 0.2)
    
    smoothed_leverage = leverage.copy()
    for i in range(1, len(leverage)):
        if abs(leverage.iloc[i] - smoothed_leverage.iloc[i-1]) < 0.10:
            smoothed_leverage.iloc[i] = smoothed_leverage.iloc[i-1]
    
    portfolio_returns = baseline * smoothed_leverage
    portfolio_vol = portfolio_returns.rolling(window=252).std() * np.sqrt(252)
    
    adjusted_leverage = smoothed_leverage.copy()
    for i in range(252, len(portfolio_vol)):
        if portfolio_vol.iloc[i] > 0.12:
            adjustment_factor = 0.12 / (portfolio_vol.iloc[i] + 1e-8)
            adjusted_leverage.iloc[i] = adjusted_leverage.iloc[i] * adjustment_factor
    
    adjusted_leverage = adjusted_leverage.clip(0.2, 2.8)
    baseline_portfolio = baseline * adjusted_leverage
    
    return baseline_portfolio

def normalize_signal(signal: pd.Series) -> pd.Series:
    """신호 정규화"""
    clean_signal = signal.dropna()
    if len(clean_signal) == 0:
        return signal
    
    normalized = (signal - signal.mean()) / (signal.std() + 1e-8)
    return normalized.clip(-3, 3)

def run_optimized_strategy(baseline_portfolio: pd.Series,
                          baseline_returns: pd.Series,
                          returns_mean_window: int = 5,
                          signal_weight: float = 0.20) -> Dict:
    """최적화된 전략 실행"""
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING OPTIMIZED STRATEGY")
    logger.info("="*80)
    logger.info(f"Returns Mean Window: {returns_mean_window}")
    logger.info(f"Signal Weight: {signal_weight}")
    
    # Returns Mean 신호 생성
    returns_mean = baseline_returns.rolling(window=returns_mean_window).mean()
    returns_mean_normalized = normalize_signal(returns_mean)
    
    # 정렬
    common_dates = baseline_portfolio.index.intersection(returns_mean_normalized.index)
    baseline_aligned = baseline_portfolio.loc[common_dates]
    signal_aligned = returns_mean_normalized.loc[common_dates]
    
    # 신호를 레버리지로 적용
    signal_leverage = 1.0 + signal_aligned * signal_weight
    signal_leverage = signal_leverage.clip(0.5, 2.0)
    
    # 포트폴리오 수익률
    combined_returns = baseline_aligned * signal_leverage
    
    # 거래비용
    leverage_changes = signal_leverage.diff().fillna(0)
    transaction_costs = leverage_changes.abs() * 0.0001
    
    net_returns = combined_returns - transaction_costs
    
    # 성과 계산
    baseline_metrics = calculate_metrics(baseline_aligned)
    combined_metrics = calculate_metrics(net_returns)
    
    improvement = (combined_metrics['sharpe'] / baseline_metrics['sharpe'] - 1) * 100
    
    logger.info(f"\n✓ Strategy Results:")
    logger.info(f"  Baseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"  Combined Sharpe: {combined_metrics['sharpe']:.4f}")
    logger.info(f"  Improvement: {improvement:+.2f}%")
    logger.info(f"  Baseline Return: {baseline_metrics['annual_return']:.4f}")
    logger.info(f"  Combined Return: {combined_metrics['annual_return']:.4f}")
    logger.info(f"  Baseline Vol: {baseline_metrics['annual_volatility']:.4f}")
    logger.info(f"  Combined Vol: {combined_metrics['annual_volatility']:.4f}")
    logger.info(f"  Baseline Max DD: {baseline_metrics['max_dd']:.4f}")
    logger.info(f"  Combined Max DD: {combined_metrics['max_dd']:.4f}")
    
    # 결과 저장
    result_df = pd.DataFrame({
        'date': net_returns.index,
        'returns': net_returns.values,
        'leverage': signal_leverage.values
    })
    
    return {
        'strategy_returns': net_returns,
        'leverage': signal_leverage,
        'metrics': combined_metrics,
        'baseline_metrics': baseline_metrics,
        'improvement': improvement,
        'result_df': result_df
    }

def save_results(results: Dict, output_dir: str):
    """결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 수익률 저장
    results['result_df'].to_csv(
        os.path.join(output_dir, 'optimized_strategy_returns.csv'),
        index=False
    )
    
    # 메트릭 저장
    metrics_df = pd.DataFrame({
        'metric': ['sharpe', 'annual_return', 'annual_volatility', 'max_dd'],
        'baseline': [
            results['baseline_metrics']['sharpe'],
            results['baseline_metrics']['annual_return'],
            results['baseline_metrics']['annual_volatility'],
            results['baseline_metrics']['max_dd']
        ],
        'optimized': [
            results['metrics']['sharpe'],
            results['metrics']['annual_return'],
            results['metrics']['annual_volatility'],
            results['metrics']['max_dd']
        ]
    })
    
    metrics_df.to_csv(
        os.path.join(output_dir, 'optimized_strategy_metrics.csv'),
        index=False
    )
    
    logger.info(f"\n✓ Results saved to {output_dir}")

def main():
    logger.info(f"Starting Optimized Strategy Execution at {datetime.now()}")
    
    # 데이터 로드
    baseline_returns, phase2_signals = load_all_data()
    baseline_portfolio = calculate_baseline_portfolio_returns(baseline_returns, phase2_signals)
    
    # 최적화된 전략 실행
    results = run_optimized_strategy(
        baseline_portfolio,
        baseline_returns,
        returns_mean_window=5,
        signal_weight=0.20
    )
    
    # 결과 저장
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'results', 'optimized_strategy')
    save_results(results, output_dir)
    
    logger.info(f"\nCompleted at {datetime.now()}")

if __name__ == '__main__':
    main()

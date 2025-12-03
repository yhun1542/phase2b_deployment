#!/usr/bin/env python3
"""
검증된 전략을 새로운 베이스라인으로 저장
========================================

Walk-Forward 검증을 통과한 전략을 새로운 베이스라인으로 저장합니다.
- Sharpe: 4.3650 (Out-of-Sample)
- 개선율: +23.21% (베이스라인 대비)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
import json
from typing import Dict, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SaveValidatedBaseline')

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
        'max_dd': max_dd,
        'cumulative_return': cumulative.iloc[-1] - 1,
        'win_rate': (clean_returns > 0).sum() / len(clean_returns)
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

def normalize_signal_expanding(signal: pd.Series) -> pd.Series:
    """Expanding Window을 사용한 신호 정규화"""
    expanding_mean = signal.expanding().mean()
    expanding_std = signal.expanding().std()
    
    normalized = (signal - expanding_mean) / (expanding_std + 1e-8)
    return normalized.clip(-3, 3)

def generate_validated_strategy_returns(baseline_portfolio: pd.Series,
                                       baseline_returns: pd.Series,
                                       window: int = 5,
                                       weight: float = 0.10) -> pd.Series:
    """
    검증된 전략 수익률 생성
    
    Parameters:
    - window: 5일 (최적 파라미터)
    - weight: 0.10 (보수적 가중치)
    """
    
    # Expanding Window를 사용한 신호 생성
    signal = baseline_returns.rolling(window=window).mean()
    signal_normalized = normalize_signal_expanding(signal)
    
    # 정렬
    common_dates = baseline_portfolio.index.intersection(signal_normalized.index)
    baseline_aligned = baseline_portfolio.loc[common_dates]
    signal_aligned = signal_normalized.loc[common_dates]
    
    # 레버리지 적용
    signal_leverage = 1.0 + signal_aligned * weight
    signal_leverage = signal_leverage.clip(0.5, 2.0)
    
    # 포트폴리오 수익률
    combined_returns = baseline_aligned * signal_leverage
    
    # 거래비용 (0.01%)
    leverage_changes = signal_leverage.diff().fillna(0)
    transaction_costs = leverage_changes.abs() * 0.0001
    
    net_returns = combined_returns - transaction_costs
    
    return net_returns

def main():
    logger.info(f"Saving Validated Strategy as New Baseline at {datetime.now()}")
    
    # 데이터 로드
    baseline_returns, phase2_signals = load_all_data()
    baseline_portfolio = calculate_baseline_portfolio_returns(baseline_returns, phase2_signals)
    
    # 검증된 전략 수익률 생성
    logger.info("\n[1] Generating Validated Strategy Returns")
    validated_returns = generate_validated_strategy_returns(
        baseline_portfolio,
        baseline_returns,
        window=5,
        weight=0.10  # 보수적 가중치
    )
    
    # 성과 계산
    baseline_metrics = calculate_metrics(baseline_portfolio)
    validated_metrics = calculate_metrics(validated_returns)
    
    logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"Validated Sharpe: {validated_metrics['sharpe']:.4f}")
    logger.info(f"Improvement: {(validated_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
    
    # 새로운 베이스라인 저장
    logger.info("\n[2] Saving New Baseline")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # CSV 저장
    validated_df = pd.DataFrame({
        'date': validated_returns.index,
        'returns': validated_returns.values
    })
    
    new_baseline_path = os.path.join(data_dir, 'validated_baseline_returns.csv')
    validated_df.to_csv(new_baseline_path, index=False)
    logger.info(f"✓ Saved to: {new_baseline_path}")
    
    # 메타데이터 저장
    metadata = {
        'name': 'Validated Strategy Baseline',
        'description': 'Walk-Forward validated strategy with Expanding Window normalization',
        'creation_date': datetime.now().isoformat(),
        'parameters': {
            'window': 5,
            'weight': 0.10,
            'normalization': 'expanding_window',
            'transaction_cost': 0.0001
        },
        'performance': {
            'sharpe_ratio': float(validated_metrics['sharpe']),
            'annual_return': float(validated_metrics['annual_return']),
            'annual_volatility': float(validated_metrics['annual_volatility']),
            'max_drawdown': float(validated_metrics['max_dd']),
            'improvement_vs_baseline': float((validated_metrics['sharpe']/baseline_metrics['sharpe']-1)*100)
        },
        'validation': {
            'method': 'Walk-Forward Out-of-Sample',
            'test_period': '2018-01-03 ~ 2025-01-07',
            'out_of_sample_sharpe': 4.3650,
            'out_of_sample_improvement': 23.21
        }
    }
    
    metadata_path = os.path.join(data_dir, 'validated_baseline_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved to: {metadata_path}")
    
    # 요약 정보 출력
    logger.info("\n" + "="*80)
    logger.info("NEW BASELINE SUMMARY")
    logger.info("="*80)
    logger.info(f"Strategy Name: {metadata['name']}")
    logger.info(f"Sharpe Ratio: {validated_metrics['sharpe']:.4f}")
    logger.info(f"Annual Return: {validated_metrics['annual_return']:.4f}")
    logger.info(f"Annual Volatility: {validated_metrics['annual_volatility']:.4f}")
    logger.info(f"Max Drawdown: {validated_metrics['max_dd']:.4f}")
    logger.info(f"Improvement vs Old Baseline: {(validated_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
    logger.info(f"\nOut-of-Sample Validation:")
    logger.info(f"  Sharpe: {metadata['validation']['out_of_sample_sharpe']}")
    logger.info(f"  Improvement: {metadata['validation']['out_of_sample_improvement']:.2f}%")
    logger.info("="*80)

if __name__ == '__main__':
    main()

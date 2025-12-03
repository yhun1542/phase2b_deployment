#!/usr/bin/env python3
"""
대안 전략 3: 선형 레버리지 + 동적 변동성 조합
=============================================

선형 레버리지 조정과 동적 변동성 타겟팅을 결합합니다:

1. 신호 강도에 따라 선형 레버리지 계산 (기존 최적 전략)
2. 포트폴리오 변동성이 목표를 초과하면 레버리지 감소

목표:
- 수익성과 리스크 관리의 균형
- 최적 전략의 수익성 + 동적 변동성의 리스크 관리
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HybridStrategy')

def calculate_net_metrics(returns: pd.Series, leverage_changes: pd.Series, 
                          cost_per_trade: float = 0.0001) -> dict:
    transaction_costs = leverage_changes.abs() * cost_per_trade
    net_returns = returns - transaction_costs
    
    if len(net_returns.dropna()) < 10:
        return {}
    
    annual_return = (1 + net_returns.dropna()).prod() ** (252 / len(net_returns.dropna())) - 1
    annual_vol = net_returns.dropna().std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    return {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'annual_volatility': annual_vol
    }

class HybridStrategy:
    def __init__(self, baseline_returns: pd.Series, signals: pd.DataFrame):
        self.signals_shifted = signals.shift(1)
        common_dates = baseline_returns.index.intersection(self.signals_shifted.index)
        self.baseline_aligned = baseline_returns.loc[common_dates]
        self.signals_aligned = self.signals_shifted.loc[common_dates]
        
        baseline_metrics = calculate_net_metrics(self.baseline_aligned, pd.Series(0, index=self.baseline_aligned.index))
        self.baseline_sharpe = baseline_metrics['sharpe']
        logger.info(f"Baseline Sharpe: {self.baseline_sharpe:.4f}")

    def run(self, target_vol=0.15):
        logger.info("\n" + "="*80)
        logger.info(f"Running Hybrid Strategy (Target Vol: {target_vol*100:.1f}%)")
        logger.info("="*80)

        # 최적 파라미터 사용
        smoothing_window = 40
        
        # 신호 스무딩
        signals_smoothed = self.signals_aligned.rolling(window=smoothing_window).mean().dropna()
        
        # 정규화 (expanding window)
        expanding_abs = signals_smoothed.abs().expanding()
        min_val, max_val = expanding_abs.min(), expanding_abs.max()
        signals_normalized = (signals_smoothed.abs() - min_val) / (max_val - min_val + 1e-8)
        
        signal_strength = signals_normalized.mean(axis=1).dropna()
        
        # 1단계: 선형 레버리지 계산
        inverse_signal = 1.0 - signal_strength
        leverage = 0.2 + inverse_signal * (2.8 - 0.2)
        
        # 2단계: 임계값 스무딩
        smoothed_leverage = leverage.copy()
        for i in range(1, len(leverage)):
            if abs(leverage.iloc[i] - smoothed_leverage.iloc[i-1]) < 0.10:
                smoothed_leverage.iloc[i] = smoothed_leverage.iloc[i-1]
        
        # 3단계: 동적 변동성 조정
        common_dates = self.baseline_aligned.index.intersection(smoothed_leverage.index)
        baseline = self.baseline_aligned.loc[common_dates]
        final_leverage = smoothed_leverage.loc[common_dates]
        
        # 포트폴리오 변동성 계산 (252일 롤링)
        portfolio_returns = baseline * final_leverage
        portfolio_vol = portfolio_returns.rolling(window=252).std() * np.sqrt(252)
        
        # 변동성이 목표를 초과하면 레버리지 감소
        adjusted_leverage = final_leverage.copy()
        for i in range(252, len(portfolio_vol)):
            if portfolio_vol.iloc[i] > target_vol:
                adjustment_factor = target_vol / (portfolio_vol.iloc[i] + 1e-8)
                adjusted_leverage.iloc[i] = adjusted_leverage.iloc[i] * adjustment_factor
        
        adjusted_leverage = adjusted_leverage.clip(0.2, 2.8)
        
        # 성과 계산
        gross_returns = baseline * adjusted_leverage
        leverage_changes = adjusted_leverage.diff().fillna(0)
        
        net_metrics = calculate_net_metrics(gross_returns, leverage_changes)
        
        logger.info(f"\nHybrid Strategy Results (Target Vol: {target_vol*100:.1f}%):")
        logger.info(f"  Net Sharpe: {net_metrics['sharpe']:.4f}")
        logger.info(f"  Improvement: {(net_metrics['sharpe'] / self.baseline_sharpe - 1)*100:+.2f}%")
        logger.info(f"  Annual Return: {net_metrics['annual_return']*100:.2f}%")
        logger.info(f"  Annual Volatility: {net_metrics['annual_volatility']*100:.2f}%")

        return net_metrics

def main():
    baseline_path = '/home/ubuntu/ares7-ensemble/tuning/results/step6_balanced_daily_returns.csv'
    signals_path = '/home/ubuntu/ares8_turbo_test/engine_signals_phase2_ec2.csv'
    
    baseline_df = pd.read_csv(baseline_path, parse_dates=['date'], index_col='date')
    signals_df = pd.read_csv(signals_path, parse_dates=['date'], index_col='date')
    
    strategy = HybridStrategy(baseline_df['returns'], signals_df)
    
    # 다양한 목표 변동성 테스트
    results = {}
    for target_vol in [0.12, 0.15, 0.18, 0.20]:
        metrics = strategy.run(target_vol)
        results[f'target_vol_{target_vol}'] = metrics

if __name__ == '__main__':
    main()

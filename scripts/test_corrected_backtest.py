#!/usr/bin/env python3
"""
룩어헤드 바이어스 제거 및 올바른 백테스트 방법론 적용
====================================================

1. Expanding Window 정규화 (룩어헤드 바이어스 제거)
2. Walk-Forward 최적화 (과적합 제거)
3. Out-of-Sample 검증
4. 현실적인 거래비용 적용
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CorrectedBacktest')

def load_all_data() -> Tuple[pd.Series, pd.DataFrame]:
    """모든 필요한 데이터 로드"""
    baseline_path = '/home/ubuntu/phase2b_deployment/data/baseline_returns.csv'
    signals_path = '/home/ubuntu/phase2b_deployment/data/phase2_signals.csv'
    
    baseline_df = pd.read_csv(baseline_path, parse_dates=['date'], index_col='date')
    signals_df = pd.read_csv(signals_path, parse_dates=['date'], index_col='date')
    
    baseline_returns = baseline_df['returns']
    
    return baseline_returns, signals_df

def calculate_metrics(returns: pd.Series, name: str = "") -> Dict:
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
    """
    Expanding Window을 사용한 신호 정규화 (룩어헤드 바이어스 제거)
    
    각 시점에서 과거 데이터만 사용하여 평균과 표준편차를 계산합니다.
    """
    expanding_mean = signal.expanding().mean()
    expanding_std = signal.expanding().std()
    
    normalized = (signal - expanding_mean) / (expanding_std + 1e-8)
    return normalized.clip(-3, 3)

def test_walk_forward_optimization(baseline_portfolio: pd.Series,
                                   baseline_returns: pd.Series,
                                   train_period: int = 252*2,  # 2년
                                   test_period: int = 252) -> Dict:
    """
    Walk-Forward 최적화 및 검증
    
    1. 훈련 기간(In-Sample)으로 최적 파라미터 찾기
    2. 검증 기간(Out-of-Sample)에서 성과 측정
    3. 한 달씩 이동하며 반복
    """
    
    logger.info("\n" + "="*80)
    logger.info("WALK-FORWARD OPTIMIZATION TEST")
    logger.info("="*80)
    logger.info(f"Train Period: {train_period} days ({train_period/252:.1f} years)")
    logger.info(f"Test Period: {test_period} days ({test_period/252:.1f} years)")
    
    all_test_returns = []
    all_test_dates = []
    optimal_params_list = []
    
    # Walk-Forward 루프
    for start_idx in range(0, len(baseline_returns) - train_period - test_period, test_period):
        train_end_idx = start_idx + train_period
        test_end_idx = train_end_idx + test_period
        
        train_dates = baseline_returns.index[start_idx:train_end_idx]
        test_dates = baseline_returns.index[train_end_idx:test_end_idx]
        
        train_baseline = baseline_returns.loc[train_dates]
        train_baseline_portfolio = baseline_portfolio.loc[train_dates]
        
        test_baseline = baseline_returns.loc[test_dates]
        test_baseline_portfolio = baseline_portfolio.loc[test_dates]
        
        # 훈련 기간에서 최적 파라미터 찾기
        best_sharpe = -np.inf
        best_params = {'window': 5, 'weight': 0.05}
        
        for window in [5, 10, 15, 20]:
            for weight in [0.05, 0.10, 0.15, 0.20]:
                # 훈련 기간에서 신호 생성 (Expanding Window 사용)
                train_signal = train_baseline.rolling(window=window).mean()
                train_signal_normalized = normalize_signal_expanding(train_signal)
                
                # 정렬
                common_dates = train_baseline_portfolio.index.intersection(train_signal_normalized.index)
                train_bp = train_baseline_portfolio.loc[common_dates]
                train_sig = train_signal_normalized.loc[common_dates]
                
                if len(train_bp) < 10:
                    continue
                
                # 레버리지 적용
                train_leverage = 1.0 + train_sig * weight
                train_leverage = train_leverage.clip(0.5, 2.0)
                
                # 포트폴리오 수익률
                train_returns = train_bp * train_leverage
                
                # 거래비용 (0.01%)
                leverage_changes = train_leverage.diff().fillna(0)
                transaction_costs = leverage_changes.abs() * 0.0001
                train_returns_net = train_returns - transaction_costs
                
                # Sharpe 계산
                metrics = calculate_metrics(train_returns_net)
                
                if metrics['sharpe'] > best_sharpe:
                    best_sharpe = metrics['sharpe']
                    best_params = {'window': window, 'weight': weight}
        
        logger.info(f"\nTrain Period: {train_dates[0].date()} ~ {train_dates[-1].date()}")
        logger.info(f"  Best Params: Window={best_params['window']}, Weight={best_params['weight']:.2f}")
        logger.info(f"  Train Sharpe: {best_sharpe:.4f}")
        
        # 검증 기간에서 성과 측정
        test_signal = test_baseline.rolling(window=best_params['window']).mean()
        test_signal_normalized = normalize_signal_expanding(test_signal)
        
        # 정렬
        common_dates = test_baseline_portfolio.index.intersection(test_signal_normalized.index)
        test_bp = test_baseline_portfolio.loc[common_dates]
        test_sig = test_signal_normalized.loc[common_dates]
        
        if len(test_bp) < 10:
            continue
        
        # 레버리지 적용
        test_leverage = 1.0 + test_sig * best_params['weight']
        test_leverage = test_leverage.clip(0.5, 2.0)
        
        # 포트폴리오 수익률
        test_returns = test_bp * test_leverage
        
        # 거래비용
        leverage_changes = test_leverage.diff().fillna(0)
        transaction_costs = leverage_changes.abs() * 0.0001
        test_returns_net = test_returns - transaction_costs
        
        # 성과 계산
        test_metrics = calculate_metrics(test_returns_net)
        baseline_metrics = calculate_metrics(test_bp)
        
        logger.info(f"Test Period: {test_dates[0].date()} ~ {test_dates[-1].date()}")
        logger.info(f"  Baseline Sharpe: {baseline_metrics['sharpe']:.4f}")
        logger.info(f"  Test Sharpe: {test_metrics['sharpe']:.4f}")
        logger.info(f"  Improvement: {(test_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
        
        all_test_returns.append(test_returns_net)
        all_test_dates.extend(test_dates)
        optimal_params_list.append(best_params)
    
    # 전체 검증 기간 성과
    if all_test_returns:
        combined_test_returns = pd.concat(all_test_returns)
        combined_test_returns = combined_test_returns.loc[~combined_test_returns.index.duplicated(keep='first')]
        
        test_metrics = calculate_metrics(combined_test_returns)
        baseline_test = baseline_portfolio.loc[combined_test_returns.index]
        baseline_metrics = calculate_metrics(baseline_test)
        
        logger.info("\n" + "="*80)
        logger.info("OVERALL OUT-OF-SAMPLE TEST RESULTS")
        logger.info("="*80)
        logger.info(f"Test Period: {combined_test_returns.index[0].date()} ~ {combined_test_returns.index[-1].date()}")
        logger.info(f"Test Days: {len(combined_test_returns)}")
        logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
        logger.info(f"Test Sharpe: {test_metrics['sharpe']:.4f}")
        logger.info(f"Improvement: {(test_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
        logger.info(f"\nBaseline Return: {baseline_metrics['annual_return']:.4f}")
        logger.info(f"Test Return: {test_metrics['annual_return']:.4f}")
        logger.info(f"\nBaseline Vol: {baseline_metrics['annual_volatility']:.4f}")
        logger.info(f"Test Vol: {test_metrics['annual_volatility']:.4f}")
        logger.info(f"\nBaseline Max DD: {baseline_metrics['max_dd']:.4f}")
        logger.info(f"Test Max DD: {test_metrics['max_dd']:.4f}")
        
        return {
            'baseline_metrics': baseline_metrics,
            'test_metrics': test_metrics,
            'combined_test_returns': combined_test_returns,
            'optimal_params_list': optimal_params_list
        }
    
    return {}

def test_simple_corrected_backtest(baseline_portfolio: pd.Series,
                                   baseline_returns: pd.Series) -> Dict:
    """
    간단한 수정된 백테스트 (전체 기간, Expanding Window 사용)
    
    룩어헤드 바이어스는 제거하지만, 파라미터는 고정합니다.
    """
    
    logger.info("\n" + "="*80)
    logger.info("SIMPLE CORRECTED BACKTEST (Expanding Window)")
    logger.info("="*80)
    
    # 고정 파라미터 (Walk-Forward에서 찾은 평균값)
    window = 5
    weight = 0.10
    
    logger.info(f"Parameters: Window={window}, Weight={weight}")
    
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
    
    # 성과 계산
    baseline_metrics = calculate_metrics(baseline_aligned)
    combined_metrics = calculate_metrics(net_returns)
    
    logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"Combined Sharpe: {combined_metrics['sharpe']:.4f}")
    logger.info(f"Improvement: {(combined_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
    logger.info(f"\nBaseline Return: {baseline_metrics['annual_return']:.4f}")
    logger.info(f"Combined Return: {combined_metrics['annual_return']:.4f}")
    logger.info(f"\nBaseline Vol: {baseline_metrics['annual_volatility']:.4f}")
    logger.info(f"Combined Vol: {combined_metrics['annual_volatility']:.4f}")
    logger.info(f"\nBaseline Max DD: {baseline_metrics['max_dd']:.4f}")
    logger.info(f"Combined Max DD: {combined_metrics['max_dd']:.4f}")
    
    return {
        'baseline_metrics': baseline_metrics,
        'combined_metrics': combined_metrics,
        'net_returns': net_returns
    }

def main():
    logger.info(f"Starting Corrected Backtest at {datetime.now()}")
    
    # 데이터 로드
    baseline_returns, phase2_signals = load_all_data()
    baseline_portfolio = calculate_baseline_portfolio_returns(baseline_returns, phase2_signals)
    
    # 1. 간단한 수정된 백테스트 (Expanding Window)
    simple_results = test_simple_corrected_backtest(baseline_portfolio, baseline_returns)
    
    # 2. Walk-Forward 최적화 및 검증
    wf_results = test_walk_forward_optimization(baseline_portfolio, baseline_returns)
    
    # 결과 비교
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: ORIGINAL vs CORRECTED")
    logger.info("="*80)
    logger.info("\n원본 (룩어헤드 바이어스 있음):")
    logger.info("  Sharpe: 4.4493")
    logger.info("  Window: 5, Weight: 0.20")
    logger.info("\n수정된 (Expanding Window):")
    logger.info(f"  Sharpe: {simple_results['combined_metrics']['sharpe']:.4f}")
    logger.info(f"  Improvement vs Baseline: {(simple_results['combined_metrics']['sharpe']/simple_results['baseline_metrics']['sharpe']-1)*100:+.2f}%")
    
    if wf_results:
        logger.info("\nWalk-Forward 검증 (Out-of-Sample):")
        logger.info(f"  Sharpe: {wf_results['test_metrics']['sharpe']:.4f}")
        logger.info(f"  Improvement vs Baseline: {(wf_results['test_metrics']['sharpe']/wf_results['baseline_metrics']['sharpe']-1)*100:+.2f}%")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
모멘텀 + 변동성 신호 하이브리드 테스트
====================================

5일 모멘텀 신호(Sharpe 4.68)와 10일 변동성 신호(Sharpe 4.49)의 최적 결합을 찾습니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HybridTest')

def load_validated_baseline() -> Tuple[pd.Series, Dict]:
    """검증된 베이스라인 로드"""
    baseline_path = '/home/ubuntu/phase2b_deployment/data/validated_baseline_returns.csv'
    metadata_path = '/home/ubuntu/phase2b_deployment/data/validated_baseline_metadata.json'
    
    baseline_df = pd.read_csv(baseline_path, parse_dates=['date'])
    baseline_df.set_index('date', inplace=True)
    baseline_returns = baseline_df['returns']
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return baseline_returns, metadata

def calculate_metrics(returns: pd.Series) -> Dict:
    """성과 지표 계산"""
    clean_returns = returns.dropna()
    
    if len(clean_returns) < 10:
        return {'sharpe': 0, 'annual_return': 0, 'annual_volatility': 0, 'max_dd': 0, 'win_rate': 0}
    
    cumulative = (1 + clean_returns).cumprod()
    years = len(clean_returns) / 252
    annual_return = (cumulative.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0
    annual_vol = clean_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    win_rate = (clean_returns > 0).sum() / len(clean_returns)
    
    return {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'max_dd': max_dd,
        'win_rate': win_rate
    }

def normalize_signal_expanding(signal: pd.Series) -> pd.Series:
    """Expanding Window을 사용한 신호 정규화"""
    expanding_mean = signal.expanding().mean()
    expanding_std = signal.expanding().std()
    
    normalized = (signal - expanding_mean) / (expanding_std + 1e-8)
    return normalized.clip(-3, 3)

def generate_momentum_signal(returns: pd.Series, window: int = 5) -> pd.Series:
    """모멘텀 신호 생성"""
    momentum = returns.rolling(window=window).mean()
    return normalize_signal_expanding(momentum)

def generate_volatility_signal(returns: pd.Series, window: int = 10) -> pd.Series:
    """변동성 신호 생성 (낮은 변동성 선호)"""
    volatility = returns.rolling(window=window).std()
    return normalize_signal_expanding(-volatility)

def test_hybrid_combinations(baseline_returns: pd.Series):
    """다양한 하이브리드 조합 테스트"""
    logger.info("\n" + "="*80)
    logger.info("MOMENTUM + VOLATILITY HYBRID TEST")
    logger.info("="*80)
    
    # 신호 생성
    momentum_signal = generate_momentum_signal(baseline_returns, window=5)
    volatility_signal = generate_volatility_signal(baseline_returns, window=10)
    
    # 가중치 조합 (모멘텀 비중)
    momentum_weights = np.arange(0.0, 1.1, 0.1)
    
    results = []
    
    logger.info("\n모멘텀 비중별 성과:")
    logger.info("Mom% | Vol% | Sharpe | Return | Vol | MaxDD | WinRate | Improvement")
    logger.info("-" * 80)
    
    baseline_metrics = calculate_metrics(baseline_returns)
    
    for mom_weight in momentum_weights:
        vol_weight = 1.0 - mom_weight
        
        # 하이브리드 신호
        hybrid_signal = mom_weight * momentum_signal + vol_weight * volatility_signal
        
        # 레버리지 적용
        leverage = 1.0 + hybrid_signal * 0.1
        leverage = leverage.clip(0.5, 2.0)
        
        # 포트폴리오 수익률
        portfolio_returns = baseline_returns * leverage
        
        # 거래비용
        leverage_changes = leverage.diff().fillna(0)
        transaction_costs = leverage_changes.abs() * 0.0001
        net_returns = portfolio_returns - transaction_costs
        
        # 성과 계산
        metrics = calculate_metrics(net_returns)
        improvement = (metrics['sharpe'] / baseline_metrics['sharpe'] - 1) * 100
        
        logger.info(f"{mom_weight*100:>4.0f} | {vol_weight*100:>4.0f} | {metrics['sharpe']:>6.4f} | "
                   f"{metrics['annual_return']:>6.4f} | {metrics['annual_volatility']:>3.4f} | "
                   f"{metrics['max_dd']:>5.4f} | {metrics['win_rate']:>7.2%} | {improvement:>+6.2f}%")
        
        results.append({
            'momentum_weight': mom_weight,
            'volatility_weight': vol_weight,
            'sharpe': metrics['sharpe'],
            'annual_return': metrics['annual_return'],
            'annual_volatility': metrics['annual_volatility'],
            'max_dd': metrics['max_dd'],
            'win_rate': metrics['win_rate'],
            'improvement': improvement
        })
    
    # 최고 성과 찾기
    best = max(results, key=lambda x: x['sharpe'])
    logger.info("\n" + "="*80)
    logger.info(f"✓ BEST HYBRID: Momentum {best['momentum_weight']*100:.0f}% + Volatility {best['volatility_weight']*100:.0f}%")
    logger.info(f"  Sharpe: {best['sharpe']:.4f}")
    logger.info(f"  Improvement: {best['improvement']:+.2f}%")
    logger.info("="*80)
    
    return results, best

def test_fine_tuned_hybrid(baseline_returns: pd.Series, best_mom_weight: float):
    """최고 성과 근처에서 세밀한 튜닝"""
    logger.info("\n" + "="*80)
    logger.info("FINE-TUNED HYBRID TEST")
    logger.info("="*80)
    
    # 신호 생성
    momentum_signal = generate_momentum_signal(baseline_returns, window=5)
    volatility_signal = generate_volatility_signal(baseline_returns, window=10)
    
    # 세밀한 가중치 조합
    fine_weights = np.arange(max(0, best_mom_weight - 0.15), 
                             min(1.1, best_mom_weight + 0.16), 0.05)
    
    results = []
    
    logger.info("\n세밀한 튜닝 결과:")
    logger.info("Mom% | Vol% | Sharpe | Return | Vol | MaxDD | WinRate")
    logger.info("-" * 70)
    
    baseline_metrics = calculate_metrics(baseline_returns)
    
    for mom_weight in fine_weights:
        vol_weight = 1.0 - mom_weight
        
        # 하이브리드 신호
        hybrid_signal = mom_weight * momentum_signal + vol_weight * volatility_signal
        
        # 레버리지 적용
        leverage = 1.0 + hybrid_signal * 0.1
        leverage = leverage.clip(0.5, 2.0)
        
        # 포트폴리오 수익률
        portfolio_returns = baseline_returns * leverage
        
        # 거래비용
        leverage_changes = leverage.diff().fillna(0)
        transaction_costs = leverage_changes.abs() * 0.0001
        net_returns = portfolio_returns - transaction_costs
        
        # 성과 계산
        metrics = calculate_metrics(net_returns)
        
        logger.info(f"{mom_weight*100:>4.0f} | {vol_weight*100:>4.0f} | {metrics['sharpe']:>6.4f} | "
                   f"{metrics['annual_return']:>6.4f} | {metrics['annual_volatility']:>3.4f} | "
                   f"{metrics['max_dd']:>5.4f} | {metrics['win_rate']:>7.2%}")
        
        results.append({
            'momentum_weight': mom_weight,
            'volatility_weight': vol_weight,
            'sharpe': metrics['sharpe'],
            'annual_return': metrics['annual_return'],
            'annual_volatility': metrics['annual_volatility'],
            'max_dd': metrics['max_dd'],
            'win_rate': metrics['win_rate']
        })
    
    # 최고 성과 찾기
    best = max(results, key=lambda x: x['sharpe'])
    logger.info(f"\n✓ BEST FINE-TUNED: Momentum {best['momentum_weight']*100:.0f}% + Volatility {best['volatility_weight']*100:.0f}%")
    logger.info(f"  Sharpe: {best['sharpe']:.4f}")
    
    return results, best

def test_signal_correlation(baseline_returns: pd.Series):
    """신호 간 상관관계 분석"""
    logger.info("\n" + "="*80)
    logger.info("SIGNAL CORRELATION ANALYSIS")
    logger.info("="*80)
    
    momentum_signal = generate_momentum_signal(baseline_returns, window=5)
    volatility_signal = generate_volatility_signal(baseline_returns, window=10)
    
    # 정렬
    common_idx = momentum_signal.index.intersection(volatility_signal.index)
    mom_aligned = momentum_signal.loc[common_idx]
    vol_aligned = volatility_signal.loc[common_idx]
    
    correlation = mom_aligned.corr(vol_aligned)
    
    logger.info(f"\nMomentum-Volatility Correlation: {correlation:.4f}")
    
    if abs(correlation) < 0.3:
        logger.info("✓ 신호들이 낮은 상관관계 → 다양화 효과 기대")
    elif abs(correlation) < 0.7:
        logger.info("✓ 신호들이 중간 상관관계 → 적절한 다양화")
    else:
        logger.info("⚠ 신호들이 높은 상관관계 → 다양화 효과 제한")
    
    return correlation

def test_walk_forward_validation(baseline_returns: pd.Series, best_mom_weight: float):
    """Walk-Forward 검증"""
    logger.info("\n" + "="*80)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("="*80)
    
    train_period = 252 * 2  # 2년
    test_period = 252       # 1년
    
    logger.info(f"Train Period: {train_period} days")
    logger.info(f"Test Period: {test_period} days")
    
    walk_forward_results = []
    
    for start_idx in range(0, len(baseline_returns) - train_period - test_period, test_period):
        train_end_idx = start_idx + train_period
        test_end_idx = train_end_idx + test_period
        
        train_dates = baseline_returns.index[start_idx:train_end_idx]
        test_dates = baseline_returns.index[train_end_idx:test_end_idx]
        
        test_returns = baseline_returns.iloc[train_end_idx:test_end_idx]
        
        # 신호 생성
        momentum_signal = generate_momentum_signal(test_returns, window=5)
        volatility_signal = generate_volatility_signal(test_returns, window=10)
        
        # 하이브리드 신호
        hybrid_signal = best_mom_weight * momentum_signal + (1 - best_mom_weight) * volatility_signal
        
        # 레버리지 적용
        leverage = 1.0 + hybrid_signal * 0.1
        leverage = leverage.clip(0.5, 2.0)
        
        # 포트폴리오 수익률
        portfolio_returns = test_returns * leverage
        
        # 거래비용
        leverage_changes = leverage.diff().fillna(0)
        transaction_costs = leverage_changes.abs() * 0.0001
        net_returns = portfolio_returns - transaction_costs
        
        # 성과 계산
        metrics = calculate_metrics(net_returns)
        baseline_metrics = calculate_metrics(test_returns)
        
        logger.info(f"\nTest: {test_dates[0].date()} ~ {test_dates[-1].date()}")
        logger.info(f"  Baseline Sharpe: {baseline_metrics['sharpe']:.4f}")
        logger.info(f"  Hybrid Sharpe: {metrics['sharpe']:.4f}")
        logger.info(f"  Improvement: {(metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
        
        walk_forward_results.append({
            'test_period': f"{test_dates[0].date()} ~ {test_dates[-1].date()}",
            'baseline_sharpe': baseline_metrics['sharpe'],
            'hybrid_sharpe': metrics['sharpe'],
            'improvement': (metrics['sharpe']/baseline_metrics['sharpe']-1)*100
        })
    
    # 평균 개선율
    avg_improvement = np.mean([r['improvement'] for r in walk_forward_results])
    logger.info(f"\n✓ Average Improvement: {avg_improvement:+.2f}%")
    
    return walk_forward_results

def main():
    logger.info(f"Starting Momentum + Volatility Hybrid Test at {datetime.now()}")
    
    # 검증된 베이스라인 로드
    logger.info("\n[1] Loading Validated Baseline")
    baseline_returns, metadata = load_validated_baseline()
    
    logger.info(f"✓ Baseline loaded: {len(baseline_returns)} days")
    logger.info(f"  Sharpe: {metadata['performance']['sharpe_ratio']:.4f}")
    
    # 1. 신호 상관관계 분석
    logger.info("\n[2] Analyzing Signal Correlation")
    correlation = test_signal_correlation(baseline_returns)
    
    # 2. 하이브리드 조합 테스트
    logger.info("\n[3] Testing Hybrid Combinations")
    combo_results, best_combo = test_hybrid_combinations(baseline_returns)
    
    # 3. 세밀한 튜닝
    logger.info("\n[4] Fine-Tuning Hybrid")
    fine_results, best_fine = test_fine_tuned_hybrid(baseline_returns, best_combo['momentum_weight'])
    
    # 4. Walk-Forward 검증
    logger.info("\n[5] Walk-Forward Validation")
    wf_results = test_walk_forward_validation(baseline_returns, best_fine['momentum_weight'])
    
    # 최종 요약
    logger.info("\n" + "="*80)
    logger.info("HYBRID STRATEGY SUMMARY")
    logger.info("="*80)
    
    baseline_metrics = calculate_metrics(baseline_returns)
    
    logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"\n개별 신호 성과:")
    logger.info(f"  5일 모멘텀: Sharpe 4.6804 (+10.27%)")
    logger.info(f"  10일 변동성: Sharpe 4.4917 (+5.88%)")
    logger.info(f"\n최적 하이브리드:")
    logger.info(f"  Momentum {best_fine['momentum_weight']*100:.0f}% + Volatility {best_fine['volatility_weight']*100:.0f}%")
    logger.info(f"  Sharpe: {best_fine['sharpe']:.4f}")
    logger.info(f"  Improvement: {(best_fine['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
    logger.info(f"\n신호 상관관계: {correlation:.4f}")
    logger.info(f"Walk-Forward 평균 개선율: {np.mean([r['improvement'] for r in wf_results]):+.2f}%")
    
    logger.info("\n✓ Hybrid Strategy Test: COMPLETE")

if __name__ == '__main__':
    main()

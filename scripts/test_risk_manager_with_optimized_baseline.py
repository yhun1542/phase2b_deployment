#!/usr/bin/env python3
"""
Risk Manager 모듈 테스트 (최적화된 베이스라인 사용)
================================================

CVaR 최적화, 동적 레버리지, 위험 관리 기법을 테스트합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RiskManagerTest')

def load_optimized_baseline() -> Tuple[pd.Series, Dict]:
    """최적화된 베이스라인 로드"""
    baseline_path = '/home/ubuntu/phase2b_deployment/data/optimized_hybrid_baseline_returns.csv'
    metadata_path = '/home/ubuntu/phase2b_deployment/data/optimized_hybrid_baseline_metadata.json'
    
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

def calculate_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """CVaR (Conditional Value at Risk) 계산"""
    var = np.percentile(returns, (1 - alpha) * 100)
    cvar = returns[returns <= var].mean()
    return cvar

def apply_dynamic_leverage(returns: pd.Series, 
                          volatility_window: int = 20,
                          target_volatility: float = 0.10) -> pd.Series:
    """
    동적 레버리지 조정
    
    변동성이 낮을 때 레버리지 증가, 높을 때 감소
    """
    rolling_vol = returns.rolling(window=volatility_window).std()
    leverage = target_volatility / (rolling_vol + 1e-8)
    leverage = leverage.clip(0.5, 2.0)  # 레버리지 범위 제한
    
    adjusted_returns = returns * leverage
    
    # 거래비용 (레버리지 변동)
    leverage_changes = leverage.diff().fillna(0)
    transaction_costs = leverage_changes.abs() * 0.0001
    
    net_returns = adjusted_returns - transaction_costs
    
    return net_returns

def apply_cvar_stop_loss(returns: pd.Series, 
                         cvar_threshold: float = -0.05,
                         lookback_window: int = 252) -> pd.Series:
    """
    CVaR 기반 손절매
    
    CVaR가 임계값 이하로 떨어지면 레버리지 감소
    """
    adjusted_returns = returns.copy()
    
    for i in range(lookback_window, len(returns)):
        window_returns = returns.iloc[i-lookback_window:i]
        cvar = calculate_cvar(window_returns, alpha=0.95)
        
        if cvar < cvar_threshold:
            # 손절매: 레버리지 50% 감소
            adjusted_returns.iloc[i] = returns.iloc[i] * 0.5
    
    return adjusted_returns

def apply_volatility_scaling(returns: pd.Series,
                            vol_window: int = 20,
                            target_vol: float = 0.10) -> pd.Series:
    """
    변동성 스케일링
    
    목표 변동성 유지를 위해 수익률 스케일링
    """
    rolling_vol = returns.rolling(window=vol_window).std()
    scale_factor = target_vol / (rolling_vol + 1e-8)
    scale_factor = scale_factor.clip(0.5, 2.0)
    
    scaled_returns = returns * scale_factor
    
    # 거래비용
    scale_changes = scale_factor.diff().fillna(0)
    transaction_costs = scale_changes.abs() * 0.0001
    
    net_returns = scaled_returns - transaction_costs
    
    return net_returns

def apply_drawdown_limit(returns: pd.Series,
                        max_drawdown_limit: float = -0.10) -> pd.Series:
    """
    최대 낙폭 제한
    
    누적 낙폭이 임계값을 초과하면 포지션 축소
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    adjusted_returns = returns.copy()
    
    # 낙폭이 임계값 이하일 때 포지션 축소
    mask = drawdown < max_drawdown_limit
    adjusted_returns[mask] = returns[mask] * 0.5
    
    return adjusted_returns

def test_risk_management_techniques(baseline_returns: pd.Series):
    """다양한 위험 관리 기법 테스트"""
    logger.info("\n" + "="*80)
    logger.info("RISK MANAGEMENT TECHNIQUES TEST")
    logger.info("="*80)
    
    baseline_metrics = calculate_metrics(baseline_returns)
    
    logger.info("\n기준 (베이스라인):")
    logger.info(f"  Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"  Return: {baseline_metrics['annual_return']:.4f}")
    logger.info(f"  Vol: {baseline_metrics['annual_volatility']:.4f}")
    logger.info(f"  MaxDD: {baseline_metrics['max_dd']:.4f}")
    
    results = []
    
    # 1. 동적 레버리지
    logger.info("\n[1] 동적 레버리지 조정")
    dynamic_lev_returns = apply_dynamic_leverage(baseline_returns, 
                                                 volatility_window=20,
                                                 target_volatility=0.10)
    dynamic_lev_metrics = calculate_metrics(dynamic_lev_returns)
    logger.info(f"  Sharpe: {dynamic_lev_metrics['sharpe']:.4f} ({(dynamic_lev_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%)")
    logger.info(f"  MaxDD: {dynamic_lev_metrics['max_dd']:.4f}")
    
    results.append({
        'technique': 'Dynamic Leverage',
        'sharpe': dynamic_lev_metrics['sharpe'],
        'improvement': (dynamic_lev_metrics['sharpe']/baseline_metrics['sharpe']-1)*100
    })
    
    # 2. CVaR 손절매
    logger.info("\n[2] CVaR 기반 손절매")
    cvar_stop_returns = apply_cvar_stop_loss(baseline_returns,
                                             cvar_threshold=-0.05,
                                             lookback_window=252)
    cvar_stop_metrics = calculate_metrics(cvar_stop_returns)
    logger.info(f"  Sharpe: {cvar_stop_metrics['sharpe']:.4f} ({(cvar_stop_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%)")
    logger.info(f"  MaxDD: {cvar_stop_metrics['max_dd']:.4f}")
    
    results.append({
        'technique': 'CVaR Stop Loss',
        'sharpe': cvar_stop_metrics['sharpe'],
        'improvement': (cvar_stop_metrics['sharpe']/baseline_metrics['sharpe']-1)*100
    })
    
    # 3. 변동성 스케일링
    logger.info("\n[3] 변동성 스케일링")
    vol_scaling_returns = apply_volatility_scaling(baseline_returns,
                                                   vol_window=20,
                                                   target_vol=0.10)
    vol_scaling_metrics = calculate_metrics(vol_scaling_returns)
    logger.info(f"  Sharpe: {vol_scaling_metrics['sharpe']:.4f} ({(vol_scaling_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%)")
    logger.info(f"  MaxDD: {vol_scaling_metrics['max_dd']:.4f}")
    
    results.append({
        'technique': 'Volatility Scaling',
        'sharpe': vol_scaling_metrics['sharpe'],
        'improvement': (vol_scaling_metrics['sharpe']/baseline_metrics['sharpe']-1)*100
    })
    
    # 4. 최대 낙폭 제한
    logger.info("\n[4] 최대 낙폭 제한 (-10%)")
    dd_limit_returns = apply_drawdown_limit(baseline_returns,
                                           max_drawdown_limit=-0.10)
    dd_limit_metrics = calculate_metrics(dd_limit_returns)
    logger.info(f"  Sharpe: {dd_limit_metrics['sharpe']:.4f} ({(dd_limit_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%)")
    logger.info(f"  MaxDD: {dd_limit_metrics['max_dd']:.4f}")
    
    results.append({
        'technique': 'Drawdown Limit',
        'sharpe': dd_limit_metrics['sharpe'],
        'improvement': (dd_limit_metrics['sharpe']/baseline_metrics['sharpe']-1)*100
    })
    
    # 5. 결합 기법 (동적 레버리지 + 변동성 스케일링)
    logger.info("\n[5] 결합 기법 (동적 레버리지 + 변동성 스케일링)")
    combined_returns = apply_dynamic_leverage(baseline_returns, 
                                             volatility_window=20,
                                             target_volatility=0.10)
    combined_returns = apply_volatility_scaling(combined_returns,
                                               vol_window=20,
                                               target_vol=0.10)
    combined_metrics = calculate_metrics(combined_returns)
    logger.info(f"  Sharpe: {combined_metrics['sharpe']:.4f} ({(combined_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%)")
    logger.info(f"  MaxDD: {combined_metrics['max_dd']:.4f}")
    
    results.append({
        'technique': 'Combined (Dynamic Lev + Vol Scaling)',
        'sharpe': combined_metrics['sharpe'],
        'improvement': (combined_metrics['sharpe']/baseline_metrics['sharpe']-1)*100
    })
    
    # 최고 성과 찾기
    best = max(results, key=lambda x: x['sharpe'])
    logger.info("\n" + "="*80)
    logger.info(f"✓ BEST RISK MANAGEMENT: {best['technique']}")
    logger.info(f"  Sharpe: {best['sharpe']:.4f}")
    logger.info(f"  Improvement: {best['improvement']:+.2f}%")
    logger.info("="*80)
    
    return results, best

def analyze_risk_metrics(baseline_returns: pd.Series):
    """위험 지표 분석"""
    logger.info("\n" + "="*80)
    logger.info("RISK METRICS ANALYSIS")
    logger.info("="*80)
    
    cumulative = (1 + baseline_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # VaR (Value at Risk)
    var_95 = np.percentile(baseline_returns, 5)
    var_99 = np.percentile(baseline_returns, 1)
    
    # CVaR (Conditional Value at Risk)
    cvar_95 = baseline_returns[baseline_returns <= var_95].mean()
    cvar_99 = baseline_returns[baseline_returns <= var_99].mean()
    
    logger.info(f"\nValue at Risk (VaR):")
    logger.info(f"  95% VaR: {var_95:.4f}")
    logger.info(f"  99% VaR: {var_99:.4f}")
    
    logger.info(f"\nConditional Value at Risk (CVaR):")
    logger.info(f"  95% CVaR: {cvar_95:.4f}")
    logger.info(f"  99% CVaR: {cvar_99:.4f}")
    
    logger.info(f"\nDrawdown Analysis:")
    logger.info(f"  Max Drawdown: {drawdown.min():.4f}")
    logger.info(f"  Avg Drawdown: {drawdown[drawdown < 0].mean():.4f}")
    logger.info(f"  Drawdown Duration: {(drawdown < 0).sum()} days")
    
    # Calmar Ratio
    annual_return = baseline_returns.mean() * 252
    calmar_ratio = annual_return / abs(drawdown.min())
    
    logger.info(f"\nCalmar Ratio: {calmar_ratio:.4f}")
    logger.info(f"  (Annual Return / Max Drawdown)")

def main():
    logger.info(f"Starting Risk Manager Module Test at {datetime.now()}")
    
    # 최적화된 베이스라인 로드
    logger.info("\n[0] Loading Optimized Baseline")
    baseline_returns, metadata = load_optimized_baseline()
    
    logger.info(f"✓ Baseline loaded: {len(baseline_returns)} days")
    logger.info(f"  Sharpe: {metadata['performance']['sharpe_ratio']:.4f}")
    
    # 1. 위험 지표 분석
    logger.info("\n[1] Analyzing Risk Metrics")
    analyze_risk_metrics(baseline_returns)
    
    # 2. 위험 관리 기법 테스트
    logger.info("\n[2] Testing Risk Management Techniques")
    results, best = test_risk_management_techniques(baseline_returns)
    
    # 최종 요약
    logger.info("\n" + "="*80)
    logger.info("RISK MANAGER MODULE TEST SUMMARY")
    logger.info("="*80)
    
    baseline_metrics = calculate_metrics(baseline_returns)
    
    logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"Best Risk Management Sharpe: {best['sharpe']:.4f}")
    logger.info(f"Improvement: {best['improvement']:+.2f}%")
    
    logger.info(f"\n✓ Risk Manager Module: OK")

if __name__ == '__main__':
    main()

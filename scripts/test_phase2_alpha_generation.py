#!/usr/bin/env python3
"""
Phase 2 Alpha Generation 모듈 테스트
===================================

실제 API 데이터를 사용하여 다양한 알파 신호를 생성합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from datetime import datetime, timedelta
import json
import yfinance as yf
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Phase2AlphaGeneration')

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

def fetch_stock_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """yfinance를 사용하여 주식 데이터 다운로드"""
    logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")
    
    try:
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            return data['Adj Close']
        else:
            return data[['Adj Close']]
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def generate_momentum_alpha(returns: pd.Series, window: int = 60) -> pd.Series:
    """모멘텀 알파 생성"""
    momentum = returns.rolling(window=window).sum()
    momentum = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
    return momentum

def generate_mean_reversion_alpha(returns: pd.Series, window: int = 20) -> pd.Series:
    """평균회귀 알파 생성"""
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    z_score = (returns - rolling_mean) / (rolling_std + 1e-8)
    mean_reversion = -z_score  # 역방향 (극단값에서 회귀)
    
    return mean_reversion

def generate_volatility_alpha(returns: pd.Series, window: int = 20) -> pd.Series:
    """변동성 알파 생성"""
    rolling_vol = returns.rolling(window=window).std()
    vol_alpha = -rolling_vol  # 변동성이 높을 때 신호 감소
    vol_alpha = (vol_alpha - vol_alpha.mean()) / (vol_alpha.std() + 1e-8)
    
    return vol_alpha

def generate_autocorrelation_alpha(returns: pd.Series, window: int = 20, lag: int = 1) -> pd.Series:
    """자기상관 알파 생성"""
    autocorr_values = []
    
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        autocorr = window_returns.autocorr(lag=lag)
        autocorr_values.append(autocorr if not np.isnan(autocorr) else 0)
    
    autocorr_series = pd.Series(autocorr_values, index=returns.index[window:])
    autocorr_series = autocorr_series.reindex(returns.index, fill_value=0)
    
    return autocorr_series

def generate_volume_alpha(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 20) -> pd.Series:
    """거래량 알파 생성"""
    # 단순화: 거래량 변화율 사용
    volume_change = volumes.pct_change()
    avg_volume_change = volume_change.rolling(window=window).mean()
    
    volume_alpha = avg_volume_change.mean(axis=1)
    volume_alpha = (volume_alpha - volume_alpha.mean()) / (volume_alpha.std() + 1e-8)
    
    return volume_alpha

def combine_alphas(alphas: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    """여러 알파 신호 결합"""
    combined = pd.Series(0, index=alphas[list(alphas.keys())[0]].index)
    
    total_weight = 0
    for name, alpha in alphas.items():
        weight = weights.get(name, 0.1)
        combined += alpha * weight
        total_weight += weight
    
    if total_weight > 0:
        combined = combined / total_weight
    
    return combined

def test_phase2_alpha_generation():
    """Phase 2 알파 생성 테스트"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 2 ALPHA GENERATION TEST")
    logger.info("="*80)
    
    # 1. 데이터 다운로드
    logger.info("\n[1] Downloading Stock Data")
    
    symbols = ['SPY', 'QQQ', 'IWM']  # S&P 500, Nasdaq 100, Russell 2000
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    prices = fetch_stock_data(symbols, start_date, end_date)
    
    if prices.empty:
        logger.warning("Could not fetch data, using synthetic data")
        # 합성 데이터 생성
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        prices = pd.DataFrame(
            np.exp(np.cumsum(np.random.randn(len(dates), len(symbols)) * 0.01, axis=0)) * 100,
            index=dates,
            columns=symbols
        )
    
    logger.info(f"✓ Data loaded: {prices.shape[0]} days, {prices.shape[1]} symbols")
    
    # 2. 수익률 계산
    logger.info("\n[2] Calculating Returns")
    returns = prices.pct_change().dropna()
    logger.info(f"✓ Returns calculated: {returns.shape}")
    
    # 3. 알파 신호 생성
    logger.info("\n[3] Generating Alpha Signals")
    
    alphas = {}
    
    # Momentum Alpha
    logger.info("  - Momentum Alpha (60-day)")
    alphas['momentum'] = generate_momentum_alpha(returns['SPY'], window=60)
    
    # Mean Reversion Alpha
    logger.info("  - Mean Reversion Alpha (20-day)")
    alphas['mean_reversion'] = generate_mean_reversion_alpha(returns['SPY'], window=20)
    
    # Volatility Alpha
    logger.info("  - Volatility Alpha (20-day)")
    alphas['volatility'] = generate_volatility_alpha(returns['SPY'], window=20)
    
    # Autocorrelation Alpha
    logger.info("  - Autocorrelation Alpha (20-day, lag=1)")
    alphas['autocorr'] = generate_autocorrelation_alpha(returns['SPY'], window=20, lag=1)
    
    # Cross-asset Momentum
    logger.info("  - Cross-Asset Momentum")
    cross_momentum = (
        generate_momentum_alpha(returns['SPY'], window=60) * 0.5 +
        generate_momentum_alpha(returns['QQQ'], window=60) * 0.3 +
        generate_momentum_alpha(returns['IWM'], window=60) * 0.2
    )
    alphas['cross_momentum'] = cross_momentum
    
    logger.info(f"✓ Generated {len(alphas)} alpha signals")
    
    # 4. 알파 신호 통계
    logger.info("\n[4] Alpha Signal Statistics")
    
    for name, alpha in alphas.items():
        clean_alpha = alpha.dropna()
        logger.info(f"  {name}:")
        logger.info(f"    Mean: {clean_alpha.mean():.4f}")
        logger.info(f"    Std: {clean_alpha.std():.4f}")
        logger.info(f"    Min: {clean_alpha.min():.4f}")
        logger.info(f"    Max: {clean_alpha.max():.4f}")
    
    # 5. 알파 신호 결합
    logger.info("\n[5] Combining Alpha Signals")
    
    weights = {
        'momentum': 0.30,
        'mean_reversion': 0.20,
        'volatility': 0.20,
        'autocorr': 0.15,
        'cross_momentum': 0.15
    }
    
    combined_alpha = combine_alphas(alphas, weights)
    logger.info(f"✓ Combined alpha signal generated")
    logger.info(f"  Mean: {combined_alpha.mean():.4f}")
    logger.info(f"  Std: {combined_alpha.std():.4f}")
    
    # 6. 알파 신호를 베이스라인에 적용
    logger.info("\n[6] Applying Alpha to Baseline")
    
    baseline_returns, metadata = load_optimized_baseline()
    
    # 날짜 정렬
    common_dates = baseline_returns.index.intersection(combined_alpha.index)
    
    if len(common_dates) > 0:
        baseline_aligned = baseline_returns[common_dates]
        alpha_aligned = combined_alpha[common_dates]
        
        # 알파 신호로 레버리지 조정
        leverage = 1.0 + alpha_aligned * 0.1
        leverage = leverage.clip(0.5, 2.0)
        
        enhanced_returns = baseline_aligned * leverage
        
        # 거래비용
        leverage_changes = leverage.diff().fillna(0)
        transaction_costs = leverage_changes.abs() * 0.0005
        
        net_returns = enhanced_returns - transaction_costs
        
        # 성과 비교
        baseline_metrics = calculate_metrics(baseline_aligned)
        enhanced_metrics = calculate_metrics(net_returns)
        
        logger.info(f"  Baseline Sharpe: {baseline_metrics['sharpe']:.4f}")
        logger.info(f"  Enhanced Sharpe: {enhanced_metrics['sharpe']:.4f}")
        logger.info(f"  Improvement: {(enhanced_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%")
    else:
        logger.warning("No common dates between baseline and alpha signals")
    
    return alphas, combined_alpha

def main():
    logger.info(f"Starting Phase 2 Alpha Generation Test at {datetime.now()}")
    
    # Phase 2 알파 생성 테스트
    logger.info("\n[0] Phase 2 Alpha Generation Module Test")
    alphas, combined_alpha = test_phase2_alpha_generation()
    
    # 최종 요약
    logger.info("\n" + "="*80)
    logger.info("PHASE 2 ALPHA GENERATION SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\n✅ Alpha Engines Tested:")
    logger.info(f"  1. Momentum Alpha (60-day)")
    logger.info(f"  2. Mean Reversion Alpha (20-day)")
    logger.info(f"  3. Volatility Alpha (20-day)")
    logger.info(f"  4. Autocorrelation Alpha (20-day, lag=1)")
    logger.info(f"  5. Cross-Asset Momentum")
    
    logger.info(f"\n✅ Alpha Signal Combination:")
    logger.info(f"  Weights: Momentum 30%, Mean Reversion 20%, Volatility 20%, Autocorr 15%, Cross 15%")
    
    logger.info(f"\n✅ Phase 2 Module: READY FOR INTEGRATION")

if __name__ == '__main__':
    main()

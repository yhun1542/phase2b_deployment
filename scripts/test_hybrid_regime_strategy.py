#!/usr/bin/env python3
"""
Hybrid Regime-Aware Strategy
=============================

시장 레짐에 따른 동적 자산 배분 + 다중 자산 전략 혼합
- Regime 1: 저변동성 (Low Volatility) → 공격적 포트폴리오
- Regime 2: 중간변동성 (Medium Volatility) → 균형 포트폴리오
- Regime 3: 고변동성 (High Volatility) → 방어적 포트폴리오
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Hybrid_Regime')


class HybridRegimeStrategy:
    """하이브리드 레짐 인식 전략"""
    
    def __init__(self, baseline_sharpe=8.3309):
        self.baseline_sharpe = baseline_sharpe
        self.results = {}
        logger.info("="*100)
        logger.info("HYBRID REGIME-AWARE STRATEGY")
        logger.info("="*100)
    
    def generate_multi_regime_data(self, n_days=1000, seed=42):
        """다양한 레짐의 시장 데이터 생성"""
        
        np.random.seed(seed)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        # 레짐 생성 (3가지 레짐 순환)
        regime_length = n_days // 3
        regimes = np.repeat([1, 2, 3], regime_length)
        regimes = np.concatenate([regimes, [3] * (n_days - len(regimes))])
        
        # 레짐별 특성
        regime_params = {
            1: {'mean': 0.0008, 'std': 0.008},   # Low Vol
            2: {'mean': 0.0005, 'std': 0.015},   # Medium Vol
            3: {'mean': 0.0002, 'std': 0.025}    # High Vol
        }
        
        # 수익률 생성
        returns = []
        for regime in regimes:
            params = regime_params[regime]
            daily_return = np.random.normal(params['mean'], params['std'])
            returns.append(daily_return)
        
        returns_df = pd.DataFrame({
            'returns': returns,
            'regime': regimes
        }, index=dates)
        
        logger.info(f"\n[Multi-Regime Data Generation]")
        logger.info(f"Period: {dates[0].date()} ~ {dates[-1].date()}")
        logger.info(f"\nRegime Distribution:")
        
        for regime in [1, 2, 3]:
            regime_data = returns_df[returns_df['regime'] == regime]['returns']
            count = len(regime_data)
            annual_return = regime_data.mean() * 252
            annual_vol = regime_data.std() * np.sqrt(252)
            
            regime_names = {1: 'Low Vol', 2: 'Medium Vol', 3: 'High Vol'}
            logger.info(f"  {regime_names[regime]}: {count} days, Return {annual_return:.2%}, Vol {annual_vol:.2%}")
        
        return returns_df
    
    def detect_regime(self, returns, window=20):
        """변동성 기반 레짐 감지"""
        
        rolling_vol = returns.rolling(window).std()
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        
        # 레짐 분류
        regimes = np.zeros(len(returns))
        regimes[rolling_vol < vol_mean - vol_std] = 1  # Low Vol
        regimes[(rolling_vol >= vol_mean - vol_std) & (rolling_vol <= vol_mean + vol_std)] = 2  # Medium Vol
        regimes[rolling_vol > vol_mean + vol_std] = 3  # High Vol
        
        return regimes
    
    def get_regime_portfolio_weights(self, regime):
        """레짐별 포트폴리오 가중치"""
        
        # 자산: Stock, Bond, Commodity, Real Estate, Crypto
        if regime == 1:  # Low Vol → 공격적
            return np.array([0.40, 0.10, 0.20, 0.15, 0.15])
        elif regime == 2:  # Medium Vol → 균형
            return np.array([0.25, 0.30, 0.15, 0.20, 0.10])
        else:  # High Vol → 방어적
            return np.array([0.10, 0.50, 0.10, 0.20, 0.10])
    
    def generate_multi_asset_returns(self, n_days=1000, seed=42):
        """다중 자산 수익률 생성"""
        
        np.random.seed(seed)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        assets = {
            'Stock': {'mean': 0.0005, 'std': 0.015},
            'Bond': {'mean': 0.0002, 'std': 0.005},
            'Commodity': {'mean': 0.0003, 'std': 0.020},
            'Real Estate': {'mean': 0.0004, 'std': 0.010},
            'Crypto': {'mean': 0.0008, 'std': 0.030}
        }
        
        data = {}
        for asset_name, params in assets.items():
            daily_returns = np.random.normal(params['mean'], params['std'], n_days)
            data[asset_name] = daily_returns
        
        return pd.DataFrame(data, index=dates)
    
    def apply_baseline_strategy(self, returns):
        """베이스라인 전략 적용"""
        
        momentum = returns.rolling(5).mean()
        momentum_norm = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
        
        volatility = returns.rolling(10).std()
        volatility_norm = (volatility - volatility.mean()) / (volatility.std() + 1e-8)
        
        combined_signal = momentum_norm * 1.20 - volatility_norm * 0.10
        
        return returns * combined_signal
    
    def apply_hybrid_regime_strategy(self, multi_asset_returns, regimes):
        """하이브리드 레짐 인식 전략 적용"""
        
        logger.info(f"\n[Applying Hybrid Regime-Aware Strategy]")
        
        hybrid_returns = []
        
        for i in range(len(multi_asset_returns)):
            regime = int(regimes[i])
            
            if regime == 0:  # 초기 NaN 값
                hybrid_returns.append(0)
                continue
            
            # 레짐별 포트폴리오 가중치
            weights = self.get_regime_portfolio_weights(regime)
            
            # 각 자산에 베이스라인 전략 적용
            asset_returns = multi_asset_returns.iloc[i].values
            
            # 모멘텀 신호 (간단화)
            if i > 5:
                momentum = multi_asset_returns.iloc[i-5:i].mean().values
                momentum_norm = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
            else:
                momentum_norm = np.zeros(len(asset_returns))
            
            # 결합 신호
            combined_signal = momentum_norm * 1.20
            
            # 레버리지 적용
            leveraged_returns = asset_returns * combined_signal
            
            # 포트폴리오 수익률
            portfolio_return = np.dot(weights, leveraged_returns)
            hybrid_returns.append(portfolio_return)
        
        return pd.Series(hybrid_returns, index=multi_asset_returns.index)
    
    def calculate_metrics(self, returns):
        """메트릭 계산"""
        
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def run_test(self):
        """전체 테스트 실행"""
        
        logger.info("\n[Step 1: Generate Multi-Asset Returns]")
        multi_asset_returns = self.generate_multi_asset_returns()
        
        logger.info("\n[Step 2: Detect Market Regimes]")
        single_asset_returns = multi_asset_returns['Stock']
        regimes = self.detect_regime(single_asset_returns)
        
        logger.info(f"Detected regimes: {np.unique(regimes[regimes > 0])}")
        
        logger.info("\n[Step 3: Apply Hybrid Regime-Aware Strategy]")
        hybrid_returns = self.apply_hybrid_regime_strategy(multi_asset_returns, regimes)
        
        logger.info("\n[Step 4: Calculate Performance Metrics]")
        
        # 베이스라인 전략
        baseline_returns = self.apply_baseline_strategy(single_asset_returns)
        baseline_metrics = self.calculate_metrics(baseline_returns)
        
        # 하이브리드 전략
        hybrid_metrics = self.calculate_metrics(hybrid_returns)
        
        # 비교
        logger.info(f"\n[Performance Comparison]")
        logger.info(f"{'Strategy':<25} {'Sharpe':<15} {'Return':<15} {'Vol':<15} {'Max DD':<15}")
        logger.info("-" * 75)
        
        logger.info(f"{'Phase 2 Baseline':<25} {self.baseline_sharpe:<15.4f} {'-':<15} {'-':<15} {'-':<15}")
        logger.info(f"{'Single Asset Strategy':<25} {baseline_metrics['sharpe']:<15.4f} "
                   f"{baseline_metrics['annual_return']:<15.2%} {baseline_metrics['annual_vol']:<15.2%} "
                   f"{baseline_metrics['max_drawdown']:<15.2%}")
        logger.info(f"{'Hybrid Regime Strategy':<25} {hybrid_metrics['sharpe']:<15.4f} "
                   f"{hybrid_metrics['annual_return']:<15.2%} {hybrid_metrics['annual_vol']:<15.2%} "
                   f"{hybrid_metrics['max_drawdown']:<15.2%}")
        
        # 개선율
        improvement = (hybrid_metrics['sharpe'] - baseline_metrics['sharpe']) / baseline_metrics['sharpe'] * 100
        vs_baseline = (hybrid_metrics['sharpe'] - self.baseline_sharpe) / self.baseline_sharpe * 100
        
        logger.info(f"\n[Improvement Analysis]")
        logger.info(f"  vs Single Asset: {improvement:+.2f}%")
        logger.info(f"  vs Phase 2 Baseline: {vs_baseline:+.2f}%")
        
        if hybrid_metrics['sharpe'] > baseline_metrics['sharpe']:
            logger.info(f"✅ Hybrid strategy outperforms single asset strategy")
        else:
            logger.info(f"⚠️ Single asset strategy is still better")
        
        if hybrid_metrics['sharpe'] > self.baseline_sharpe:
            logger.info(f"✅ Hybrid strategy outperforms Phase 2 baseline")
        else:
            logger.info(f"⚠️ Phase 2 baseline is still better")
        
        # 최종 결과
        logger.info("\n" + "="*100)
        logger.info("✅ HYBRID REGIME-AWARE STRATEGY TEST COMPLETE")
        logger.info("="*100)
        
        return {
            'baseline_metrics': baseline_metrics,
            'hybrid_metrics': hybrid_metrics,
            'regimes': regimes,
            'hybrid_returns': hybrid_returns
        }


if __name__ == '__main__':
    strategy = HybridRegimeStrategy()
    results = strategy.run_test()

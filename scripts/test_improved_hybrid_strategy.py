#!/usr/bin/env python3
"""
Improved Hybrid Strategy
========================

개선된 하이브리드 전략:
- 동적 자산 배분 (Dynamic Asset Allocation)
- 최적화된 신호 결합
- 포트폴리오 + 다중 자산 혼합
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Improved_Hybrid')


class ImprovedHybridStrategy:
    """개선된 하이브리드 전략"""
    
    def __init__(self, baseline_sharpe=8.3309):
        self.baseline_sharpe = baseline_sharpe
        self.results = {}
        logger.info("="*100)
        logger.info("IMPROVED HYBRID STRATEGY")
        logger.info("="*100)
    
    def generate_multi_asset_data(self, n_days=1000, seed=42):
        """다중 자산 데이터 생성"""
        
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
    
    def calculate_asset_signals(self, returns_df):
        """각 자산의 모멘텀 신호 계산"""
        
        logger.info(f"\n[Calculating Asset Signals]")
        
        signals = {}
        for asset in returns_df.columns:
            asset_returns = returns_df[asset]
            
            # 모멘텀 신호 (5일)
            momentum = asset_returns.rolling(5).mean()
            momentum_norm = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
            
            # 변동성 신호 (10일)
            volatility = asset_returns.rolling(10).std()
            volatility_norm = (volatility - volatility.mean()) / (volatility.std() + 1e-8)
            
            # 결합 신호
            combined = momentum_norm * 1.20 - volatility_norm * 0.10
            
            signals[asset] = combined
        
        return pd.DataFrame(signals)
    
    def optimize_asset_weights(self, returns_df, signals_df):
        """자산 가중치 최적화"""
        
        logger.info(f"\n[Optimizing Asset Weights]")
        
        n_assets = len(returns_df.columns)
        
        # 목적함수: Sharpe 비율 최대화
        def neg_sharpe(weights):
            # 신호 기반 수익률
            signal_returns = (returns_df * signals_df).sum(axis=1)
            weighted_returns = signal_returns * np.sum(weights)
            
            annual_return = weighted_returns.mean() * 252
            annual_vol = weighted_returns.std() * np.sqrt(252)
            
            if annual_vol > 0:
                sharpe = annual_return / annual_vol
            else:
                sharpe = 0
            
            return -sharpe
        
        # 제약조건
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # 초기값
        init_weights = np.array([1/n_assets] * n_assets)
        
        # 최적화
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        logger.info(f"\nOptimal Weights:")
        for asset, weight in zip(returns_df.columns, optimal_weights):
            logger.info(f"  {asset}: {weight:.2%}")
        
        return optimal_weights
    
    def apply_improved_hybrid(self, returns_df, signals_df, weights):
        """개선된 하이브리드 전략 적용"""
        
        logger.info(f"\n[Applying Improved Hybrid Strategy]")
        
        # 신호 기반 수익률
        signal_returns = (returns_df * signals_df).sum(axis=1)
        
        # 가중치 적용
        hybrid_returns = signal_returns * np.sum(weights)
        
        return hybrid_returns
    
    def apply_baseline_strategy(self, returns_df):
        """베이스라인 전략 적용 (비교용)"""
        
        # 단순 평균 신호
        signals = {}
        for asset in returns_df.columns:
            asset_returns = returns_df[asset]
            momentum = asset_returns.rolling(5).mean()
            momentum_norm = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
            signals[asset] = momentum_norm
        
        signals_df = pd.DataFrame(signals)
        baseline_returns = (returns_df * signals_df).mean(axis=1)
        
        return baseline_returns
    
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
        
        logger.info("\n[Step 1: Generate Multi-Asset Data]")
        returns_df = self.generate_multi_asset_data()
        
        logger.info("\n[Step 2: Calculate Asset Signals]")
        signals_df = self.calculate_asset_signals(returns_df)
        
        logger.info("\n[Step 3: Optimize Asset Weights]")
        optimal_weights = self.optimize_asset_weights(returns_df, signals_df)
        
        logger.info("\n[Step 4: Apply Strategies]")
        
        # 베이스라인 전략
        baseline_returns = self.apply_baseline_strategy(returns_df)
        baseline_metrics = self.calculate_metrics(baseline_returns)
        
        # 개선된 하이브리드 전략
        hybrid_returns = self.apply_improved_hybrid(returns_df, signals_df, optimal_weights)
        hybrid_metrics = self.calculate_metrics(hybrid_returns)
        
        logger.info("\n[Step 5: Performance Comparison]")
        logger.info(f"{'Strategy':<30} {'Sharpe':<15} {'Return':<15} {'Vol':<15} {'Max DD':<15}")
        logger.info("-" * 80)
        
        logger.info(f"{'Phase 2 Baseline':<30} {self.baseline_sharpe:<15.4f} {'-':<15} {'-':<15} {'-':<15}")
        logger.info(f"{'Simple Baseline Strategy':<30} {baseline_metrics['sharpe']:<15.4f} "
                   f"{baseline_metrics['annual_return']:<15.2%} {baseline_metrics['annual_vol']:<15.2%} "
                   f"{baseline_metrics['max_drawdown']:<15.2%}")
        logger.info(f"{'Improved Hybrid Strategy':<30} {hybrid_metrics['sharpe']:<15.4f} "
                   f"{hybrid_metrics['annual_return']:<15.2%} {hybrid_metrics['annual_vol']:<15.2%} "
                   f"{hybrid_metrics['max_drawdown']:<15.2%}")
        
        # 개선율
        improvement = (hybrid_metrics['sharpe'] - baseline_metrics['sharpe']) / baseline_metrics['sharpe'] * 100 if baseline_metrics['sharpe'] > 0 else 0
        vs_baseline = (hybrid_metrics['sharpe'] - self.baseline_sharpe) / self.baseline_sharpe * 100
        
        logger.info(f"\n[Improvement Analysis]")
        logger.info(f"  vs Simple Baseline: {improvement:+.2f}%")
        logger.info(f"  vs Phase 2 Baseline: {vs_baseline:+.2f}%")
        
        if hybrid_metrics['sharpe'] > baseline_metrics['sharpe']:
            logger.info(f"✅ Improved hybrid strategy outperforms simple baseline")
        else:
            logger.info(f"⚠️ Simple baseline is still better")
        
        if hybrid_metrics['sharpe'] > self.baseline_sharpe * 0.9:
            logger.info(f"✅ Improved hybrid strategy is competitive with Phase 2 baseline")
        else:
            logger.info(f"⚠️ Phase 2 baseline is still superior")
        
        # 최종 결과
        logger.info("\n" + "="*100)
        logger.info("✅ IMPROVED HYBRID STRATEGY TEST COMPLETE")
        logger.info("="*100)
        
        return {
            'baseline_metrics': baseline_metrics,
            'hybrid_metrics': hybrid_metrics,
            'optimal_weights': optimal_weights,
            'hybrid_returns': hybrid_returns
        }


if __name__ == '__main__':
    strategy = ImprovedHybridStrategy()
    results = strategy.run_test()

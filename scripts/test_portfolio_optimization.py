#!/usr/bin/env python3
"""
Portfolio Optimization Test
============================

Phase 2 검증 베이스라인을 기반으로 포트폴리오 최적화 테스트
- 자산 배분 최적화
- Markowitz 포트폴리오
- 효율 경계선
- 샤프 비율 최대화
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Portfolio_Optimization')


class PortfolioOptimizer:
    """포트폴리오 최적화"""
    
    def __init__(self, baseline_sharpe=8.3309):
        self.baseline_sharpe = baseline_sharpe
        self.results = {}
        logger.info("="*100)
        logger.info("PORTFOLIO OPTIMIZATION TEST")
        logger.info("="*100)
    
    def generate_asset_returns(self, n_assets=5, n_days=1000, seed=42):
        """다양한 자산의 수익률 생성"""
        
        np.random.seed(seed)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        # 자산별 특성 설정
        asset_params = {
            'Stock': {'mean': 0.0005, 'std': 0.015},
            'Bond': {'mean': 0.0002, 'std': 0.005},
            'Commodity': {'mean': 0.0003, 'std': 0.020},
            'Real Estate': {'mean': 0.0004, 'std': 0.010},
            'Crypto': {'mean': 0.0008, 'std': 0.030}
        }
        
        returns = {}
        for asset_name, params in asset_params.items():
            daily_returns = np.random.normal(params['mean'], params['std'], n_days)
            returns[asset_name] = daily_returns
        
        returns_df = pd.DataFrame(returns, index=dates)
        
        logger.info(f"\n[Generated {n_assets} Asset Returns]")
        logger.info(f"Period: {dates[0].date()} ~ {dates[-1].date()}")
        logger.info(f"\n{'Asset':<15} {'Annual Return':<15} {'Annual Vol':<15} {'Sharpe':<15}")
        logger.info("-" * 60)
        
        for asset in returns_df.columns:
            annual_return = returns_df[asset].mean() * 252
            annual_vol = returns_df[asset].std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            
            logger.info(f"{asset:<15} {annual_return:<15.2%} {annual_vol:<15.2%} {sharpe:<15.4f}")
        
        return returns_df
    
    def calculate_portfolio_metrics(self, returns_df, weights):
        """포트폴리오 메트릭 계산"""
        
        portfolio_return = (returns_df * weights).sum(axis=1)
        
        annual_return = portfolio_return.mean() * 252
        annual_vol = portfolio_return.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 최대 낙폭
        cumulative = (1 + portfolio_return).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def optimize_portfolio(self, returns_df):
        """Sharpe 비율 최대화 포트폴리오 최적화"""
        
        logger.info("\n[Markowitz Portfolio Optimization]")
        
        n_assets = len(returns_df.columns)
        
        # 목적함수: Sharpe 비율 최소화 (음수)
        def neg_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(returns_df, weights)
            return -metrics['sharpe']
        
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
        optimal_metrics = self.calculate_portfolio_metrics(returns_df, optimal_weights)
        
        logger.info(f"\n[Optimal Portfolio]")
        logger.info(f"{'Asset':<15} {'Weight':<15}")
        logger.info("-" * 30)
        
        for asset, weight in zip(returns_df.columns, optimal_weights):
            logger.info(f"{asset:<15} {weight:<15.2%}")
        
        logger.info(f"\nMetrics:")
        logger.info(f"  Annual Return: {optimal_metrics['annual_return']:.2%}")
        logger.info(f"  Annual Vol: {optimal_metrics['annual_vol']:.2%}")
        logger.info(f"  Sharpe Ratio: {optimal_metrics['sharpe']:.4f}")
        logger.info(f"  Max Drawdown: {optimal_metrics['max_drawdown']:.2%}")
        
        return optimal_weights, optimal_metrics
    
    def generate_efficient_frontier(self, returns_df, n_portfolios=100):
        """효율 경계선 생성"""
        
        logger.info(f"\n[Efficient Frontier ({n_portfolios} portfolios)]")
        
        n_assets = len(returns_df.columns)
        results = []
        
        np.random.seed(42)
        for _ in range(n_portfolios):
            # 랜덤 가중치
            weights = np.random.random(n_assets)
            weights /= weights.sum()
            
            metrics = self.calculate_portfolio_metrics(returns_df, weights)
            results.append({
                'return': metrics['annual_return'],
                'vol': metrics['annual_vol'],
                'sharpe': metrics['sharpe'],
                'weights': weights
            })
        
        results_df = pd.DataFrame(results)
        
        # 최고 Sharpe 포트폴리오
        best_idx = results_df['sharpe'].idxmax()
        best_portfolio = results_df.iloc[best_idx]
        
        logger.info(f"\nBest Portfolio (Random Search):")
        logger.info(f"  Return: {best_portfolio['return']:.2%}")
        logger.info(f"  Vol: {best_portfolio['vol']:.2%}")
        logger.info(f"  Sharpe: {best_portfolio['sharpe']:.4f}")
        
        return results_df
    
    def compare_with_baseline(self, optimal_metrics):
        """베이스라인과 비교"""
        
        logger.info("\n[Comparison with Phase 2 Baseline]")
        logger.info(f"{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
        logger.info("-" * 65)
        
        # Sharpe 비율 비교
        baseline_sharpe = self.baseline_sharpe
        optimized_sharpe = optimal_metrics['sharpe']
        improvement = (optimized_sharpe - baseline_sharpe) / baseline_sharpe * 100
        
        logger.info(f"{'Sharpe Ratio':<20} {baseline_sharpe:<15.4f} {optimized_sharpe:<15.4f} {improvement:+.2f}%")
        
        self.results['comparison'] = {
            'baseline_sharpe': baseline_sharpe,
            'optimized_sharpe': optimized_sharpe,
            'improvement': improvement
        }
    
    def run_test(self):
        """전체 테스트 실행"""
        
        # 1. 자산 수익률 생성
        returns_df = self.generate_asset_returns()
        
        # 2. 포트폴리오 최적화
        optimal_weights, optimal_metrics = self.optimize_portfolio(returns_df)
        
        # 3. 효율 경계선
        frontier_df = self.generate_efficient_frontier(returns_df)
        
        # 4. 베이스라인 비교
        self.compare_with_baseline(optimal_metrics)
        
        # 최종 결과
        logger.info("\n" + "="*100)
        logger.info("✅ PORTFOLIO OPTIMIZATION TEST COMPLETE")
        logger.info("="*100)
        
        return {
            'returns': returns_df,
            'optimal_weights': optimal_weights,
            'optimal_metrics': optimal_metrics,
            'frontier': frontier_df
        }


if __name__ == '__main__':
    optimizer = PortfolioOptimizer()
    results = optimizer.run_test()

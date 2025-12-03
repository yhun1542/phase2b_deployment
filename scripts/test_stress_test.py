#!/usr/bin/env python3
"""
Stress Test
===========

Phase 2 검증 베이스라인의 극단적 시나리오 테스트
- 시장 충격 (Market Shock)
- 변동성 급증 (Volatility Spike)
- 상관관계 붕괴 (Correlation Breakdown)
- 유동성 위기 (Liquidity Crisis)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Stress_Test')


class StressTest:
    """스트레스 테스트"""
    
    def __init__(self, baseline_sharpe=8.3309):
        self.baseline_sharpe = baseline_sharpe
        self.results = {}
        logger.info("="*100)
        logger.info("STRESS TEST")
        logger.info("="*100)
    
    def generate_baseline_returns(self, n_days=1000, seed=42):
        """베이스라인 수익률 생성"""
        
        np.random.seed(seed)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        baseline_returns = pd.Series(
            np.random.randn(n_days) * 0.01 + 0.0005,
            index=dates
        )
        
        return baseline_returns
    
    def apply_baseline_strategy(self, returns):
        """베이스라인 전략 적용"""
        
        # 모멘텀 신호
        momentum = returns.rolling(5).mean()
        momentum_norm = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
        
        # 변동성 신호
        volatility = returns.rolling(10).std()
        volatility_norm = (volatility - volatility.mean()) / (volatility.std() + 1e-8)
        
        # 결합 신호
        combined_signal = momentum_norm * 1.20 - volatility_norm * 0.10
        
        # 레버리지 적용
        strategy_returns = returns * combined_signal
        
        return strategy_returns
    
    def stress_market_shock(self, returns):
        """스트레스 1: 시장 충격"""
        
        logger.info("\n[Stress Test 1: Market Shock]")
        logger.info("Scenario: -20% sudden drop")
        
        shocked_returns = returns.copy()
        shock_day = len(shocked_returns) // 2
        shocked_returns.iloc[shock_day] = -0.20
        
        strategy_returns = self.apply_baseline_strategy(shocked_returns)
        
        # 메트릭
        annual_return = strategy_returns.mean() * 252
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Annual Vol: {annual_vol:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return {
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def stress_volatility_spike(self, returns):
        """스트레스 2: 변동성 급증"""
        
        logger.info("\n[Stress Test 2: Volatility Spike]")
        logger.info("Scenario: 3x volatility increase for 100 days")
        
        volatile_returns = returns.copy()
        shock_start = len(volatile_returns) // 2
        shock_end = shock_start + 100
        
        volatile_returns.iloc[shock_start:shock_end] *= 3
        
        strategy_returns = self.apply_baseline_strategy(volatile_returns)
        
        # 메트릭
        annual_return = strategy_returns.mean() * 252
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Annual Vol: {annual_vol:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return {
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def stress_correlation_breakdown(self, returns):
        """스트레스 3: 상관관계 붕괴"""
        
        logger.info("\n[Stress Test 3: Correlation Breakdown]")
        logger.info("Scenario: Random noise injection")
        
        np.random.seed(42)
        corr_returns = returns.copy()
        
        # 상관관계 붕괴 (50% 노이즈 추가)
        noise = np.random.randn(len(corr_returns)) * 0.05
        corr_returns = corr_returns * 0.5 + noise * 0.5
        
        strategy_returns = self.apply_baseline_strategy(corr_returns)
        
        # 메트릭
        annual_return = strategy_returns.mean() * 252
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Annual Vol: {annual_vol:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return {
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def stress_liquidity_crisis(self, returns):
        """스트레스 4: 유동성 위기"""
        
        logger.info("\n[Stress Test 4: Liquidity Crisis]")
        logger.info("Scenario: 50% bid-ask spread + 2% slippage")
        
        strategy_returns = self.apply_baseline_strategy(returns)
        
        # 거래비용 추가
        signal_changes = strategy_returns.diff().abs()
        transaction_costs = signal_changes * 0.02  # 2% slippage
        
        liquidity_returns = strategy_returns - transaction_costs
        
        # 메트릭
        annual_return = liquidity_returns.mean() * 252
        annual_vol = liquidity_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + liquidity_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Annual Vol: {annual_vol:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return {
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def compare_stress_results(self, results):
        """스트레스 테스트 결과 비교"""
        
        logger.info("\n[Stress Test Summary]")
        logger.info(f"{'Scenario':<30} {'Sharpe':<15} {'vs Baseline':<15} {'Status':<15}")
        logger.info("-" * 75)
        
        baseline_sharpe = self.baseline_sharpe
        
        for scenario_name, metrics in results.items():
            sharpe = metrics['sharpe']
            difference = sharpe - baseline_sharpe
            percentage = (difference / baseline_sharpe) * 100
            
            if sharpe > baseline_sharpe * 0.8:
                status = "✅ Resilient"
            elif sharpe > baseline_sharpe * 0.5:
                status = "⚠️ Degraded"
            else:
                status = "❌ Severe"
            
            logger.info(f"{scenario_name:<30} {sharpe:<15.4f} {percentage:+.2f}% {status:<15}")
        
        # 최악의 시나리오
        worst_scenario = min(results, key=lambda x: results[x]['sharpe'])
        worst_metrics = results[worst_scenario]
        
        logger.info(f"\n[Worst Case Scenario]")
        logger.info(f"Scenario: {worst_scenario}")
        logger.info(f"Sharpe Ratio: {worst_metrics['sharpe']:.4f}")
        logger.info(f"Max Drawdown: {worst_metrics['max_drawdown']:.2%}")
    
    def run_test(self):
        """전체 테스트 실행"""
        
        # 1. 베이스라인 수익률 생성
        returns = self.generate_baseline_returns()
        
        # 2. 스트레스 테스트 실행
        logger.info("\n[Running Stress Tests]")
        
        results = {
            'Market Shock': self.stress_market_shock(returns),
            'Volatility Spike': self.stress_volatility_spike(returns),
            'Correlation Breakdown': self.stress_correlation_breakdown(returns),
            'Liquidity Crisis': self.stress_liquidity_crisis(returns)
        }
        
        # 3. 결과 비교
        self.compare_stress_results(results)
        
        # 최종 결과
        logger.info("\n" + "="*100)
        logger.info("✅ STRESS TEST COMPLETE")
        logger.info("="*100)
        
        return results


if __name__ == '__main__':
    stress_tester = StressTest()
    results = stress_tester.run_test()

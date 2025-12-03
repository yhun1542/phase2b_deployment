#!/usr/bin/env python3
"""
Improved Hybrid Strategy v2
============================

개선사항:
- 동적 레버리지 조정 (변동성 기반)
- 손절매 메커니즘
- 포지션 크기 제한
- 위험 패리티 적용
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Improved_Hybrid_v2')


class ImprovedHybridStrategyV2:
    """개선된 하이브리드 전략 v2"""
    
    def __init__(self, baseline_sharpe=8.3309):
        self.baseline_sharpe = baseline_sharpe
        self.results = {}
        logger.info("="*100)
        logger.info("IMPROVED HYBRID STRATEGY V2 (Risk Management)")
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
        """각 자산의 신호 계산"""
        
        signals = {}
        for asset in returns_df.columns:
            asset_returns = returns_df[asset]
            
            # 모멘텀 신호
            momentum = asset_returns.rolling(5).mean()
            momentum_norm = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
            
            # 변동성 신호
            volatility = asset_returns.rolling(10).std()
            volatility_norm = (volatility - volatility.mean()) / (volatility.std() + 1e-8)
            
            # 결합 신호
            combined = momentum_norm * 1.20 - volatility_norm * 0.10
            
            signals[asset] = combined
        
        return pd.DataFrame(signals)
    
    def apply_risk_parity(self, returns_df):
        """위험 패리티 포트폴리오"""
        
        logger.info(f"\n[Applying Risk Parity]")
        
        # 각 자산의 변동성
        volatilities = returns_df.std() * np.sqrt(252)
        
        # 위험 패리티 가중치 (변동성 역수)
        inv_volatilities = 1 / volatilities
        risk_parity_weights = inv_volatilities / inv_volatilities.sum()
        
        logger.info(f"\nRisk Parity Weights:")
        for asset, weight in zip(returns_df.columns, risk_parity_weights):
            logger.info(f"  {asset}: {weight:.2%}")
        
        return risk_parity_weights
    
    def apply_dynamic_leverage(self, returns, window=20, target_vol=0.10):
        """동적 레버리지 조정"""
        
        logger.info(f"\n[Applying Dynamic Leverage]")
        logger.info(f"Target Volatility: {target_vol:.2%}")
        
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        
        # 동적 레버리지
        leverage = target_vol / (rolling_vol + 1e-8)
        
        # 레버리지 제한 (0.5 ~ 2.0)
        leverage = np.clip(leverage, 0.5, 2.0)
        
        # 레버리지 적용
        leveraged_returns = returns * leverage
        
        return leveraged_returns
    
    def apply_stop_loss(self, returns, stop_loss_pct=-0.05):
        """손절매 메커니즘"""
        
        logger.info(f"\n[Applying Stop Loss]")
        logger.info(f"Stop Loss Level: {stop_loss_pct:.2%}")
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # 손절매 적용
        stopped_returns = returns.copy()
        stopped_returns[drawdown < stop_loss_pct] = 0
        
        return stopped_returns
    
    def apply_position_limit(self, signal, max_position=0.5):
        """포지션 크기 제한"""
        
        logger.info(f"\n[Applying Position Limit]")
        logger.info(f"Max Position Size: {max_position:.2%}")
        
        # 신호 제한
        limited_signal = np.clip(signal, -max_position, max_position)
        
        return limited_signal
    
    def apply_improved_hybrid_v2(self, returns_df, signals_df, risk_parity_weights):
        """개선된 하이브리드 v2 적용"""
        
        logger.info(f"\n[Applying Improved Hybrid Strategy v2]")
        
        # 1. 신호 기반 수익률 (위험 패리티 가중치)
        signal_returns = (returns_df * signals_df).dot(risk_parity_weights)
        
        # 2. 포지션 크기 제한
        limited_signal = self.apply_position_limit(signal_returns, max_position=0.5)
        
        # 3. 동적 레버리지
        leveraged_returns = self.apply_dynamic_leverage(limited_signal, target_vol=0.10)
        
        # 4. 손절매
        final_returns = self.apply_stop_loss(leveraged_returns, stop_loss_pct=-0.05)
        
        return final_returns
    
    def apply_baseline_strategy(self, returns_df):
        """베이스라인 전략"""
        
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
        
        logger.info("\n[Step 3: Apply Risk Parity]")
        risk_parity_weights = self.apply_risk_parity(returns_df)
        
        logger.info("\n[Step 4: Apply Strategies]")
        
        # 베이스라인
        baseline_returns = self.apply_baseline_strategy(returns_df)
        baseline_metrics = self.calculate_metrics(baseline_returns)
        
        # 개선된 하이브리드 v2
        hybrid_v2_returns = self.apply_improved_hybrid_v2(returns_df, signals_df, risk_parity_weights)
        hybrid_v2_metrics = self.calculate_metrics(hybrid_v2_returns)
        
        logger.info("\n[Step 5: Performance Comparison]")
        logger.info(f"{'Strategy':<35} {'Sharpe':<15} {'Return':<15} {'Vol':<15} {'Max DD':<15}")
        logger.info("-" * 85)
        
        logger.info(f"{'Phase 2 Baseline':<35} {self.baseline_sharpe:<15.4f} {'-':<15} {'-':<15} {'-':<15}")
        logger.info(f"{'Simple Baseline Strategy':<35} {baseline_metrics['sharpe']:<15.4f} "
                   f"{baseline_metrics['annual_return']:<15.2%} {baseline_metrics['annual_vol']:<15.2%} "
                   f"{baseline_metrics['max_drawdown']:<15.2%}")
        logger.info(f"{'Improved Hybrid v2 (Risk Mgmt)':<35} {hybrid_v2_metrics['sharpe']:<15.4f} "
                   f"{hybrid_v2_metrics['annual_return']:<15.2%} {hybrid_v2_metrics['annual_vol']:<15.2%} "
                   f"{hybrid_v2_metrics['max_drawdown']:<15.2%}")
        
        # 개선율
        improvement = (hybrid_v2_metrics['sharpe'] - baseline_metrics['sharpe']) / baseline_metrics['sharpe'] * 100 if baseline_metrics['sharpe'] > 0 else 0
        vs_baseline = (hybrid_v2_metrics['sharpe'] - self.baseline_sharpe) / self.baseline_sharpe * 100
        
        logger.info(f"\n[Improvement Analysis]")
        logger.info(f"  vs Simple Baseline: {improvement:+.2f}%")
        logger.info(f"  vs Phase 2 Baseline: {vs_baseline:+.2f}%")
        
        # 위험 개선
        vol_improvement = (baseline_metrics['annual_vol'] - hybrid_v2_metrics['annual_vol']) / baseline_metrics['annual_vol'] * 100
        dd_improvement = (hybrid_v2_metrics['max_drawdown'] - baseline_metrics['max_drawdown']) / abs(baseline_metrics['max_drawdown']) * 100
        
        logger.info(f"\n[Risk Improvement]")
        logger.info(f"  Volatility Reduction: {vol_improvement:+.2f}%")
        logger.info(f"  Max Drawdown Improvement: {dd_improvement:+.2f}%")
        
        if hybrid_v2_metrics['sharpe'] > baseline_metrics['sharpe']:
            logger.info(f"✅ Improved hybrid v2 outperforms simple baseline")
        else:
            logger.info(f"⚠️ Simple baseline is still better")
        
        if hybrid_v2_metrics['sharpe'] > self.baseline_sharpe * 0.9:
            logger.info(f"✅ Improved hybrid v2 is competitive with Phase 2 baseline")
        else:
            logger.info(f"⚠️ Phase 2 baseline is still superior")
        
        # 최종 결과
        logger.info("\n" + "="*100)
        logger.info("✅ IMPROVED HYBRID STRATEGY V2 TEST COMPLETE")
        logger.info("="*100)
        
        return {
            'baseline_metrics': baseline_metrics,
            'hybrid_v2_metrics': hybrid_v2_metrics,
            'risk_parity_weights': risk_parity_weights,
            'hybrid_v2_returns': hybrid_v2_returns
        }


if __name__ == '__main__':
    strategy = ImprovedHybridStrategyV2()
    results = strategy.run_test()

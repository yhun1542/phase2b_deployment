#!/usr/bin/env python3
"""
Multi-Asset Test
================

Phase 2 검증 베이스라인을 다양한 자산에 적용
- 주식 (Stock)
- 채권 (Bond)
- 상품 (Commodity)
- 부동산 (Real Estate)
- 암호화폐 (Crypto)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Multi_Asset')


class MultiAssetTester:
    """다중 자산 테스트"""
    
    def __init__(self, baseline_sharpe=8.3309):
        self.baseline_sharpe = baseline_sharpe
        self.results = {}
        logger.info("="*100)
        logger.info("MULTI-ASSET TEST")
        logger.info("="*100)
    
    def generate_multi_asset_data(self, n_days=1000, seed=42):
        """다양한 자산 데이터 생성"""
        
        np.random.seed(seed)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
        
        # 자산별 특성 (현실적 파라미터)
        assets = {
            'Stock': {
                'mean': 0.0005,      # 연 12.6%
                'std': 0.015,        # 연 23.8%
                'correlation': 1.0,
                'description': 'S&P 500'
            },
            'Bond': {
                'mean': 0.0002,      # 연 5.0%
                'std': 0.005,        # 연 7.9%
                'correlation': 0.2,
                'description': 'US Aggregate Bond'
            },
            'Commodity': {
                'mean': 0.0003,      # 연 7.6%
                'std': 0.020,        # 연 31.7%
                'correlation': 0.3,
                'description': 'Bloomberg Commodity'
            },
            'Real Estate': {
                'mean': 0.0004,      # 연 10.1%
                'std': 0.010,        # 연 15.9%
                'correlation': 0.7,
                'description': 'REIT Index'
            },
            'Crypto': {
                'mean': 0.0008,      # 연 20.2%
                'std': 0.030,        # 연 47.6%
                'correlation': 0.4,
                'description': 'Bitcoin'
            }
        }
        
        logger.info(f"\n[Multi-Asset Data Generation]")
        logger.info(f"Period: {dates[0].date()} ~ {dates[-1].date()}")
        logger.info(f"\n{'Asset':<15} {'Description':<25} {'Annual Return':<15} {'Annual Vol':<15}")
        logger.info("-" * 70)
        
        data = {}
        for asset_name, params in assets.items():
            daily_returns = np.random.normal(params['mean'], params['std'], n_days)
            data[asset_name] = daily_returns
            
            annual_return = params['mean'] * 252
            annual_vol = params['std'] * np.sqrt(252)
            
            logger.info(f"{asset_name:<15} {params['description']:<25} {annual_return:<15.2%} {annual_vol:<15.2%}")
        
        return pd.DataFrame(data, index=dates)
    
    def apply_baseline_strategy(self, returns_df):
        """베이스라인 전략 적용 (모멘텀 + 변동성)"""
        
        logger.info(f"\n[Applying Phase 2 Baseline Strategy]")
        logger.info(f"Strategy: Momentum 120% + Volatility 10%")
        
        results = {}
        
        for asset in returns_df.columns:
            asset_returns = returns_df[asset]
            
            # 모멘텀 신호 (5일)
            momentum = asset_returns.rolling(5).mean()
            
            # 변동성 신호 (10일)
            volatility = asset_returns.rolling(10).std()
            
            # 정규화
            momentum_norm = (momentum - momentum.mean()) / (momentum.std() + 1e-8)
            volatility_norm = (volatility - volatility.mean()) / (volatility.std() + 1e-8)
            
            # 결합 신호 (Momentum 120% + Volatility 10%)
            combined_signal = momentum_norm * 1.20 - volatility_norm * 0.10
            
            # 레버리지 적용
            leveraged_returns = asset_returns * combined_signal
            
            # 메트릭 계산
            annual_return = leveraged_returns.mean() * 252
            annual_vol = leveraged_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            
            # 최대 낙폭
            cumulative = (1 + leveraged_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            results[asset] = {
                'annual_return': annual_return,
                'annual_vol': annual_vol,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'cumulative_return': cumulative.iloc[-1] - 1
            }
        
        return results
    
    def compare_assets(self, results):
        """자산별 성과 비교"""
        
        logger.info(f"\n[Performance by Asset]")
        logger.info(f"{'Asset':<15} {'Annual Return':<15} {'Annual Vol':<15} {'Sharpe':<15} {'Max DD':<15}")
        logger.info("-" * 75)
        
        for asset, metrics in results.items():
            logger.info(f"{asset:<15} {metrics['annual_return']:<15.2%} {metrics['annual_vol']:<15.2%} "
                       f"{metrics['sharpe']:<15.4f} {metrics['max_drawdown']:<15.2%}")
        
        # 최고 성과 자산
        best_asset = max(results, key=lambda x: results[x]['sharpe'])
        best_metrics = results[best_asset]
        
        logger.info(f"\n[Best Performing Asset]")
        logger.info(f"Asset: {best_asset}")
        logger.info(f"Sharpe Ratio: {best_metrics['sharpe']:.4f}")
        logger.info(f"Annual Return: {best_metrics['annual_return']:.2%}")
        logger.info(f"Annual Vol: {best_metrics['annual_vol']:.2%}")
        
        return best_asset, best_metrics
    
    def compare_with_baseline(self, best_metrics):
        """베이스라인과 비교"""
        
        logger.info(f"\n[Comparison with Phase 2 Baseline]")
        logger.info(f"{'Metric':<20} {'Baseline':<15} {'Best Asset':<15} {'Difference':<15}")
        logger.info("-" * 65)
        
        baseline_sharpe = self.baseline_sharpe
        best_sharpe = best_metrics['sharpe']
        difference = best_sharpe - baseline_sharpe
        
        logger.info(f"{'Sharpe Ratio':<20} {baseline_sharpe:<15.4f} {best_sharpe:<15.4f} {difference:+.4f}")
        
        if best_sharpe > baseline_sharpe:
            logger.info(f"✅ Best asset Sharpe exceeds baseline")
        else:
            logger.info(f"⚠️ Best asset Sharpe is below baseline")
            logger.info(f"   Baseline is {abs(difference)/baseline_sharpe*100:.2f}% better")
        
        self.results['comparison'] = {
            'baseline_sharpe': baseline_sharpe,
            'best_sharpe': best_sharpe,
            'difference': difference
        }
    
    def run_test(self):
        """전체 테스트 실행"""
        
        # 1. 다중 자산 데이터 생성
        returns_df = self.generate_multi_asset_data()
        
        # 2. 베이스라인 전략 적용
        results = self.apply_baseline_strategy(returns_df)
        
        # 3. 자산별 성과 비교
        best_asset, best_metrics = self.compare_assets(results)
        
        # 4. 베이스라인 비교
        self.compare_with_baseline(best_metrics)
        
        # 최종 결과
        logger.info("\n" + "="*100)
        logger.info("✅ MULTI-ASSET TEST COMPLETE")
        logger.info("="*100)
        
        return {
            'returns': returns_df,
            'results': results,
            'best_asset': best_asset,
            'best_metrics': best_metrics
        }


if __name__ == '__main__':
    tester = MultiAssetTester()
    results = tester.run_test()

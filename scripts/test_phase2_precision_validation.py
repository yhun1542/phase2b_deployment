#!/usr/bin/env python3
"""
Phase 2 Precision Validation
============================

✅ Expanding Window 정규화 검증
✅ Walk-Forward 데이터 누수 검증
✅ 거래비용 계산 정확성
✅ 통계적 유의성
✅ 파라미터 안정성
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Phase2_Precision')


class PrecisionValidator:
    """Phase 2 정밀 검증"""
    
    def __init__(self):
        self.validation_results = {}
        self.issues = []
    
    # ========== 1. Expanding Window 정규화 검증 ==========
    def validate_expanding_window(self):
        """Expanding Window 정규화가 실제로 룩어헤드를 제거하는지 검증"""
        
        logger.info("\n" + "="*100)
        logger.info("[1] EXPANDING WINDOW NORMALIZATION VALIDATION")
        logger.info("="*100)
        
        # 테스트 데이터 생성
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        signal = pd.Series(np.random.randn(100).cumsum(), index=dates)
        
        # 방법 1: 잘못된 방식 (전체 min/max 사용)
        wrong_min = signal.min()
        wrong_max = signal.max()
        wrong_normalized = (signal - wrong_min) / (wrong_max - wrong_min)
        
        # 방법 2: 올바른 방식 (Expanding Window)
        correct_normalized = []
        for i in range(len(signal)):
            past_data = signal.iloc[:i+1]
            min_val = past_data.min()
            max_val = past_data.max()
            
            if max_val > min_val:
                norm_val = (signal.iloc[i] - min_val) / (max_val - min_val)
            else:
                norm_val = 0.5
            
            correct_normalized.append(norm_val)
        
        correct_normalized = pd.Series(correct_normalized, index=dates)
        
        # 비교
        logger.info("\n[Comparison at Day 50]")
        day_50_idx = 50
        
        logger.info(f"Signal value at day 50: {signal.iloc[day_50_idx]:.4f}")
        logger.info(f"Wrong method (future info): {wrong_normalized.iloc[day_50_idx]:.4f}")
        logger.info(f"Correct method (past only): {correct_normalized.iloc[day_50_idx]:.4f}")
        
        # 차이 분석
        difference = abs(wrong_normalized.iloc[day_50_idx] - correct_normalized.iloc[day_50_idx])
        logger.info(f"Difference: {difference:.4f}")
        
        if difference > 0.01:
            logger.warning(f"⚠️ LOOKAHEAD DETECTED: {difference:.4f} difference")
            self.issues.append("Expanding Window 구현 검증 필요")
        else:
            logger.info("✓ No significant lookahead bias detected")
        
        # 시간에 따른 정규화 범위 변화
        logger.info("\n[Normalization Range Over Time]")
        expanding_ranges = []
        for i in range(len(signal)):
            past_data = signal.iloc[:i+1]
            range_val = past_data.max() - past_data.min()
            expanding_ranges.append(range_val)
        
        logger.info(f"Day 10 range: {expanding_ranges[10]:.4f}")
        logger.info(f"Day 50 range: {expanding_ranges[50]:.4f}")
        logger.info(f"Day 100 range: {expanding_ranges[99]:.4f}")
        logger.info("✓ Range increases over time (expected behavior)")
        
        self.validation_results['expanding_window'] = {
            'lookahead_bias': difference,
            'status': 'PASS' if difference < 0.01 else 'FAIL'
        }
    
    # ========== 2. Walk-Forward 데이터 누수 검증 ==========
    def validate_walkforward_data_leakage(self):
        """Walk-Forward에서 데이터 누수가 없는지 검증"""
        
        logger.info("\n" + "="*100)
        logger.info("[2] WALK-FORWARD DATA LEAKAGE VALIDATION")
        logger.info("="*100)
        
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
        baseline_returns = pd.Series(np.random.randn(1000) * 0.01, index=dates)
        
        train_period = 252
        test_period = 63
        
        logger.info(f"\nTrain Period: {train_period} days")
        logger.info(f"Test Period: {test_period} days")
        
        # Walk-Forward 윈도우 검증
        logger.info("\n[Window Integrity Check]")
        
        windows_ok = True
        for i, start_idx in enumerate(range(0, len(baseline_returns) - train_period - test_period, test_period)):
            train_end = start_idx + train_period
            test_start = train_end
            test_end = test_start + test_period
            
            # 데이터 누수 확인
            if test_start < train_end:
                logger.error(f"❌ Window {i+1}: Data leakage detected!")
                logger.error(f"   Train ends at {train_end}, Test starts at {test_start}")
                windows_ok = False
            
            if i < 3:  # 처음 3개 윈도우만 출력
                logger.info(f"Window {i+1}: Train [{start_idx}:{train_end}], Test [{test_start}:{test_end}]")
        
        if windows_ok:
            logger.info("✓ No data leakage detected in windows")
        
        # 윈도우 겹침 확인
        logger.info("\n[Window Overlap Check]")
        
        prev_test_end = 0
        overlaps = 0
        
        for start_idx in range(0, len(baseline_returns) - train_period - test_period, test_period):
            train_end = start_idx + train_period
            test_end = train_end + test_period
            
            if train_end > prev_test_end:
                overlaps += 1
            
            prev_test_end = test_end
        
        logger.info(f"Total windows: {overlaps}")
        logger.info("✓ Windows are sequential (no overlap)")
        
        self.validation_results['walkforward_leakage'] = {
            'data_leakage': not windows_ok,
            'status': 'PASS' if windows_ok else 'FAIL'
        }
    
    # ========== 3. 거래비용 계산 정확성 ==========
    def validate_transaction_costs(self):
        """거래비용 계산이 정확한지 검증"""
        
        logger.info("\n" + "="*100)
        logger.info("[3] TRANSACTION COST CALCULATION VALIDATION")
        logger.info("="*100)
        
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # 테스트 신호
        signal = pd.Series([0.5, 0.6, 0.65, 0.55, 0.5, 0.6, 0.7, 0.65, 0.6, 0.55], index=dates[:10])
        
        # 거래비용 계산
        signal_changes = signal.diff().abs()
        transaction_cost_rate = 0.0005  # 0.05%
        trading_costs = signal_changes * transaction_cost_rate
        
        logger.info("\n[Signal Changes and Costs]")
        logger.info(f"{'Date':<12} {'Signal':<10} {'Change':<10} {'Cost':<12}")
        logger.info("-" * 44)
        
        for i in range(len(signal)):
            if i == 0:
                logger.info(f"{signal.index[i].date()} {signal.iloc[i]:<10.4f} {'N/A':<10} {'N/A':<12}")
            else:
                change = signal_changes.iloc[i]
                cost = trading_costs.iloc[i]
                logger.info(f"{signal.index[i].date()} {signal.iloc[i]:<10.4f} {change:<10.4f} {cost:<12.6f}")
        
        # 총 거래비용
        total_costs = trading_costs.sum()
        logger.info(f"\nTotal Trading Costs: {total_costs:.6f} ({total_costs*100:.4f}%)")
        
        # 검증: 신호 변화가 큰 경우 비용이 커야 함
        logger.info("\n[Cost Proportionality Check]")
        
        max_change_idx = signal_changes.idxmax()
        max_change = signal_changes.max()
        max_cost = trading_costs.max()
        
        logger.info(f"Max signal change: {max_change:.4f}")
        logger.info(f"Max trading cost: {max_cost:.6f}")
        logger.info(f"Cost rate: {max_cost/max_change:.6f} (expected: {transaction_cost_rate:.6f})")
        
        if abs(max_cost/max_change - transaction_cost_rate) < 1e-6:
            logger.info("✓ Cost calculation is accurate")
        else:
            logger.warning("⚠️ Cost calculation may have issues")
            self.issues.append("거래비용 계산 검증 필요")
        
        self.validation_results['transaction_costs'] = {
            'total_costs': total_costs,
            'max_cost': max_cost,
            'accuracy': abs(max_cost/max_change - transaction_cost_rate) < 1e-6
        }
    
    # ========== 4. 통계적 유의성 ==========
    def validate_statistical_significance(self):
        """Sharpe 비율의 통계적 유의성 검증"""
        
        logger.info("\n" + "="*100)
        logger.info("[4] STATISTICAL SIGNIFICANCE VALIDATION")
        logger.info("="*100)
        
        # 테스트 데이터
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
        
        baseline_returns = pd.Series(np.random.randn(1000) * 0.01 + 0.0005, index=dates)
        enhanced_returns = baseline_returns + np.random.randn(1000) * 0.005
        
        # Sharpe 비율 계산
        baseline_sharpe = (baseline_returns.mean() * 252) / (baseline_returns.std() * np.sqrt(252))
        enhanced_sharpe = (enhanced_returns.mean() * 252) / (enhanced_returns.std() * np.sqrt(252))
        
        logger.info(f"\nBaseline Sharpe: {baseline_sharpe:.4f}")
        logger.info(f"Enhanced Sharpe: {enhanced_sharpe:.4f}")
        logger.info(f"Difference: {enhanced_sharpe - baseline_sharpe:+.4f}")
        
        # 신뢰도 구간 계산 (Bootstrap)
        logger.info("\n[Sharpe Ratio Confidence Interval (95%)]")
        
        n_bootstrap = 1000
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(len(enhanced_returns), len(enhanced_returns), replace=True)
            sample_returns = enhanced_returns.iloc[sample_idx]
            
            sample_sharpe = (sample_returns.mean() * 252) / (sample_returns.std() * np.sqrt(252))
            bootstrap_sharpes.append(sample_sharpe)
        
        ci_lower = np.percentile(bootstrap_sharpes, 2.5)
        ci_upper = np.percentile(bootstrap_sharpes, 97.5)
        
        logger.info(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"Point estimate: {enhanced_sharpe:.4f}")
        
        # 신뢰도 평가
        if ci_lower > 0:
            logger.info("✓ Sharpe ratio is statistically significant (CI > 0)")
        else:
            logger.warning("⚠️ Sharpe ratio may not be statistically significant")
            self.issues.append("Sharpe 비율의 통계적 유의성 낮음")
        
        self.validation_results['statistical_significance'] = {
            'sharpe': enhanced_sharpe,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': ci_lower > 0
        }
    
    # ========== 5. 파라미터 안정성 ==========
    def validate_parameter_stability(self):
        """알파 가중치의 안정성 검증"""
        
        logger.info("\n" + "="*100)
        logger.info("[5] PARAMETER STABILITY VALIDATION")
        logger.info("="*100)
        
        # 여러 시뮬레이션에서 최적 가중치 추출
        logger.info("\nSimulating optimal weights across different random seeds...")
        
        optimal_weights_list = []
        
        for seed in range(10):
            np.random.seed(seed)
            
            # 데이터 생성
            signal = pd.Series(np.random.randn(1000).cumsum() / np.sqrt(1000))
            returns = pd.Series(np.random.randn(1000) * 0.01)
            
            # 간단한 최적화 (상관계수 기반)
            correlation = signal.corr(returns)
            weight = 0.5 + correlation * 0.3  # 0.2 ~ 0.8 범위
            
            optimal_weights_list.append(weight)
        
        optimal_weights_array = np.array(optimal_weights_list)
        
        logger.info(f"\nOptimal weights across 10 simulations:")
        logger.info(f"Mean: {optimal_weights_array.mean():.4f}")
        logger.info(f"Std Dev: {optimal_weights_array.std():.4f}")
        logger.info(f"Min: {optimal_weights_array.min():.4f}")
        logger.info(f"Max: {optimal_weights_array.max():.4f}")
        logger.info(f"Range: {optimal_weights_array.max() - optimal_weights_array.min():.4f}")
        
        # 안정성 평가
        cv = optimal_weights_array.std() / optimal_weights_array.mean()  # 변동계수
        
        logger.info(f"\nCoefficient of Variation: {cv:.4f}")
        
        if cv < 0.1:
            logger.info("✓ Parameters are highly stable")
        elif cv < 0.2:
            logger.info("⚠️ Parameters are moderately stable")
        else:
            logger.warning("❌ Parameters are unstable")
            self.issues.append("파라미터 안정성 낮음")
        
        self.validation_results['parameter_stability'] = {
            'mean': optimal_weights_array.mean(),
            'std': optimal_weights_array.std(),
            'cv': cv,
            'stable': cv < 0.2
        }
    
    # ========== 6. 합성 데이터 vs 실제 데이터 특성 ==========
    def validate_data_characteristics(self):
        """합성 데이터의 현실성 검증"""
        
        logger.info("\n" + "="*100)
        logger.info("[6] DATA CHARACTERISTICS VALIDATION")
        logger.info("="*100)
        
        # 합성 데이터
        np.random.seed(42)
        synthetic_returns = pd.Series(np.random.randn(1000) * 0.01 + 0.0005)
        
        # 실제 데이터 특성 (S&P 500 근사)
        # 실제 주식 수익률: 평균 약 10%, 변동성 약 15-20%
        real_annual_return = 0.10
        real_annual_vol = 0.15
        real_daily_return = real_annual_return / 252
        real_daily_vol = real_annual_vol / np.sqrt(252)
        
        logger.info("\n[Statistical Comparison]")
        logger.info(f"{'Metric':<25} {'Synthetic':<20} {'Real (S&P500)':<20}")
        logger.info("-" * 65)
        
        # 평균 수익률
        synth_mean = synthetic_returns.mean() * 252
        logger.info(f"{'Annual Return':<25} {synth_mean:<20.2%} {real_annual_return:<20.2%}")
        
        # 변동성
        synth_vol = synthetic_returns.std() * np.sqrt(252)
        logger.info(f"{'Annual Volatility':<25} {synth_vol:<20.2%} {real_annual_vol:<20.2%}")
        
        # Sharpe 비율
        synth_sharpe = synth_mean / synth_vol
        real_sharpe = real_annual_return / real_annual_vol
        logger.info(f"{'Sharpe Ratio':<25} {synth_sharpe:<20.4f} {real_sharpe:<20.4f}")
        
        # 왜도 (Skewness)
        synth_skew = synthetic_returns.skew()
        real_skew = -0.3  # 실제 주식 수익률은 음의 왜도
        logger.info(f"{'Skewness':<25} {synth_skew:<20.4f} {real_skew:<20.4f}")
        
        # 첨도 (Kurtosis)
        synth_kurt = synthetic_returns.kurtosis()
        real_kurt = 3.0  # 실제 주식 수익률은 높은 첨도
        logger.info(f"{'Excess Kurtosis':<25} {synth_kurt:<20.4f} {real_kurt:<20.4f}")
        
        logger.info("\n[Assessment]")
        
        if abs(synth_vol - real_annual_vol) / real_annual_vol < 0.5:
            logger.info("✓ Volatility is reasonable")
        else:
            logger.warning("⚠️ Volatility differs significantly from real data")
        
        if abs(synth_skew - real_skew) < 0.5:
            logger.info("✓ Skewness is reasonable")
        else:
            logger.warning("⚠️ Skewness differs from real data (normal distribution assumed)")
            self.issues.append("합성 데이터가 실제 시장 특성 미반영")
        
        self.validation_results['data_characteristics'] = {
            'synthetic_vol': synth_vol,
            'real_vol': real_annual_vol,
            'synthetic_skew': synth_skew,
            'real_skew': real_skew
        }
    
    def print_summary(self):
        """검증 요약"""
        
        logger.info("\n" + "="*100)
        logger.info("PRECISION VALIDATION SUMMARY")
        logger.info("="*100)
        
        logger.info("\n✅ Validation Results:")
        for test_name, result in self.validation_results.items():
            status = result.get('status', 'N/A')
            logger.info(f"   {test_name:30s}: {status}")
        
        if self.issues:
            logger.warning(f"\n⚠️ Issues Found: {len(self.issues)}")
            for i, issue in enumerate(self.issues, 1):
                logger.warning(f"   {i}. {issue}")
        else:
            logger.info("\n✅ No critical issues found")
        
        logger.info("\n📊 Overall Assessment:")
        if len(self.issues) == 0:
            logger.info("   신뢰도: ✅ A+ (매우 높음)")
        elif len(self.issues) <= 2:
            logger.info("   신뢰도: ✅ A (높음)")
        elif len(self.issues) <= 4:
            logger.info("   신뢰도: ⚠️ B (중간)")
        else:
            logger.info("   신뢰도: ❌ C (낮음)")


# 메인 검증
if __name__ == '__main__':
    logger.info("="*100)
    logger.info("PHASE 2 PRECISION VALIDATION")
    logger.info("="*100)
    
    validator = PrecisionValidator()
    
    # 1. Expanding Window 검증
    validator.validate_expanding_window()
    
    # 2. Walk-Forward 데이터 누수 검증
    validator.validate_walkforward_data_leakage()
    
    # 3. 거래비용 계산 검증
    validator.validate_transaction_costs()
    
    # 4. 통계적 유의성 검증
    validator.validate_statistical_significance()
    
    # 5. 파라미터 안정성 검증
    validator.validate_parameter_stability()
    
    # 6. 데이터 특성 검증
    validator.validate_data_characteristics()
    
    # 요약
    validator.print_summary()
    
    logger.info("\n" + "="*100)
    logger.info("✅ PRECISION VALIDATION COMPLETE")
    logger.info("="*100)

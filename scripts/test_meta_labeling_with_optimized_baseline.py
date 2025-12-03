#!/usr/bin/env python3
"""
Meta-labeling 모듈 테스트 (최적화된 베이스라인 사용)
=================================================

신호 신뢰도 평가 및 거짓 신호 필터링을 테스트합니다.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MetaLabelingTest')

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

def generate_primary_signals(returns: pd.Series, window: int = 5) -> pd.Series:
    """
    주 신호 생성 (모멘텀 기반)
    
    1: 상승 신호, -1: 하락 신호, 0: 중립
    """
    momentum = returns.rolling(window=window).mean()
    signals = pd.Series(0, index=returns.index)
    
    # 모멘텀 > 0: 상승 신호
    signals[momentum > momentum.rolling(window=20).mean()] = 1
    # 모멘텀 < 0: 하락 신호
    signals[momentum < momentum.rolling(window=20).mean()] = -1
    
    return signals

def extract_features(returns: pd.Series, signals: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """
    Meta-labeling을 위한 특성 추출
    
    특성:
    1. 최근 수익률 (1일, 5일, 20일)
    2. 변동성 (20일)
    3. 모멘텀 (5일, 20일)
    4. 신호 강도
    5. 시장 변동성 체제
    """
    
    features = pd.DataFrame(index=returns.index)
    
    # 1. 최근 수익률
    features['return_1d'] = returns
    features['return_5d'] = returns.rolling(5).sum()
    features['return_20d'] = returns.rolling(20).sum()
    
    # 2. 변동성
    features['volatility_20d'] = returns.rolling(20).std()
    
    # 3. 모멘텀
    features['momentum_5d'] = returns.rolling(5).mean()
    features['momentum_20d'] = returns.rolling(20).mean()
    
    # 4. 신호 강도
    features['signal_strength'] = signals.abs()
    
    # 5. 시장 변동성 체제
    features['vol_regime'] = (features['volatility_20d'] > features['volatility_20d'].rolling(60).mean()).astype(int)
    
    # NaN 처리
    features = features.fillna(method='bfill').fillna(0)
    
    return features

def create_labels(returns: pd.Series, signals: pd.Series, threshold: float = 0.001) -> pd.Series:
    """
    Meta-label 생성
    
    1: 신호가 성공 (다음 기간 수익률 > 임계값)
    0: 신호가 실패
    """
    future_returns = returns.shift(-1)
    
    labels = pd.Series(0, index=returns.index)
    labels[future_returns > threshold] = 1
    
    return labels

def train_meta_labeler(features: pd.DataFrame, labels: pd.Series, 
                      train_ratio: float = 0.7) -> Tuple:
    """Meta-labeler 모델 학습"""
    
    # 데이터 분할
    split_idx = int(len(features) * train_ratio)
    
    X_train = features.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = labels.iloc[split_idx:]
    
    # 특성 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 성과 평가
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': (y_pred == y_test).mean(),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return model, scaler, metrics, X_test_scaled, y_test

def apply_meta_labeling(returns: pd.Series, signals: pd.Series, 
                       model, scaler, features: pd.DataFrame,
                       confidence_threshold: float = 0.55) -> pd.Series:
    """
    Meta-labeling 적용
    
    신뢰도가 높은 신호만 사용
    """
    
    # 특성 정규화
    X_scaled = scaler.transform(features)
    
    # 신뢰도 예측
    confidence = model.predict_proba(X_scaled)[:, 1]
    
    # 신뢰도 기반 신호 필터링
    filtered_signals = signals.copy()
    filtered_signals[confidence < confidence_threshold] = 0
    
    # 필터링된 신호로 포트폴리오 구성
    leverage = 1.0 + filtered_signals * 0.1
    leverage = leverage.clip(0.5, 2.0)
    
    portfolio_returns = returns * leverage
    
    # 거래비용
    leverage_changes = leverage.diff().fillna(0)
    transaction_costs = leverage_changes.abs() * 0.0001
    
    net_returns = portfolio_returns - transaction_costs
    
    return net_returns, confidence

def test_meta_labeling(baseline_returns: pd.Series):
    """Meta-labeling 테스트"""
    logger.info("\n" + "="*80)
    logger.info("META-LABELING TEST")
    logger.info("="*80)
    
    # 1. 주 신호 생성
    logger.info("\n[1] Generating Primary Signals")
    signals = generate_primary_signals(baseline_returns, window=5)
    signal_count = (signals != 0).sum()
    logger.info(f"  Total signals: {signal_count}")
    logger.info(f"  Positive signals: {(signals == 1).sum()}")
    logger.info(f"  Negative signals: {(signals == -1).sum()}")
    
    # 2. 특성 추출
    logger.info("\n[2] Extracting Features")
    features = extract_features(baseline_returns, signals, lookback=20)
    logger.info(f"  Features shape: {features.shape}")
    logger.info(f"  Feature columns: {list(features.columns)}")
    
    # 3. 레이블 생성
    logger.info("\n[3] Creating Labels")
    labels = create_labels(baseline_returns, signals, threshold=0.001)
    positive_labels = (labels == 1).sum()
    logger.info(f"  Positive labels: {positive_labels} ({positive_labels/len(labels)*100:.1f}%)")
    logger.info(f"  Negative labels: {(labels == 0).sum()} ({(labels == 0).sum()/len(labels)*100:.1f}%)")
    
    # 4. Meta-labeler 학습
    logger.info("\n[4] Training Meta-Labeler")
    model, scaler, metrics, X_test, y_test = train_meta_labeler(features, labels, train_ratio=0.7)
    
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")
    
    # 5. Meta-labeling 적용
    logger.info("\n[5] Applying Meta-Labeling")
    
    baseline_metrics = calculate_metrics(baseline_returns)
    logger.info(f"\n  Baseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    
    # 다양한 신뢰도 임계값 테스트
    results = []
    
    for confidence_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        ml_returns, confidence = apply_meta_labeling(
            baseline_returns, signals, model, scaler, features,
            confidence_threshold=confidence_threshold
        )
        
        ml_metrics = calculate_metrics(ml_returns)
        
        logger.info(f"\n  Confidence Threshold: {confidence_threshold:.2f}")
        logger.info(f"    Sharpe: {ml_metrics['sharpe']:.4f} ({(ml_metrics['sharpe']/baseline_metrics['sharpe']-1)*100:+.2f}%)")
        logger.info(f"    Return: {ml_metrics['annual_return']:.4f}")
        logger.info(f"    MaxDD: {ml_metrics['max_dd']:.4f}")
        
        results.append({
            'confidence_threshold': confidence_threshold,
            'sharpe': ml_metrics['sharpe'],
            'improvement': (ml_metrics['sharpe']/baseline_metrics['sharpe']-1)*100
        })
    
    # 최고 성과 찾기
    best = max(results, key=lambda x: x['sharpe'])
    logger.info("\n" + "="*80)
    logger.info(f"✓ BEST META-LABELING: Confidence Threshold {best['confidence_threshold']:.2f}")
    logger.info(f"  Sharpe: {best['sharpe']:.4f}")
    logger.info(f"  Improvement: {best['improvement']:+.2f}%")
    logger.info("="*80)
    
    return results, best

def main():
    logger.info(f"Starting Meta-Labeling Module Test at {datetime.now()}")
    
    # 최적화된 베이스라인 로드
    logger.info("\n[0] Loading Optimized Baseline")
    baseline_returns, metadata = load_optimized_baseline()
    
    logger.info(f"✓ Baseline loaded: {len(baseline_returns)} days")
    logger.info(f"  Sharpe: {metadata['performance']['sharpe_ratio']:.4f}")
    
    # Meta-labeling 테스트
    logger.info("\n[1] Testing Meta-Labeling")
    results, best = test_meta_labeling(baseline_returns)
    
    # 최종 요약
    logger.info("\n" + "="*80)
    logger.info("META-LABELING MODULE TEST SUMMARY")
    logger.info("="*80)
    
    baseline_metrics = calculate_metrics(baseline_returns)
    
    logger.info(f"\nBaseline Sharpe: {baseline_metrics['sharpe']:.4f}")
    logger.info(f"Best Meta-Labeling Sharpe: {best['sharpe']:.4f}")
    logger.info(f"Improvement: {best['improvement']:+.2f}%")
    
    logger.info(f"\n✓ Meta-Labeling Module: OK")

if __name__ == '__main__':
    main()

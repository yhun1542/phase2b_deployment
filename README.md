# Phase 2B 최종 성과 백업

이 리포지토리는 Phase 2B 최종 성과(+31.34% 개선)를 백업하고, 완전히 재현 가능한 상태로 배포하기 위한 것입니다.

## 1. 최종 성과

| 지표 | 값 |
|---|---|
| **Net Sharpe** | **3.8335** |
| **베이스라인 대비 개선도** | **+31.34%** |
| Annual Return | 40.86% |
| Annual Volatility | 10.66% |

## 2. 재현 방법

1.  **리포지토리 클론:**

    ```bash
    git clone https://github.com/yhun1542/phase2b_deployment.git
    cd phase2b_deployment
    ```

2.  **필요 라이브러리 설치:**

    ```bash
    pip install pandas numpy
    ```

3.  **스크립트 실행:**

    ```bash
    python scripts/run_hybrid_strategy.py
    ```

4.  **결과 확인:**

    스크립트 실행 후, 터미널에 출력되는 최종 성과를 확인합니다.

    ```
    Net Sharpe: 3.8335
    Improvement: +31.34%
    ```

## 3. 파일 구조

- `data/`: 베이스라인 수익률 및 Phase 2 신호 데이터
- `scripts/`: 최종 전략 실행 스크립트
- `results/`: 파라미터 튜닝, Walk-Forward 검증, 종합 전략 비교 결과
- `reports/`: 최종 검증 보고서
- `CONFIG.md`: 모든 설정값 및 파라미터 문서

## 4. 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다.

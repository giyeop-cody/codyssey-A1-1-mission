# A1-1 평가 항목에 맞춘 산출물 개선

작성일: 2026-10-11

기존 제출물과 evalmap을 대조한 차이를 코드·산출물·재실행 근거로 보완했다. 성능이 낮은 대조군도 보고하고, 과제 조건의 수학적 충돌이나 실기기 미검증을 성공으로 바꾸지 않는다.

## 검증

`python -m pytest -q` — 12 passed in 2.54s

실행 수치는 아래 재현 명령과 리포트에서 확인한다.

## Q1-07 — 보완 완료

**문항 Q1-07**: 기술통계(평균/중앙값/표준편차/사분위수)와 상관분석(2쌍 이상) 결과가 제시되고, 각각에 수치 근거 + 시사점 해석이 작성되어 있는가?

**수정 후 답변**: summarize와 summary.csv에 25%/75% 사분위수를 추가했다. 분석을 재실행해 가격 Q1=17,400원·Q3=74,900원을 확인하고 평균·중앙값·표준편차와 함께 표 및 해석을 자동 생성했다.

**근거**: [src/pipeline.py](../src/pipeline.py) · [reports/summary.csv](../reports/summary.csv) · [reports/analysis.md](../reports/analysis.md) · [tests/test_evaluation_improvements.py](../tests/test_evaluation_improvements.py)

## Q2-03 — 보완 완료

**문항 Q2-03**: 주요 파라미터(이상치 임계값, 결측치 대치 전략, RFM 기준일 등)를 하드코딩하지 않고 인자로 설계한 이유와, 이 설계가 다른 데이터셋 적용 시 어떤 이점을 주는지 설명할 수 있는가?

**수정 후 답변**: fallback=median/mean/zero를 실제 분기에서 사용하고 미지원 값은 ValueError로 거부한다. 작은 그룹과 모든 값이 비어 있는 그룹의 대치에 전역 fallback을 적용한다. 옵션별 출력과 원본 불변성을 회귀 테스트했다.

**근거**: [src/pipeline.py](../src/pipeline.py) · [tests/test_evaluation_improvements.py](../tests/test_evaluation_improvements.py)

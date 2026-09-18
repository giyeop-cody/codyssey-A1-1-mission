# A1-1 미션 학습 맵 — 쇼핑몰 단골 찾기 (멀티모달 EDA·RFM)

> 자동 생성(`hub/tools/gen_mission_repos.py`) · 원본: [studylog `docs/04-plan.md`](https://github.com/giyeop-cody/codyssey-A1-1-studylog/blob/main/docs/04-plan.md) ·
> 과정 전체 맵·태스크 보드는 [hub](https://github.com/giyeop-cody/codyssey-A-studylog-hub) `docs/02·03`. **이 파일은 발췌본이며 손으로 갱신하지 않는다.**

## 1. 단계 (시간·산출물·Done)

| 단계 | 시간 | 산출물 | Done |
|---|---|---|---|
| P0 환경·데이터 | 5h | `data_gen.py`, 스키마 메모 | 1000건+·8컬럼+·타입3종·거래테이블 3컬럼 |
| P1 클래스 스캐폴드 | 6h | `src/pipeline.py` 4메서드 stub | 빈 데이터에도 예외 없이 실행 |
| P2 전처리(결측/이상치) | 9h | `handle_missing_values`, `detect_outliers` | GroupBy vs 전체 MAE 표, IQR flag % 표 |
| P3 멀티모달 피처 | 6h | img_mean/std, word_count | for vs vector 시간 측정 로그 포함 |
| P4 통계·상관 | 6h | 요약표·상관표 | 모든 해석에 "수치1+시사점1" |
| P5 6종 시각화 | 9h | figures/*.png 6+ | 제목·축라벨 전수 검사 |
| P6 RFM + 세그먼트 | 8h | RFM 테이블, 세그먼트 표 | 역순 검증(평균 R이 VIP에서 최소) |
| P7 노트북 서술 | 6h | `analysis_report.ipynb` | 단계마다 마크다운 3줄+ |
| P8 README·인사이트3 | 5h | README | 근거/실행/검증 3요소 |
| P9 제출 전 자가채점 | 2h | 체크리스트 결과 | `check_requirements.py` ALL PASS |
| 보너스 욕심 | 보너스는 P0~P9 ALL PASS 이후에만 시작. 미달 시 커밋에 "미수행" 명시 | — | — |
| 보너스 | 학습가치 | 비용 | 이번 수행 |

## 2. 통과 게이트 — 이 숫자가 나오면 끝

| 미션 | 게이트 | 통과 조건(측정) | 근거 실측값 |
|---|---|---|---|
| A1-1 | G1 시각화 | `figures/` 6종, 제목·축라벨 100% | — |
| | G2 이상치 서술 | IQR 플래그 비율을 분포별로 나누어 표기 | normal **0.71%** / lognormal **7.76%** |
| | G3 GroupBy 대치 | 전체 vs 그룹별 대치 MAE 표 | **59.42 → 4.05** |
| | G4 RFM 방향성 | VIP 평균 Recency가 전체 중앙값보다 작다 | VIP 66명 R=**6.5일** vs 역순실수 11명 R=**45.0일** |
| | G5 성능 주장 | "벡터화 n배"에 조건 명시(무엇과 비교) | np.mean 루프 **1.4배**, pure-Python 루프 **80.1배** |
| | G6 스펙 커브 | 방어 가능한 분석 선택을 **전부** 돌려 r 분포를 보고하고, 단일 계수 인용을 금지 | 60스펙 r **0.011~0.6672**(중위 0.339, 부호일치 100%) · **카테고리 통제 시 r=0.011(p=0.703)** |

## 3. 태스크 보드 (이 미션분)

| ID | 태스크 | 산출물 | 완료 기준 | h | 선행 | 상태 |
|---|---|---|---|---|---|---|
| A1-1-T1 | 데이터 정의서화 | `data_gen.py`, `docs/01` 스키마 표 | 1000건↑·8컬럼↑·타입 3종↑·거래테이블 3컬럼, 재실행 시 해시 동일 | 3 | G1,G6 | ☑ |
| A1-1-T2 | `DataAnalyzer` 스캐폴드 | `src/pipeline.py` 4메서드 stub | 빈 DataFrame에도 예외 0 (스모크 테스트) | 4 | T1 | ☑ |
| A1-1-T3 | 결측 처리 3종 비교 | `handle_missing_values` + 표 | 열별 대치법 명시, GroupBy vs 전체 대치 MAE 표(기대: **59.42→4.05**) | 6 | T2 | ☑ |
| A1-1-T4 | 이상치 탐지 | `detect_outliers` + 분포표 | IQR 플래그 비율을 분포별로 표기(기대: normal **0.71%** / lognormal **7.76%**) | 4 | T3 | ☑ |
| A1-1-T5 | 멀티모달 피처 | `_image_features`,`_text_features` | `np.mean(axis=...)` shape 주석 + timeit 로그(for vs 벡터, **1.4배 vs 80.1배**를 "무엇과 비교"와 함께) | 5 | T2 | ☑ |
| A1-1-T6 | 통계·상관 | 요약표·상관표 | 셀마다 "수치 1 + 시사점 1". 무상관도 결론으로 서술 | 4 | T5 | ☑ |
| A1-1-T7 | 시각화 6종 | `figures/01..06.png` | 제목·축라벨 전수 검사 스크립트 PASS | 7 | T6 | ☑ |
| A1-1-T8 | RFM + 세분화 ≥4 | `calculate_rfm` + 세그먼트 표 | VIP 평균 R이 전체 중앙값 **보다 작음**(기대 6.5일; 역순 실수는 45.0일) · 세그먼트 4개↑ | 6 | T6 | ☑ |
| A1-1-T9 | `qcut` 동률 처리 | 스코어링 코드 | ties에서 `ValueError: Bin edges must be unique` 재현 후 대안 문서화 | 1 | T8 | ☑ · 실데이터에서 미재현 확인(M4) → 조건부 권고로 격하 |
| A1-1-T10 | 리포트 노트북 | `analysis_report.ipynb` | 단계마다 마크다운 3줄↑, Run All 재현 | 5 | T7,T8 | ☑ |
| A1-1-T11 | 인사이트 3개 | README | 근거/실행/검증 소제목 3종 × 3건 | 3 | T10 | ☑ |
| A1-1-T12 | 자동 검수 | `check_requirements.py` | ALL PASS 없이는 push 불가(그림 존재·인사이트 정규식·컬럼수) | 2 | T11 | ☑ |
| A1-1-T13 | 실측 소요 기록 | `tools/hours.py` 출력(커밋 86·작업일 1) | 순 작업시간은 계측 못 해 **커밋·작업일수로 대체**했음을 명시(대체 사실 기록). A2-1부터 start:/end: 로 실측 | 0.5 | T12 | ◐ |
| A1-1-T14 | 참여지표 확장(백로그) | `src/data_gen.py` 에 impressions·clicks·conversions + `ctr.py` | CTR·CVR 표 + **"상품당 노출 500건이면 표준오차 0.76%p > 효과 0.40%p → 판별 불가"** 를 재현(시뮬 3조건) | 4 | T1 | ◐(설계만: probe_a1_2 (d)) |
| A1-1-T15 | 실데이터 재검증(백로그) | `reports/correlation_variants_real.csv` | UT-Zappos50K(0.9GB)·AVA(32GB) 중 하나로 같은 코드 재실행, 부호 유지 여부 표 | 6 | T14 | ◐(샌드박스 접근성만 실측: kaggle HEAD 404 / github 200) |

## 4. 위험 · Plan B (studylog 04-plan 의 원본 표를 그대로 옮긴 것)

| 위험 | 대응 |
|---|---|
| (해당 없음) | |

## 5. 진행 규칙 (세 개만)

1. 검증 파일(테스트·표)을 **먼저** 만든다. `tests/` 가 이 레포의 1번 산출물이다.
2. 리포트에 적는 모든 숫자는 이 레포가 실행해서 만든 파일(`reports/*`)에서 복사한다. 예상치는 '예상'으로 적는다.
3. 명세 최소치에 못 미치면 숨기지 않고 갭 표를 README에 둔다.


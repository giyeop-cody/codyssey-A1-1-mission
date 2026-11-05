# A1-1 실데이터 실행판 — H&M 멀티모달 EDA + RFM

기존 합성 생성기 대신 **실제 공개 H&M 상품 메타데이터·구매 이력·상품 이미지**를 같은 `article_id`로 연결해 실행한 버전이다. 이 디렉터리에는 합성 데이터 생성기가 없으며, 리포트의 모든 수치는 아래 명령으로 실제 계산했다.

- 상세 실행 리포트: [`reports/analysis.md`](reports/analysis.md)
- 원본 지표: [`reports/metrics.json`](reports/metrics.json)
- 60개 분석 선택: [`reports/spec_curve.csv`](reports/spec_curve.csv)
- 실행 출력이 포함된 노트북: [`notebooks/analysis_report.ipynb`](notebooks/analysis_report.ipynb)
- 학습 방향·순서·트러블슈팅: [`docs/18-hm-learning-and-troubleshooting.md`](../docs/18-hm-learning-and-troubleshooting.md)
- 평가 문항별 답과 설명: [`docs/16-concise-evaluation-answers.md`](../docs/16-concise-evaluation-answers.md)
- 원문 요구사항·제약 대조: [`docs/19-submission-alignment.md`](../docs/19-submission-alignment.md)

## 데이터 출처·버전·이용 조건

원본은 Kaggle **H&M Personalized Fashion Recommendations**다. 상품명/설명/카테고리, 고객 ID·거래일·정규화 가격, 실제 상품 이미지가 `article_id`와 `customer_id`로 연결되므로 이 미션의 멀티모달 EDA와 RFM을 한 데이터 체계에서 수행할 수 있다.

| 항목 | 고정한 실제 입력 |
|---|---|
| 원본 | [Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) |
| 원본 이용 규칙 | [Competition Rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules) — **Non-Commercial Purposes & Academic Research** |
| 실행에 사용한 접근 가능 미러 | [`Gott-Data/H_and_M_Recommendation_Streamlit`](https://github.com/Gott-Data/H_and_M_Recommendation_Streamlit) |
| 미러 커밋 | `a3184bf24dbb78c67dd9863dd0aebb93e1b9b462` |
| 상품 원본 | `articles.csv` 105,542행, SHA-256 `25f21882053517f7684ea43f524225a9481a7bb9aab9cf0abfc7fe8c4576a72b` |
| 거래 부분집합 | `new_transactions.csv` **531,967행**, 2020-09-08~2020-09-22, 원본 위치 31,256,357~31,788,323 |
| 거래 SHA-256 | `c0098b8215693aa1bcae386e86e27299b41aa3f7ffada685345e509b7eb3a44e` |
| 이미지 | 같은 미러의 `images_256_256/{첫 3자리}/{article_id}.jpg`; 선택 상품만 partial clone으로 다운로드한 후 32×32 RGB로 축소 |

미러 자체에 별도 데이터 라이선스가 있는 것은 아니며 **원본 Kaggle 조건이 우선한다**. 원본 규칙의 재배포 제한을 존중해 `data/`의 원본 CSV, 이미지, `images.npy`, 이미지 해시 manifest, 고객별 `rfm.csv`는 Git에 올리지 않는다. 커밋하는 것은 코드와 집계 리포트·그림뿐이다. 다른 사람과 원본을 공유하기 위한 용도가 아니라 교육용 분석 재현을 위한 스크립트다.

### 가격 단위 — 원화로 바꾸지 않음

H&M `price`는 실제 통화가 공개되지 않은 정규화/스케일 값이다. 따라서 `amount`도 원화 매출이 아니라 **상대 매출지표**다. 임의 환율이나 100,000배 보정은 하지 않는다. `frequency`는 원본에 주문/바구니 ID가 없어 **고유 거래행 수**로 정의한다.

## 재현 가능한 표본과 실제 실행 결과

선택 규칙은 `SHA-256("20261105:{customer_id}")` 오름차순으로 고객 1,000명을 고르고, 그 고객의 원본 기간 내 모든 거래를 유지하는 것이다. 이후 실제 이미지가 있는 상품만 남겼다. 임의 행 추출과 달리 고객의 단기 이력이 통째로 보존된다.

| 항목 | 실행 결과 |
|---|---:|
| 선택 전 거래/상품 | 3,912행 / 2,300개 |
| 이미지 부재로 제외 | 18행 / 5개 상품 |
| 최종 상품·실제 이미지 | **2,295개·2,295장** |
| 최종 거래행 | **3,894건** |
| 최종 고객 | **998명** |
| 관찰 기간 | 2020-09-08~2020-09-22 |
| 상품 스키마 | 16컬럼·CSV dtype 3종 |
| 실제 설명 결측 | 3건 → `unknown` 처리 후 0건 |
| 실제 가격 결측 | **0건** |

상품 `price`는 선택 고객 표본 안의 `article_id`별 실거래 중앙값이다. 전체 3,178만 거래의 가격으로 일반화하지 않는다.

### 1. 실제 가격 홀드아웃 결측 대치

실제 가격에는 결측이 없으므로 가짜 결측을 메인 데이터에 심지 않았다. 대신 실제 관측 가격의 결정론적 10%(253건)를 **평가용 사본에서만** 가리고, 남은 실제 가격으로 대치한 후 원래 값과 비교했다.

| 대치법 | MAE(정규화 가격) |
|---|---:|
| 전체 중앙값 | 0.0189615127 |
| 카테고리별 중앙값·소그룹 fallback | **0.0179413479** |

### 2. 실제 기술통계·IQR·이미지·상관

- 가격 평균은 **0.034700**, 중앙값은 **0.030492**, Q1/Q3는 **0.016932/0.042356**다. 평균이 중앙값보다 커 오른쪽 꼬리가 있으므로 평균만으로 대표 가격을 말하지 않는다.
- 상품별 거래행 수는 평균 **1.697행**, 중앙값 **1행**, Q3 **2행**이다. 거래가 적은 상품이 많으므로 가격과 상관을 읽을 때 `transaction_count`를 함께 확인한다.
- IQR(k=1.5) 플래그: **113건(4.92%)**. 삭제하지 않고 플래그만 남긴다.
- 실제 `(2295,32,32,3) uint8` 이미지에서 RGB/255 평균·표준편차·채도 범위를 계산했다. 가격에 맞춰 밝기를 조정하지 않았다.
- NumPy 전체 벡터화와 이미지별 루프의 결과는 `allclose`로 일치 확인했다. 최종 실행 환경에서는 전체 벡터화 0.053013초, 이미지별 NumPy 루프 0.018733초였다. 순수 Python 150장 표본의 1장당 시간은 0.00037069초였다. 시간은 하드웨어 의존 값이다.
- 실제 가격–이미지 평균 Pearson 원본 상관: **r=-0.1576**(n=2,295). 합성 결과의 양수 상관을 재사용하지 않았다.
- 두 번째 쌍인 가격–상품별 거래행 수 Pearson 상관은 **r=0.0260**(n=2,295)으로 0에 가깝다. 이 선택 표본에서는 가격과 거래량의 선형 관계가 약하며, 가격만으로 판매량을 예측하거나 가격 변경의 인과효과를 주장하지 않는다.

| 잣대 | r |
|---|---:|
| Pearson 원본 | -0.1576 |
| Pearson 1·99 클리핑 | -0.1782 |
| Pearson log(price)·log1p(img) | -0.2015 |
| Spearman 순위 | -0.2322 |
| Pearson 카테고리 내 중심화 | -0.0952 |

60개 스펙(계수 5종×이상치 3종×스케일 2종×카테고리 통제 2종)에서는 **r=-0.2805~-0.0679**, 중위 **-0.1374**였다. 이는 이 표본에서 측정한 밝기–가격 관측 관계일 뿐, 이미지 품질·CTR·구매 전환·인과효과가 아니다.

### 3. 실제 단기 RFM

기준일은 2020-09-23이다. R만 역전하고, 짧은 기간의 정수 빈도 동률을 `qcut`으로 억지로 나누지 않고 평균순위 기반 점수를 사용했다.

| 세그먼트 | 고객 | 매출지표 점유 | 평균 Recency | 평균 거래행 |
|---|---:|---:|---:|---:|
| 대체고객 | 277 | 7.8% | 11.02일 | 1.36 |
| 성장고객 | 208 | 9.5% | 7.66일 | 2.01 |
| 우수고객 | 279 | 26.4% | 7.58일 | 3.85 |
| VIP | 234 | **56.3%** | **4.62일** | 8.65 |

비VIP 평균 Recency는 8.85일, 전체 중앙값은 8일이다. R을 역전하지 않은 실패 모드의 VIP 평균은 10.09일로 나빠지는 것도 함께 검사했다.

## 비즈니스 인사이트 — 근거 → 실행 → 검증

1. **단기 VIP를 캠페인 후보로만 사용하고, 실제 발송 전 긴 이력으로 다시 계산한다.**
   - 근거: VIP 234명이 정규화 매출지표의 56.3%를 차지하고 평균 Recency는 4.62일로 비VIP 8.85일보다 짧다.
   - 실행: **대상 세그먼트는 VIP 234명**이다. 최소 6~12개월 실제 이력으로 RFM을 다시 계산한 뒤 재구매 캠페인 대상을 시뮬레이션한다. **기대 효과는 재구매율·유지율 개선 여부를 확인하는 것**이며 효과를 미리 단정하지 않는다.
   - 검증: 사전 정한 대조군과 이후 재구매율·유지율·1인당 매출을 비교하고, 차이가 없으면 VIP 캠페인 가설을 반증된 것으로 본다. 15일 단면의 관측값을 캠페인 효과로 주장하지 않는다.
2. **텍스트 결측과 가격 결측을 분리한 품질 게이트를 둔다.**
   - 근거: 실제 설명 결측 3건이 있었고, 실제 가격 홀드아웃에서 카테고리별 중앙값 MAE가 전체 중앙값보다 0.0010201648 낮았다.
   - 실행: **대상 세그먼트는 VIP·우수·성장·대체고객 전체의 상품 탐색에 필요한 설명·가격 누락 레코드**다. 설명은 `unknown`으로 표시하고, 가격은 카테고리별 중앙값과 30건 미만 그룹의 전체 중앙값 fallback을 적용한다. **기대 효과는 결측으로 인한 분석 제외와 대치 오차 감소**다.
   - 검증: 추가로 확보한 실제 가격을 같은 해시 규칙으로 홀드아웃해 MAE를 다시 잰다. 그룹 대치가 전체 중앙값보다 악화되면 현재 선택이 반증되므로 전체 중앙값으로 되돌린다.
3. **이미지 피처는 가격 결론이 아니라 참여 실험의 입력으로 제한한다.**
   - 근거: 실제 Pearson 원본 r=-0.1576이며, 노출·클릭·세션 로그는 없다.
   - 실행: **대상 세그먼트는 VIP·우수고객 513명**이다. 이들에게 동일 상품의 원본/보정 이미지를 가격과 노출 위치가 같은 조건에서 무작위 배정하는 별도 실험을 설계한다. **기대 효과는 CTR·구매 전환 개선 여부를 검증하는 것**이다.
   - 검증: 추가로 노출·클릭·세션·구매와 무작위 배정 로그를 수집해 CTR과 구매 전환을 나누어 비교한다. 대조 차이가 없으면 이미지 가설은 반증되며, 현재 상관이나 선행연구 수치를 실험 효과로 인용하지 않는다.

## 실행

고정 미러 재현에는 Python 3.11과 Git, GitHub CLI(`gh`)가 필요하다. 가장 짧은 전체 실행은 다음이다.

```bash
cd mission
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py --download-mirror
```

단계별로 확인하려면:

```bash
python scripts/prepare_hm_data.py --download-mirror --n-customers 1000 --seed 20261105
python scripts/run_analysis.py
python scripts/spec_curve.py
python scripts/build_notebook.py
python scripts/check_requirements.py
pytest -q
```

`prepare`는 고정 커밋과 CSV SHA-256을 확인하고, 이미지는 선택된 파일만 lazy-fetch한다. 같은 입력·고객 수·seed면 상품/거래/이미지 순서와 내용이 재현된다. 빈 디렉터리에서 독립 재실행해 4개 파생 파일의 해시가 일치함을 확인한 기록은 [`reports/data_reproducibility.json`](reports/data_reproducibility.json)에 있다. 성능 시간만 실행 환경에 따라 달라진다.

## Kaggle 전체 원본으로 재현

고정 미러는 **커밋된 15일 결과의 재현 경로**, Kaggle 원본은 **사용자가 직접 인증·규칙 동의 후 전체 원본을 받는 별도 경로**다. 원본 전체는 관찰 기간과 모집단이 달라지므로 상품·고객·상관·RFM 수치가 현재 리포트와 달라지는 것이 정상이다.

| 경로 | 입력 | 현재 저장소에서 확인한 범위 |
|---|---|---|
| `--download-mirror` | 고정 커밋의 실제 원본 부분집합·256px 미러 이미지 | 빈 디렉터리 독립 준비, 파생 해시 일치, 전체 분석·테스트 실행 |
| `--download-original` | 사용자 Kaggle CLI로 받은 전체 CSV·원본 이미지 | 다운로드 명령 생성·mock CLI·zip 안전성·manifest 테스트만 실행; **실제 Kaggle 다운로드 미실행** |

### 사전 준비

1. Kaggle 계정으로 competition 페이지에서 **사용자 본인이 규칙에 동의**한다.
2. `requirements-original.txt`를 설치한다. 브라우저 OAuth를 지원하는 **Kaggle CLI 2.2.5 이상**을 요구하며, 현재 2.2.5의 `auth login --help`와 `competitions download --help`를 확인했다.
3. 다운로더가 다운로드 직전에 **`kaggle auth login --force`**를 자동 실행한다. 표시되는 브라우저에서 Kaggle에 로그인하고 접근을 승인하면 된다. `--force`는 기존 로그인이 있을 때 CLI가 종료 코드 1로 멈추는 동작을 피하고 OAuth 로그인을 다시 확인하기 위한 것이다.
4. API 토큰 발급·복사·환경변수 설정은 사용하지 않는다. 스크립트는 토큰용 환경변수를 subprocess에서 제거하고, 토큰 값을 사용하거나 인증 값을 출력하지 않는다. 인증은 공식 CLI의 브라우저 OAuth에 맡긴다.
5. 전체 bundle은 약 **34.56GB**다. zip+압축 해제 공간을 위해 **80GB 이상 여유 디스크**를 권장한다. 전체 거래 CSV를 Pandas로 읽으므로 충분한 RAM도 필요하며, 대규모 원본 실행 시간과 메모리는 이 환경에서 실측하지 않았다.

```bash
cd mission
source .venv/bin/activate
pip install -r requirements-original.txt
# 별도 토큰 설정이나 선행 로그인 명령 없이 아래 다운로드 명령을 실행한다.
# 다운로더가 브라우저 OAuth 로그인을 먼저 연다.
```

### 전체 멀티모달 파이프라인을 한 번에 실행

```bash
python scripts/run_all.py --download-original --n-customers 1000 --seed 20261105
```

이 명령은 `data/_kaggle/`에 전체 competition bundle을 받고, CSV·이미지를 압축 해제한 뒤 준비→분석→60스펙→노트북→요구 검사→pytest까지 실행한다. 다운로드한 CSV 3종의 SHA-256과 이미지 수는 `data/_kaggle/kaggle_original_manifest.json`에 기록한다. 준비 스크립트는 CSV 해시를 다시 대조한다.

### 다운로드와 분석을 단계별로 실행

```bash
python scripts/fetch_kaggle_original.py \
  --out data/_kaggle --include-images --test-zip

python scripts/prepare_hm_data.py \
  --source-dir data/_kaggle \
  --image-dir data/_kaggle/images \
  --n-customers 1000 --seed 20261105

python scripts/run_analysis.py
python scripts/spec_curve.py
python scripts/build_notebook.py
python scripts/check_requirements.py
pytest -q
```

`--test-zip`은 대용량에서 CRC 검사 시간이 추가된다. 기본은 압축 해제 후 zip을 삭제하며, 보존하려면 `--keep-zip`을 사용한다. 중단 후 기존 CSV와 이미지가 있으면 재사용하고, 강제 재다운로드는 `--force`다. zip-slip 경로 검사와 CSV 누락·이미지 수 검사를 우회하지 않는다.

CSV 3종만 먼저 받으려면 `--include-images` 없이 실행한다.

```bash
python scripts/fetch_kaggle_original.py --out data/_kaggle
```

이 경우 원본 거래/고객/상품 구조는 확보하지만 **이미지가 없어 본 멀티모달 파이프라인은 완료할 수 없다**. 같은 전체 분석을 재현하려면 `--include-images`가 필요하다. 이미 사용자가 다운로드한 원본이 있다면 다운로더를 실행하지 않고 `--source-dir /path/to/hm --image-dir /path/to/hm/images`로 준비할 수 있다.

> **검증 상태를 구분한다.** 현재 실행 환경은 Kaggle 직접 네트워크가 허용되지 않아 원본 다운로드를 실행하지 않았다. `fetch_kaggle_original.py`의 `--help`와 OAuth 로그인 명령·토큰 환경변수 제거·다운로드 순서·zip·manifest에 대한 mock/단위 테스트는 실행했지만, 이를 실제 원본 다운로드나 전체 원본 분석 성공으로 표현하지 않는다.

## 검증과 산출물

- data-free GitHub 클론에서 `python mission/scripts/run_all.py --download-mirror`를 실제 실행해 데이터 준비부터 **73/73 검사·28개 테스트**까지 완료했다. 실행 시간을 제외한 분석 지표가 커밋된 결과와 동일했다. 이는 아래 Kaggle 원본 OAuth 다운로드 검증과 별개다.
- `tests/`: 날짜·결측·IQR·이미지 축·동률 RFM·실데이터 provenance·60스펙·원본 미추적 검사와 Kaggle OAuth·다운로드·run_all mock 테스트, 총 **28개**
- `scripts/check_requirements.py`: 실제 데이터/산출물에 대한 자동 요구 검사 **73/73 PASS**, 기계 판독 결과는 `reports/requirements_check.json`
- `reports/analysis.md`: 위 수치를 코드가 생성한 상세 리포트
- `figures/01_hist_price.png`~`07_spec_curve.png`: 필수 6종과 스펙 커브
- `reports/rfm_segments.csv`: 고객 ID가 없는 집계만 저장

## 한계

1. 미러 거래는 전체 원본의 마지막 15일 부분집합이다. 장기 충성도·계절성·진짜 VIP를 확정하지 못한다.
2. 선택 고객 표본 내 가격 중앙값이며 상품당 거래 수가 적은 상품도 있다. `transaction_count`를 함께 읽어야 한다.
3. 이미지는 미러가 256×256으로 축소한 실제 사진이며 이 코드가 32×32로 다시 축소했다. 원본 해상도의 선명도·질감 평가는 불가능하다.
4. 반품·주문 ID·재고·평점·노출·클릭·세션이 없다. RFM의 F는 바구니 빈도가 아니며 CTR/CVR은 계산하지 않는다.
5. 공개 미러 파일의 SHA-256과 원본 행 위치를 고정했지만, 현재 커밋 수치를 Kaggle 전체 원본과 독립 대조하거나 전체 원본을 새로 다운로드해 계산한 것은 아니다. source kind는 `github_mirror_tail_window`로 기록한다. 원본 다운로더와 `--download-original` 경로는 제공하지만 실제 Kaggle 실행은 사용자 로컬에서 별도 검증이 필요하다.

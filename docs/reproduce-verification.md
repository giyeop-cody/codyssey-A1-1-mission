# 검증 자료 출처와 재현 방법

Python 3.12/macOS arm64에서 검증했다. 통합 루트의 `requirements-verification-lock.txt`와 `docs/verification-environment.json`은 실제 사용 버전을 기록한다. `python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements-verification-lock.txt`로 환경을 만든다. 플랫폼이 달라 wheel 설치나 GUI/Tesseract 경로 설정이 필요할 수 있다. CPU 연산 순서·라이브러리·폰트·영상 인코더에 따라 산출 이미지/영상 바이트는 달라질 수 있다. 원본 데이터 해시와 수치 허용 오차를 구분해서 검증한다.

각 미션에서 `python scripts/fetch_verification_inputs.py`는 아래 원본을 내려받고 SHA-256을 확인한다. `--check`는 네트워크 없이 검사한다. 파일의 고정 URL·전체 SHA·크기는 `reports/verification-inputs.json`에 있다. 대용량 원본은 Git에 복제하지 않고 고정 리비전·해시·다운로더를 업로드했다. 공개 데이터의 이용 조건은 원 출처를 따른다. 미러의 라이선스 표기를 원본의 새 허가로 해석하지 않는다.

|과제|자료·출처|이번 사용과 범위|재현|
|---|---|---|---|
|A3-1|[MIDV 미러](https://huggingface.co/datasets/zimka/midv-DoB-mini), rev `e79ed879dcf2ac01d88d1db13d12bab35f9349b2`|322장/4점 GT는 이전 실험 기록이다. 이번에는 원문, 회귀 테스트, 실제 OCR·GUI를 재검증했다. 미러 라이선스 미표기; 원본 이미지를 새로 업로드하지 않았다.|해시 다운로더 후 `python scripts/reproduce_midv_inputs.py`; 기존 게이트 명령은 README 참조. 원본 parquet와 datasets-server 재인코딩 JPEG 결과를 혼동하지 않는다.|
|A3-1|[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), rev `7f0115a4b758a71d6473b8d085751692da2fef98`|CC BY 4.0. test 0–9의 OCR/편집 PDF는 기존 측정; 기존 reports의 개별 이미지/GT SHA에 연결된다.|고정 test parquet 다운로드. 기존 `eval_hf_cord.py`, `export_hf_cord_dewarped_pdfs.py`는 rows 서버 경로이며 바이트 재인코딩 변경 시 과거 JPEG SHA와 같다고 주장할 수 없다. 기존 보고서는 보관 증거이고 이 parquet를 이용해 동일 바이트를 재측정한 결과가 아니다.|
|A3-2|[MOTChallenge MOT17](https://motchallenge.net/data/MOT17/), [HF 미러](https://huggingface.co/datasets/Morrison1025/MOT17), rev `18fea9c0394dafc611032c5d419ff50b2b28af3f`|04·11 FRCNN의 원본 JPEG 1–600 및 GT/seqinfo. 각 30 FPS 20초, 1920×1080→960×540. FRCNN 검출 결과는 쓰지 않고 자체 MOG2+Tracker를 실행. 선택 30사건 시각 검토는 자동 전 구간 측정과 별개. 조명은 실제 04 영상에 디지털 gain 적용.|다운로더→`python scripts/evaluate_real_video.py --sequence 04`, `--sequence 11`→`python scripts/prepare_manual_review.py`→`python scripts/count_reviewed_events.py`. `reports/manual-review/README.md`에 수정 이유·프레임·판정 범위.|
|A4-1|[20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset), [고정 archive](https://ndownloader.figshare.com/files/5975967)|all subset, comp.graphics/rec.sport.baseball/sci.space/talk.politics.guns, headers/footers/quotes 제거, seed42, 최소60문자, cap4000→3618문서. sklearn fetcher의 BSD는 코퍼스 라이선스 근거가 아니다.|sklearn fetcher는 archive SHA `8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610` 검증. `python -m src.corpus` 또는 실행 스크립트. 저장 캐시가 있으면 기존 캐시 사용; 새 정제 재현은 data/20ng.json을 별도 백업 후 제거한다.|
|A4-1|[NSMC](https://github.com/e9t/nsmc), rev `eafdd77e310f399a4d0083a58ade40b55a5a308d`|ratings_train.txt; `<br />`/공백 정제, 6문자 이상, seed42 비복원 표본4000 정렬. CC BY-NC 4.0. 독립 sklearn TF-IDF/코사인 비교.|해시 다운로더 후 `python scripts/verify_external_corpus.py`. 합성 fallback을 실제 코퍼스 검증으로 인정하지 않는다.|
|A6-2|[CIFAR-10 HF](https://huggingface.co/datasets/uoft-cs/cifar10), rev `0b2714987fa478483af9968de7c934580d0bb9a2`|train50000/test10000 RGB32×32 lossless parquet→CIFAR batches; torchvision 원본 tar의 MD5를 통과했다고 주장하지 않는다. 카드 license unknown. 이번에는 기존 CNN 오분류134건 중 상위30장 시각 검토; CNN 전체 학습은 이전 증거.|해시 다운로더→`python scripts/prepare_cifar_mirror.py`→`python scripts/run_cnn.py`, `python scripts/run_errors.py`; CNN seed/학습 조건 README 및 reports 참조.|
|A6-2|[torchvision ResNet18 가중치](https://download.pytorch.org/models/resnet18-f37072fd.pth)|ImageNet 사전학습 backbone; CIFAR few-shot LP/FT 대조는 기존 결과.|해시 다운로더→`python scripts/run_transfer.py`. 전체 모델 학습은 시간이 걸리며 새 실행을 과거 체크포인트와 혼동하지 않는다.|
|A6-2|[Yahoo Finance AAPL](https://finance.yahoo.com/quote/AAPL/history/), yfinance auto_adjust=True|2021-10-11≤날짜<2026-10-10, 1255행. 원본 수집 시각 2026-10-11. train878/val125/test252. 실시간 API는 조정주가가 바뀔 수 있다.|`fixtures/aapl_daily.csv`를 업로드했다. 기본 fetch는 캐시 없을 때 이 해시 고정 스냅샷을 복원. force=True만 새 수집. `python scripts/run_timeseries.py`는 LSTM/GRU×level/delta×seed0..4, 9베이스라인 및 누수 검사를 재실행.|

A1-1, A2-1, A4-2, A5-1/2, A6-1은 외부 검증 코퍼스를 추가하지 않았다. A5-1/2는 taskmap 원문 생성기와 원본 CSV의 SHA `4ddf83dd154ee23bb2374a2bbc1aa56d537ef7ad5114c2f162078756542fe1a2`를 비교했다. A6-1 gradient/XOR는 seed가 명시된 독립 입력이다. 정확한 원문 출처·SHA와 문항별 답은 각 스터디로그 `docs/14-final-reverification.md` 및 `evaluation-mapping.json`에 있다.

실제 OCR 검증은 Tesseract 5.5.3, eng/kor traineddata(125), PATH와 TESSDATA_PREFIX를 설정해 실행했다. Mac GUI는 COCOA에서 창을 띄워 검증했으며 키 입력은 실제 waitKey 호출 이후 코드로 주입했다. 물리 키보드 조작이나 학습자 개인의 설명 능력을 검증한 것은 아니다.

## 결과 검증

통합 루트에서 `python tools/verify_submission.py`로 원문 문항 187개, 질문/답변 연결, 입력 해시·시각 검토 계수를 확인한다. `python tools/run_verification_tests.py`는 미션별 pytest를 실행한다. 대규모 학습 재실행과 GUI는 별도 명령이다. 기본 테스트의 알려진 expected failure 1개는 선택 v3 원통 dewarp의 평면 키스톤 사례다. 원문 필수 scanner의 4점 투시변환과 별개인 선택 실험 한계로 남겼다. 테스트가 통과했다고 모든 입력·실험의 무오류를 인증하지 않는다.

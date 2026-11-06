# voice-blend — RVC 음성 병합 레시피 엔지니어링

RVC(v2) 음성 모델을 **실측 음색 특징으로 군집화**한 뒤, **랜덤한 병합 경로**(순서·개수·비중)를
뽑아 ckpt를 병합하고, 그 모델로 **노래 4~5곡 + 대사 5~6개**를 렌더해 **품질 감사 → 판매 문구**까지
자동화하는 도구입니다.

- 실행 절차·법/플랫폼 체크리스트: **[`PLAYBOOK.md`](PLAYBOOK.md)**
- 실제 실행 리포트(합성 음성, GPU 불필요): **[`artifacts/demo/REPORT.md`](artifacts/demo/REPORT.md)**
- 상품 페이지 예시(자동 생성): `artifacts/demo/product_*.md`
- LLM 보조 프롬프트(선택): [`prompts/voice_analysis_prompt.md`](prompts/voice_analysis_prompt.md)

```
음성 wav ──► acoustics(9축 실측) ──► manifest(권리·누출 점검) ──► clustering(군집)
                                                                     │
        plan.json ◄── recipe(랜덤 경로·게이트) ◄── pairing(대사↔노래 짝) ◄┘
            │
            ├─► merge_recipe.py  (RVC ckpt 병합, GPU)
            ├─► render_track.py  (노래/대사 렌더, GPU)
            └─► audit_renders.py (감사: 클리핑·무음·f0점프·누락) ──► selling(상품 문구)
```

## 빠른 시작 (GPU 불필요)

```bash
cd voice-blend
pip install -r requirements.txt
python scripts/demo_plan.py     # 합성 음성 8개 → 특징 실측 → 군집 → 병합 계획 → 리포트
python -m pytest -q             # 44 tests
```

`artifacts/demo/REPORT.md`에 매니페스트·군집(k 선택 근거 포함)·짝·병합 경로·산출물 목록·
감사 결과가 실제 계산값으로 남습니다.

## 실데이터

```bash
python scripts/make_manifest.py --audio-dir data/raw \
    --template data/manifests/voices.example.csv --out data/manifests/voices.measured.csv
python scripts/build_plan.py --manifest data/manifests/voices.measured.csv \
    --dialogues data/libraries/dialogues.example.json --songs data/libraries/songs.example.json \
    --checkpoints data/manifests/ckpt_paths.example.json --out plan --seed 20261106
python plan/merge_recipe.py plan/recipes/<recipe_id>.json --out models     # RVC 환경
python scripts/audit_renders.py --plan plan/plan.json --renders renders/ --out reports/audit
```

## 모듈

| 파일 | 역할 |
|---|---|
| `voice_fusion/acoustics.py` | wav → 9축(음높이·음역·비브라토·밝기·거칠기·어택·비음·숨소리·기복) 실측. numpy만 사용 |
| `voice_fusion/manifest.py` | 음성 목록 CSV, 동의 상태, shard 중복(누출) 차단, 최소 길이 점검 |
| `voice_fusion/clustering.py` | 평균 연결 응집 클러스터링 + 실루엣 k 선택(결정론적) |
| `voice_fusion/pairing.py` | 대사↔노래 1:1 짝(음역·밝기 근접) |
| `voice_fusion/recipe.py` | SHA-256 결정론적 RNG, 랜덤 병합 경로, 가중치 트리 전개, 게이트, 설명 |
| `voice_fusion/plan.py` | 계획 JSON 생성(군집→풀→경로→산출물 목록) |
| `voice_fusion/rvc.py` | ckpt 지문·호환성 검사, 병합/추론 스크립트 생성 |
| `voice_fusion/descriptors.py` | ckpt 통계 기반 proxy 특징(⚠ 판매 문구 사용 금지, 코드가 거부) |
| `voice_fusion/analysis.py` | 렌더 품질 게이트, kNN 정체성 일관성(지표군 knockout), 문구 lint |
| `voice_fusion/selling.py` | 측정 기반 음색 서술, 고지/라이선스 문구, 가격 가이드 |
| `voice_fusion/simulate.py` | 병합 비중의 신호 레벨 검증(모델 병합 아님을 명시) |
| `voice_fusion/synth.py` | 데모용 합성 음성(판매 대상 아님) |

## 설계 원칙

1. **측정 없이는 문장 없다.** 음색 서술은 실측 특징에서만 생성합니다. ckpt 통계 proxy 특징으로는
   `selling.describe_product`가 예외를 던집니다(검증용 `allow_proxy=True`만 예외).
2. **재현 가능.** 난수는 SHA-256 기반이라 파이썬/플랫폼이 바뀌어도 같은 시드 = 같은 레시피.
   `plan.json`에 시드·트리·가중치·산출물 ID가 모두 남습니다.
3. **팔 수 없는 건 팔지 않는다.** 권리 상태·최소 길이·누출·지분·유사 음성쌍·렌더 누락이
   걸리면 파이프라인이 실패합니다(경고만 하고 넘어가지 않음).
4. **법·플랫폼 문구 기본 포함.** AI 생성 고지(국내 AI 기본법 제31조), 실존 인물 아님,
   원본 음성 재배포 금지, 개인/상업 라이선스 분리.

## 알려진 한계

- **RVC v3는 공식 배포되지 않았습니다.** v2 계열끼리만 병합하세요. 버전·샘플레이트·f0 설정이
  다르면 `CheckpointInfo.compatible_with()`가 병합을 막습니다.
- 학습/렌더는 RVC WebUI와 GPU가 필요합니다(이 저장소에 torch·모델을 포함하지 않음).
- `analysis.consistency()`의 kNN 일관성은 **정체성 안정성 지표**이지 딥페이크 탐지기가 아닙니다.
- 가격 가이드는 관행 기반 기준선이며 시장 검증값이 아닙니다.
- 이 저장소에는 실제 가수·성우 음성이 없습니다. 데모의 모든 음성은 합성 신호입니다.

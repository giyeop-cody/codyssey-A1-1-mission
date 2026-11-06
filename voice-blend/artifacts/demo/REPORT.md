# voice-blend 데모 리포트

> ⚠ 모든 음성은 **합성 신호**입니다(가수 아님). 이 리포트의 목적은
> 파이프라인이 실제로 동작함을 수치로 보이는 것입니다.
> 길이 기준: 학습 권장 600초를 데모에서만 1초로 낮췄습니다.

## 1. 매니페스트(실측 특징)

| voice | kind | f0중심(st) | 음역(st) | 밝기 | 거칠기 | 어택 | 비음(dB) | 숨소리 | 기복(dB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| syn-low-speech | speech | -2.09 | 6.71 | 0.581 | 0.269 | 0.633 | -16.90 | 0.496 | 15.61 |
| syn-mid-speech | speech | 4.50 | 4.84 | 0.695 | 0.360 | 0.646 | -12.41 | 0.614 | 13.91 |
| syn-high-speech | speech | 12.08 | 5.34 | 0.781 | 0.389 | 0.632 | -10.78 | 0.744 | 13.25 |
| syn-airy-speech | speech | 9.45 | 6.67 | 0.758 | 0.459 | 0.579 | -16.04 | 0.652 | 15.98 |
| syn-low-song | song | 1.88 | 11.19 | 0.603 | 0.270 | 0.734 | -14.46 | 0.522 | 17.05 |
| syn-mid-song | song | 5.63 | 12.36 | 0.689 | 0.323 | 0.673 | -12.10 | 0.641 | 16.50 |
| syn-high-song | song | 15.25 | 14.85 | 0.827 | 0.411 | 0.660 | -9.04 | 0.770 | 15.24 |
| syn-whisper-song | song | 8.59 | 9.03 | 0.802 | 0.536 | 0.461 | -15.95 | 0.697 | 16.15 |

## 2. 군집

- k = 3 (실루엣 0.3301)
  - k=1: 실루엣 None
  - k=2: 실루엣 0.2482
  - k=3: 실루엣 0.3301
  - k=4: 실루엣 0.3021
  - k=5: 실루엣 0.2723
  - k=6: 실루엣 0.2113
- 군집 0: syn-low-speech, syn-low-song, syn-mid-song
- 군집 1: syn-mid-speech, syn-high-speech, syn-high-song
- 군집 2: syn-airy-speech, syn-whisper-song

## 3. 대사↔노래 짝

- syn-airy-speech ↔ syn-whisper-song (거리 1.2956, 가까운 축: f0_center_st, brightness, roughness)
- syn-low-speech ↔ syn-low-song (거리 2.4104, 가까운 축: roughness, breathiness)
- syn-mid-speech ↔ syn-mid-song (거리 2.8827, 가까운 축: f0_center_st, roughness, breathiness)
- syn-high-speech ↔ syn-high-song (거리 3.7553, 가까운 축: breathiness)

## 4. 병합 레시피와 산출물

# 병합 계획 요약

- 시드: `voice-blend-v1`
- 군집: k=3 (실루엣 0.3301)
- 대사↔노래 짝: 4쌍
- 병합본: 3개

## 번지는 음색 #1 (syn-whisper-song 중심)

- ID `blend-01-9459e4` · 군집 2 · 게이트 통과
- 풀: syn-airy-speech, syn-whisper-song, syn-low-speech
  (같은 군집 우선 + 전체 보충)
- 랜덤 경로:
  1) syn-whisper-song=0.750 + syn-low-speech=0.250
  2) syn-airy-speech=0.333 + carry=0.667
- 최종 비중: syn-airy-speech 33.3%, syn-whisper-song 50.0%, syn-low-speech 16.7%
- 산출물: 노래 5곡 / 대사 5개

## 날카로운 목소리 #2 (syn-low-speech 중심)

- ID `blend-02-28aba6` · 군집 0 · 게이트 통과
- 풀: syn-low-speech, syn-low-song, syn-mid-song
  (같은 군집)
- 랜덤 경로:
  1) syn-low-speech=0.667 + syn-low-song=0.333
  2) syn-mid-song=0.250 + carry=0.750
- 최종 비중: syn-mid-song 25.0%, syn-low-speech 50.0%, syn-low-song 25.0%
- 산출물: 노래 4곡 / 대사 6개

## 맑은 톤 #3 (syn-low-song 중심)

- ID `blend-03-2d108a` · 군집 0 · 게이트 통과
- 풀: syn-mid-speech, syn-mid-song, syn-low-song, syn-low-speech
  (같은 군집 우선 + 전체 보충)
- 랜덤 경로:
  1) syn-low-song=0.667 + syn-low-speech=0.333
  2) syn-mid-speech=0.333 + carry=0.667
- 최종 비중: syn-mid-speech 33.3%, syn-low-song 44.4%, syn-low-speech 22.2%
- 산출물: 노래 5곡 / 대사 5개


## 5. 판매 설명 초안

### 번지는 음색 #1 (syn-whisper-song 중심)

- 파일: `product_blend-01-9459e4.md`
- 문구 lint: 위험 표현 없음 (면책 문맥 1건 제외)
- 게이트: 통과

```text
# 번지는 음색 #1 (syn-whisper-song 중심)

> 상태: PREVIEW(synthetic) · 병합 원본 3개 · 구조 게이트 통과

## 한 줄 소개
합성 WHISPER (song) 계열을 중심으로, 합성 AIRY (speech) 33%, 합성 LOW (speech) 17%를 섞은 하이브리드 음색입니다.

## 음색 요약(실측 기반)
- 중심 음높이는 약 197Hz(중저역 대역)
- 음역 폭 중간 (7.9반음)
- 비브라토 4.2Hz로 노래 쪽 숨결이 남아 있음
- 밝기 밝고 앞으로 나옴
- 거칠기 숨소리·기식이 도드라짐
- 자음 또렷함 또렷해 대사에 유리
- 비음 -16.1dB (콧소리 적음)
- 숨소리 많아 가까이 속삭이는 느낌
- 음량 기복 중간 (16.0dB)

…
```

### 날카로운 목소리 #2 (syn-low-speech 중심)

- 파일: `product_blend-02-28aba6.md`
- 문구 lint: 위험 표현 없음 (면책 문맥 1건 제외)
- 게이트: 통과

```text
# 날카로운 목소리 #2 (syn-low-speech 중심)

> 상태: PREVIEW(synthetic) · 병합 원본 3개 · 구조 게이트 통과

## 한 줄 소개
합성 LOW (speech) 계열을 중심으로, 합성 MID (song) 25%, 합성 LOW (song) 25%를 섞은 하이브리드 음색입니다.

## 음색 요약(실측 기반)
- 중심 음높이는 약 137Hz(낮은 남성 대역)
- 음역 폭 중간 (9.2반음)
- 비브라토 5.0Hz로 노래 쪽 숨결이 남아 있음
- 밝기 밝고 앞으로 나옴
- 거칠기 숨소리·기식이 도드라짐
- 자음 또렷함 또렷해 대사에 유리
- 비음 -15.1dB (콧소리 적음)
- 숨소리 중간
- 음량 기복 중간 (16.2dB)

…
```

### 맑은 톤 #3 (syn-low-song 중심)

- 파일: `product_blend-03-2d108a.md`
- 문구 lint: 위험 표현 없음 (면책 문맥 1건 제외)
- 게이트: 통과

```text
# 맑은 톤 #3 (syn-low-song 중심)

> 상태: PREVIEW(synthetic) · 병합 원본 3개 · 구조 게이트 통과

## 한 줄 소개
합성 LOW (song) 계열을 중심으로, 합성 MID (speech) 33%, 합성 LOW (speech) 22%를 섞은 하이브리드 음색입니다.

## 음색 요약(실측 기반)
- 중심 음높이는 약 146Hz(중저역 대역)
- 음역 폭 중간 (8.1반음)
- 비브라토 5.2Hz로 노래 쪽 숨결이 남아 있음
- 밝기 밝고 앞으로 나옴
- 거칠기 숨소리·기식이 도드라짐
- 자음 또렷함 또렷해 대사에 유리
- 비음 -14.3dB (콧소리 적음)
- 숨소리 중간
- 음량 기복 중간 (15.7dB)

…
```

## 6a. 병합 비중 시뮬레이션(파형 가중 믹스 · 모델 병합 아님)

구조 검사만 수행합니다(클리핑·무음·길이). 서로 다른 음성의 파형을 그대로 더하면
운율 지표가 무의미해지므로, 운율·f0 검사는 실제 RVC 렌더(6b 방식)에서만 합니다.

| blend | 길이(s) | peak(dBFS) | 무음비 | 판정 |
|---|---:|---:|---:|---|
| blend-01-9459e4 | 4.0 | -7.96 | 0.0 | 통과 |
| blend-02-28aba6 | 4.0 | -7.72 | 0.0 | 통과 |
| blend-03-2d108a | 4.0 | -7.22 | 0.0 | 통과 |

## 6b. 품질 게이트 검증(원본 음성)

| render | kind | 길이(s) | peak(dBFS) | 무음비 | 모음시작/s | f0점프 | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| syn-low-speech | dialogue | 4.0 | -3.1 | 0.0 | 4.009 | 0.0833 | 통과 |
| syn-mid-speech | dialogue | 4.0 | -3.1 | 0.0 | 4.761 | 0.0 | 통과 |
| syn-high-speech | dialogue | 4.0 | -3.1 | 0.0 | 5.011 | 0.0 | 통과 |
| syn-airy-speech | dialogue | 4.0 | -3.1 | 0.0 | 3.759 | 0.0 | 통과 |
| syn-low-song | song | 4.0 | -3.1 | 0.0 | 2.756 | 0.0204 | 통과 |
| syn-mid-song | song | 4.0 | -3.1 | 0.0 | 3.007 | 0.0 | 통과 |
| syn-high-song | song | 4.0 | -3.1 | 0.0 | 3.257 | 0.0 | 통과 |
| syn-whisper-song | song | 4.0 | -3.1 | 0.0 | 2.756 | 0.0 | 통과 |

## 7. 다음 단계(실데이터)

1. `data/raw/<voice_id>.wav`에 학습에 쓸 음성(10분 이상, 권리 확보) 배치
2. `scripts/make_manifest.py --audio-dir data/raw --template ...`로 실측 매니페스트 생성
3. `scripts/build_plan.py` → `plan/plan.json` 확정
4. `plan/merge_recipe.py`로 RVC ckpt 병합(GPU 환경)
5. `plan/render_track.py`로 노래 4~5곡·대사 5~6개 렌더
6. `scripts/audit_renders.py`로 품질 게이트 → 통과분만 판매 문구 생성


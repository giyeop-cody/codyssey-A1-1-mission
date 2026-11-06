"""렌더 결과 검사 — "이 병합 음색이 실제로 쓸 만한가"를 수치로 판정한다.

이 파일이 답하는 질문
1. **누출/뭉개짐**: 병합 모델이 한 원본 음성에 과하게 붙지 않았는가
   → kNN 정체성(identity) 일관성과, 지표군을 하나씩 제거했을 때의 하락률.
2. **발음 안정성**: 생성 대사가 실제로 말/노래처럼 들리는가
   → 모음 시작(vowel onset) 비율, f0 점프, 클리핑, 무음 비율.
3. **판매 문구 위험**: 실제 인물을 지칭하거나 품질을 보증하는 표현이 없는가
   → `lint_copy`.

지표군(knockout) 아이디어는 Hu & Kwon(2025)의 오디오 딥페이크 탐지
"metric-family knockout consistency"에서 가져왔다. 다만 여기서는 탐지 성능이 아니라
**병합 모델의 정체성 안정성**을 재는 데 쓴다. 결과는 절대적 진실이 아니라
"검사 통과/주의" 판단용 지표로만 쓴다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("identity", r"f0_(center|range)|brightness|roughness|nasality|mfcc|formant|centroid"),
    ("rhythm", r"vowel_rate|onset_rate|speech_rate|attack|burst|pause"),
    ("vibrato", r"vibrato|jitter|shimmer"),
    ("dynamics", r"expressiveness|loudness|rms|dynamic|breathiness"),
)

# 판매 문구에서 절대 쓰면 안 되는/위험한 표현.
RISKY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"실제\s*(가수|성우|배우|아이돌|인물)\s*(의\s*)?(목소리|음성)", "실존 인물 지칭"),
    (r"(복제|재현|모사)\s*(했습니다|완벽|100%)", "복제 단정"),
    (r"(공식|정식)\s*(라이선스|제휴|콜라보)", "공식 오인"),
    (r"(완벽|무조건|100%)\s*(재현|같은|동일)", "품질 보증 단정"),
    (r"(딥페이크|대리\s*발화)", "오용 유도"),
)

# 위 표현이 "…가 아니며", "금지: …" 같은 부정/금지 문맥에 있으면 위반이 아니다.
NEGATION_TOKENS = ("아니", "않", "없", "금지", "불가", "못", "주의", "금합니다", "아님")
NEGATION_WINDOW_BEFORE = 70
NEGATION_WINDOW_AFTER = 20


def metric_family(name: str) -> str:
    for family, pattern in FAMILY_PATTERNS:
        if re.search(pattern, name):
            return family
    return "other"


def group_metrics(metric_names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in metric_names:
        groups.setdefault(metric_family(name), []).append(name)
    return groups


# ------------------------------------------------------------------ kNN 一致
def _knn_predict(train_x, train_y, query_x, k: int) -> np.ndarray:
    dist = ((query_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=-1)
    k = int(max(1, min(k, train_x.shape[0])))
    index = np.argsort(dist, axis=1)[:, :k]
    return train_y[index].mean(axis=1)


def _r2(truth: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def consistency(
    features: np.ndarray,
    target: np.ndarray,
    names: list[str],
    *,
    k: int = 5,
    folds: int = 4,
    drop_threshold: float = 60.0,
) -> dict:
    """특징→타깃(예: 원본 화자 ID의 임베딩 축) 예측의 교차검증 R²와 지표군 기여도.

    지표군 하나를 제거했을 때 R²가 `drop_threshold`% 넘게 떨어지면 그 군이
    정체성 판별에 결정적이라고 본다(→ 병합 시 그 축이 흐려지면 위험 신호).
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if y.ndim > 1:
        y = y[:, 0]
    n = x.shape[0]
    if n < folds * 2:
        raise ValueError("표본이 너무 적습니다(폴드당 2개 이상 필요)")
    fold_of = np.array([int.from_bytes(
        __import__("hashlib").sha256(f"fold:{i}".encode()).digest()[:4], "big") % folds
        for i in range(n)])

    def cv_score(columns: list[int]) -> float:
        if not columns:
            return 0.0
        pred = np.zeros(n)
        for fold in range(folds):
            test = fold_of == fold
            pred[test] = _knn_predict(x[~test][:, columns], y[~test], x[test][:, columns], k)
        return _r2(y, pred)

    groups = group_metrics(names)
    full_cols = list(range(len(names)))
    full = cv_score(full_cols)
    knockouts = []
    for family, members in sorted(groups.items()):
        cols = [i for i, name in enumerate(names) if name not in set(members)]
        score = cv_score(cols)
        drop = 0.0 if full <= 1e-9 else float((full - score) / full * 100.0)
        knockouts.append({
            "family": family,
            "metrics": members,
            "r2_without": round(score, 4),
            "drop_pct": round(drop, 2),
            "critical": bool(drop >= drop_threshold),
        })
    return {
        "knn_k": k, "folds": folds,
        "r2_full": round(full, 4),
        "knockouts": knockouts,
        "critical_families": [k_["family"] for k_ in knockouts if k_["critical"]],
    }


# ------------------------------------------------------------ 발화 안정성
def vowel_onsets(signal: np.ndarray, rate: int, hop_ms: float = 10.0,
                 delta_db: float = 3.0) -> tuple[int, float]:
    """모음 시작(음절 핵) 개수와 초당 비율. RMS 상승 엣지 기반 근사."""
    hop = max(1, int(rate * hop_ms / 1000))
    frames = signal.size // hop
    if frames < 8:
        return 0, 0.0
    rms = np.sqrt((signal[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    # 5프레임 이동평균으로 잡음 완화
    kernel = np.ones(5) / 5
    smooth = np.convolve(db, kernel, mode="same")
    floor = float(np.percentile(smooth, 20))
    ceiling = float(np.percentile(smooth, 95))
    if ceiling - floor < 6.0:
        return 0, 0.0
    threshold = floor + 0.35 * (ceiling - floor)
    above = smooth > threshold
    onsets = int(np.sum(above[1:] & ~above[:-1]))
    duration = frames * hop / rate
    return onsets, float(onsets / duration) if duration > 0 else 0.0


def f0_jump_ratio(f0: np.ndarray, limit_semitones: float = 2.0) -> float:
    """연속 유성 프레임 사이 f0 변화가 limit을 넘는 비율(병합 후 피치 흔들림 신호)."""
    f0 = np.asarray(f0, dtype=float)
    voiced = np.isfinite(f0)
    if voiced.sum() < 3:
        return 0.0
    semis = 12 * np.log2(np.where(voiced, f0, np.nan) / 130.8128)
    diffs = np.abs(np.diff(semis))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs > limit_semitones))


@dataclass
class RenderAudit:
    render_id: str
    kind: str
    duration_s: float
    peak_dbfs: float
    silence_ratio: float
    vowel_rate: float | None
    f0_jump_ratio: float | None
    flags: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.flags

    def to_dict(self) -> dict:
        return {**self.__dict__, "passed": self.passed}


def audit_render(render_id: str, kind: str, signal: np.ndarray, rate: int,
                 *, min_duration: float = 1.5,
                 max_silence: float = 0.55,
                 max_jump: float = 0.20,
                 prosody: bool = True) -> RenderAudit:
    """단일 렌더의 기본 품질 게이트. 실패 사유를 사람이 읽을 수 있게 남긴다.

    `prosody=False`는 구조 검사(길이·클리핑·무음)만 한다. 서로 다른 음성의 파형을
    단순 가산한 **시뮬레이션 믹스**처럼, 운율 지표가 원래 의미를 갖지 않는 입력에 쓴다.
    """
    from .acoustics import f0_track, stft_frames

    if kind not in ("song", "dialogue"):
        raise ValueError("kind는 song|dialogue")
    duration = signal.size / rate
    peak = float(np.max(np.abs(signal)) + 1e-12)
    peak_dbfs = 20 * np.log10(peak)
    _, rms = stft_frames(signal, rate)
    silence = float(np.mean(rms < max(1e-4, np.percentile(rms, 95) * 0.02)))
    if prosody:
        _, rate_per_sec = vowel_onsets(signal, rate)
        f0, _ = f0_track(signal, rate)
        jumps = f0_jump_ratio(f0)
    else:
        rate_per_sec, jumps = float("nan"), float("nan")

    flags = []
    if duration < min_duration:
        flags.append(f"길이 부족({duration:.2f}s)")
    if peak_dbfs > -0.1:
        flags.append(f"클리핑 위험(peak {peak_dbfs:.2f} dBFS)")
    if silence > max_silence:
        flags.append(f"무음 과다({silence:.0%})")
    if prosody and kind == "dialogue" and rate_per_sec < 2.0:
        flags.append(f"모음 시작 부족({rate_per_sec:.2f}/s)")
    if prosody and kind == "song" and rate_per_sec < 0.8:
        flags.append(f"음절 진행 부족({rate_per_sec:.2f}/s)")
    if prosody and jumps > max_jump:
        flags.append(f"f0 점프 과다({jumps:.0%})")
    return RenderAudit(render_id=render_id, kind=kind, duration_s=round(duration, 3),
                       peak_dbfs=round(peak_dbfs, 2), silence_ratio=round(silence, 4),
                       vowel_rate=None if not prosody else round(rate_per_sec, 3),
                       f0_jump_ratio=None if not prosody else round(jumps, 4),
                       flags=flags)


# ---------------------------------------------------------------- 문구 검사
def lint_copy(text: str, *, mark_negated: bool = True) -> list[dict]:
    """위험 표현 탐지. `negated=True`는 면책/금지 문맥에서 나온 것(위반 아님).

    완벽한 판별기는 아니다. 자동 생성 문구의 **회귀 방지용 안전망**으로만 쓰고,
    최종 판단은 사람이 한다. 부정 문맥 판정은 앞 70자/뒤 20자 창으로 근사한다.
    """
    hits = []
    for pattern, label in RISKY_PATTERNS:
        for match in re.finditer(pattern, text):
            before = text[max(0, match.start() - NEGATION_WINDOW_BEFORE):match.start()]
            after = text[match.end():match.end() + NEGATION_WINDOW_AFTER]
            negated = mark_negated and any(
                token in before or token in after for token in NEGATION_TOKENS)
            hits.append({"pattern": pattern, "label": label, "match": match.group(0),
                         "start": match.start(), "negated": bool(negated)})
    return hits


def lint_violations(text: str) -> list[dict]:
    """실제로 고쳐야 하는 위반만(면책 문맥 제외)."""
    return [hit for hit in lint_copy(text) if not hit["negated"]]

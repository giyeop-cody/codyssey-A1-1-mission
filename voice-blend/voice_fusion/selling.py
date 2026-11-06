"""판매용 설명글 생성 — "음색이 어떤지"를 근거 있는 문장으로 바꾼다.

원칙
1. **측정에 근거한다.** 문장은 특징 벡터(실측)와 게이트/일관성 결과에서만 만든다.
   근거 없는 수식어("미친 퀄리티")는 넣지 않는다.
2. **팔 수 없는 것은 팔지 않는다.** 대사/노래 비중, 최대 지분, 유사 음성쌍 같은
   정보를 상품 페이지에 그대로 쓴다(구매자가 재현·검증할 수 있게).
3. **법·플랫폼 문구를 기본 포함한다.** AI 생성 표시(AI 기본법 제31조), 실존 인물
   아님, 원본 음성 재배포 금지, 학습 데이터 동의 상태.
"""
from __future__ import annotations

from .analysis import lint_copy, lint_violations
from .features import FEATURE_LABELS, FEATURE_NAMES, Standardizer
from .manifest import Manifest
from .recipe import MergeNode, check_recipe, final_weights

# 축 → 쉬운 말 사전(구매자가 RVC를 몰라도 이해할 수 있게)
PLAIN = {
    "f0_center_st": "중심 음높이",
    "f0_range_st": "음역 폭",
    "f0_vibrato_hz": "비브라토 속도",
    "brightness": "밝기",
    "roughness": "거칠기(속삭임 성분)",
    "attack": "자음 또렷함",
    "nasality": "코맹맹이/비음",
    "breathiness": "숨소리",
    "expressiveness": "음량 기복",
}

LICENSE_KIT = """## 라이선스 (상품에 그대로 포함하세요)
- 허용: 개인 창작물(노래 커버, 게임·영상 대사, 방송)에서의 음성 변환 사용.
- 금지: 모델 파일/병합본/인덱스의 재배포·재판매, 실존 인물·특정 작품 사칭,
  정치·선거·타인 비방·사기 목적 사용, 미성년자 성적 콘텐츠.
- 원본 학습 음성(가수 음원·게임 대사)은 **제공하지 않습니다.** 모델만 판매합니다.
- 생성물에는 AI 생성 사실을 표시해야 합니다(국내 AI 기본법 제31조, 2026-01-22 시행).
- 구매자는 자신의 관할 법규를 확인할 책임이 있습니다.
"""

COMPLIANCE_KIT = """## 고지 문구(상품 페이지 상단)
- 본 상품은 인공지능(AI) 음성 변환 모델입니다. 실제 인물의 목소리가 아니며,
  특정 실존 인물·아티스트와 제휴·후원 관계가 없습니다.
- 학습에는 권리를 보유/허락받은 음성만 사용했습니다: {consent_summary}.
- 출력 음성은 AI 생성물입니다. 게시·판매 시 AI 사용 사실을 표시하세요.
- 모델 파일의 재판매·재배포는 금지됩니다.
"""


def _phrase(value: float, low: str, mid: str, high: str,
            q1: float, q2: float) -> str:
    return low if value < q1 else (mid if value < q2 else high)


def blend_profile(blended: dict[str, float]) -> list[str]:
    """절대값이 아니라 **해석 가능한 구간**으로 음색을 서술한다."""
    lines = []
    center_hz = 130.8128 * 2 ** (blended["f0_center_st"] / 12)
    lines.append(f"중심 음높이는 약 {center_hz:.0f}Hz"
                 f"({'낮은 남성' if center_hz < 140 else '중저역' if center_hz < 210 else '중고역' if center_hz < 330 else '높은 여성'} 대역)")
    lines.append("음역 폭 " + _phrase(blended["f0_range_st"], "좁아 말하듯 안정적", "중간", "넓어 감정 기복이 큼", 6, 14) + f" ({blended['f0_range_st']:.1f}반음)")
    lines.append("비브라토 " + (f"{blended['f0_vibrato_hz']:.1f}Hz로 " if blended["f0_vibrato_hz"] > 0 else "거의 없어 ") +
                 ("노래 쪽 숨결이 남아 있음" if blended["f0_vibrato_hz"] >= 4 else "곧은 발성 위주"))
    lines.append("밝기 " + _phrase(blended["brightness"], "어둡고 두꺼움", "중립", "밝고 앞으로 나옴", 0.30, 0.45))
    lines.append("거칠기 " + _phrase(blended["roughness"], "매끈", "중간", "숨소리·기식이 도드라짐", 0.12, 0.22))
    lines.append("자음 또렷함 " + _phrase(blended["attack"], "부드러워 노래에 유리", "중간", "또렷해 대사에 유리", 0.25, 0.45))
    lines.append("비음 " + (f"{blended['nasality']:.1f}dB " +
                            ("(콧소리 있음)" if blended["nasality"] > -8 else "(콧소리 적음)")))
    lines.append("숨소리 " + _phrase(blended["breathiness"], "적음", "중간", "많아 가까이 속삭이는 느낌", 0.25, 0.55))
    lines.append("음량 기복 " + _phrase(blended["expressiveness"], "평탄", "중간", "기복 커 감정 표현이 쉬움", 12, 24) +
                 f" ({blended['expressiveness']:.1f}dB)")
    return lines


def describe_product(
    *,
    name: str,
    node: MergeNode,
    manifest: Manifest,
    standardizer: Standardizer,
    blended_raw: dict[str, float],
    consistency: dict | None = None,
    audited: list[dict] | None = None,
    price_krw: int | None = None,
    status: str = "PREVIEW",
    allow_proxy: bool = False,
) -> dict:
    """상품 설명 초안(마크다운)과 검사 결과를 함께 돌려준다.

    `allow_proxy=False`(기본)면 체크포인트 통계 proxy 특징만 가진 음성이 섞인
    병합본은 **판매 문구 생성이 거부**된다. proxy는 음색의 실측이 아니기 때문이다.
    """
    weights = final_weights(node)
    z = {e.voice_id: standardizer.transform(e.vector())[0] for e in manifest.entries}
    raw_avg = {}
    for name_axis in FEATURE_NAMES:
        raw_avg[name_axis] = sum(weights[v] * manifest.by_id(v).features[name_axis]
                                 for v in weights)
    top = sorted(weights.items(), key=lambda kv: -kv[1])
    gate = check_recipe(node, z)
    meta = manifest.meta()
    proxy_sources = [v for v in weights
                     if manifest.by_id(v).feature_source.startswith("ckpt_stats")]
    if proxy_sources and not allow_proxy:
        raise ValueError(
            "판매 문구 생성 거부: proxy 특징만 가진 음성이 포함됨 "
            f"({', '.join(proxy_sources)}). --allow-proxy는 검증용으로만 쓰세요."
        )

    lines = [f"# {name}", "",
             f"> 상태: {status} · 병합 원본 {len(weights)}개 · 구조 게이트 "
             f"{'통과' if gate.passed else '실패'}", ""]
    if status == "PREVIEW":
        lines += ["> ⚠ 이 문서는 **렌더 이전 예측**입니다. 실제 음색 문장은 렌더 후 "
                  "`audit` 결과로 대체해야 합니다.", ""]

    lines += ["## 한 줄 소개",
              f"{meta[top[0][0]]['source']} 계열을 중심으로, "
              + ", ".join(f"{meta[v]['source']} {w:.0%}" for v, w in top[1:3])
              + "를 섞은 하이브리드 음색입니다.", ""]

    lines += ["## 음색 요약(실측 기반)"]
    lines += [f"- {line}" for line in blend_profile(blended_raw)]
    lines += ["", "## 어디에 쓰면 좋은가"]
    bright, attack = blended_raw["brightness"], blended_raw["attack"]
    use = []
    if attack >= 0.35:
        use.append("게임 대사·내레이션(자음이 또렷해 대본 전달이 안정적)")
    if blended_raw["f0_vibrato_hz"] >= 4 or blended_raw["expressiveness"] >= 18:
        use.append("노래 커버(비브라토·음량 기복이 살아 있음)")
    if bright <= 0.35:
        use.append("잔잔한 발라드·ASMR 톤")
    if bright >= 0.45:
        use.append("밝은 팝·광고 톤")
    lines += [f"- {item}" for item in (use or ["범용(특정 장르에 치우치지 않음)"])]

    lines += ["", "## 구성 비중(재현용)"]
    for voice, weight in top:
        entry = manifest.by_id(voice)
        lines.append(f"- {voice} · {entry.source} · {entry.kind} · {weight:.1%}")
    if consistency:
        lines += ["", "## 정체성 검사(kNN 교차검증)",
                  f"- 전체 R² {consistency['r2_full']}",
                  f"- 결정적 지표군: {', '.join(consistency['critical_families']) or '없음(분산된 음색)'}"]
    if audited:
        passed = sum(1 for row in audited if row["passed"])
        lines += ["", f"## 렌더 검사 {passed}/{len(audited)} 통과"]
        for row in audited:
            mark = "OK" if row["passed"] else "주의: " + "; ".join(row["flags"])
            lines.append(f"- {row['render_id']} ({row['kind']}) {mark}")

    if price_krw:
        lines += ["", "## 가격", f"- {price_krw:,}원 (개인용) / 문의 시 상업용"]

    consent_summary = ", ".join(f"{v}:{meta[v]['consent']}" for v in weights)
    lines += ["", COMPLIANCE_KIT.format(consent_summary=consent_summary), LICENSE_KIT,
              "## 재현 정보",
              f"- 병합 트리: {node.to_dict()}",
              f"- 최종 가중치: {weights}"]

    body = "\n".join(lines)
    return {"markdown": body, "lint": lint_violations(body),
            "lint_all": lint_copy(body), "weights": weights,
            "gate": gate.to_dict(), "raw_avg_note": raw_avg}


def price_tiers(song_count: int, dialogue_count: int, base_krw: int = 12000) -> dict:
    """구성 기반 가격 가이드(플랫폼 수수료·세금은 별도 표기)."""
    assets = song_count + dialogue_count
    core = base_krw + 1200 * max(0, song_count - 3) + 600 * max(0, dialogue_count - 4)
    return {
        "personal": int(round(core / 100) * 100),
        "commercial": int(round(core * 3.5 / 100) * 100),
        "assets": {"songs": song_count, "dialogues": dialogue_count},
        "note": f"총 {assets}개 렌더 포함 기준. 상업용은 3.5배 관행(플랫폼별 상이).",
    }

"""A1-1 통과 게이트 테스트 — 자동 생성(hub/tools/gen_mission_repos.py).

각 함수의 docstring 이 곧 검수표다. 조건에 적힌 숫자는 학습기록 레포의 실측값이며,
판정은 **이 레포의 산출물**(reports/*.json, figures/*, *.md)에 대해 내린다.
"""
import json
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import needs  # noqa: E402

def metric(name):
    f = ROOT / 'reports' / name
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else None

@needs()
def test_01_G1_figures_6종_제목_축라벨_100():
    """[G1] figures/ 6종, 제목·축라벨 100% | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G1")

@needs()
def test_02_G2_IQR_플래그_비율을_분포별로_나누어_표기():
    """[G2] IQR 플래그 비율을 분포별로 나누어 표기 | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G2")

@needs()
def test_03_G3_전체_vs_그룹별_대치_MAE_표():
    """[G3] 전체 vs 그룹별 대치 MAE 표 | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G3")

@needs()
def test_04_G4_VIP_평균_Recency가_전체_중앙값보다():
    """[G4] VIP 평균 Recency가 전체 중앙값보다 작다 | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G4")

@needs()
def test_05_G5_벡터화_n배에_조건_명시_무엇과_비교():
    """[G5] 벡터화 n배에 조건 명시(무엇과 비교) | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G5")

@needs()
def test_06_G6_방어_가능한_분석_선택을_전부_돌려_r_분포():
    """[G6] 방어 가능한 분석 선택을 **전부** 돌려 r 분포를 보고하고, 단일 계수 인용을 금지 | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G6")


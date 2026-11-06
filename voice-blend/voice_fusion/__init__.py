"""voice-blend: RVC 계열 음성 모델 병합 레시피 엔지니어링.

파이프라인
    acoustics   실제 wav → 9축 음색 특징(실측)
    manifest    음성 목록·동의·누출 점검
    clustering  비슷한 특징끼리 군집(결정론적)
    pairing     대사 음성 ↔ 노래 음성 짝
    recipe      가중치 트리 · 랜덤 경로 · 게이트 · 설명
    plan        전체 계획(JSON) 생성
    rvc/decriptors  체크포인트 검사·스크립트 생성(proxy 특징)
    analysis    렌더 품질·정체성 일관성 검사
    selling     판매 설명글·가격·컴플라이언스 문구
    synth       데모용 합성 음성(판매 대상 아님)
"""
__version__ = "0.1.0"

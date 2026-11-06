"""RVC 연동 — 체크포인트 지문(signature), 병합/추론 스크립트 생성, 정합성 검사.

이 모듈은 **torch가 있을 때만** 실제 ckpt를 읽는다. 없으면 명확한 안내와 함께
`MissingDependency`를 던진다(조용히 넘어가지 않는다).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

import numpy as np

# RVC 학습 산출물은 보통 logs/<실험명>/weights/*.pth + added_*.index 다.
CKPT_SUFFIXES = (".pth", ".pt", ".ckpt")


class MissingDependency(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    version: str          # v1 | v2 | unknown
    sample_rate: int      # 32000 | 40000 | 48000 | 0
    f0: bool              # True=with pitch, False=without pitch
    tensors: int
    parameter_count: int
    signature: str        # 텐서 통계 요약 해시(중복 모델 탐지용)
    signature_vector: list[float]

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        return payload

    def compatible_with(self, other: "CheckpointInfo", strict: bool = True) -> list[str]:
        """병합 가능 여부. RVC는 같은 아키텍처/샘플레이트/피치 설정만 섞을 수 있다."""
        problems = []
        if self.sample_rate != other.sample_rate:
            problems.append(f"샘플레이트 불일치 {self.sample_rate} vs {other.sample_rate}")
        if strict and self.version != other.version:
            problems.append(f"버전 불일치 {self.version} vs {other.version}")
        if strict and self.f0 != other.f0:
            problems.append(f"f0 설정 불일치 {self.f0} vs {other.f0}")
        if self.tensors != other.tensors:
            problems.append(f"텐서 개수 불일치 {self.tensors} vs {other.tensors}")
        if self.signature == other.signature:
            problems.append("동일 체크포인트 중복(지문 일치)")
        return problems


def _load_torch():
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - 환경 의존
        raise MissingDependency(
            "체크포인트를 읽으려면 torch가 필요합니다: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def inspect_checkpoint(path: str, strict_safe: bool = True) -> CheckpointInfo:
    """RVC .pth를 읽어 버전/샘플레이트/피치 설정과 지문을 만든다.

    `strict_safe=True`면 weights_only 로드를 우선 시도한다(임의 코드 실행 방지).
    """
    file = pathlib.Path(path)
    if file.suffix.lower() not in CKPT_SUFFIXES:
        raise ValueError(f"RVC 체크포인트 확장자가 아닙니다: {file.suffix}")
    torch = _load_torch()
    try:
        payload = torch.load(str(file), map_location="cpu", weights_only=True)
    except TypeError:  # 구버전 torch
        payload = torch.load(str(file), map_location="cpu")
    except Exception as exc:
        if strict_safe:
            raise RuntimeError(
                f"{file.name}: weights_only 로드 실패 — 신뢰할 수 없는 출처면 열지 마세요."
            ) from exc
        raise

    optional = {}
    for key in ("version", "sr", "f0", "info"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if value is not None:
            optional[key] = value
    state = payload.get("model") if isinstance(payload, dict) and "model" in payload else payload
    if not hasattr(state, "items"):
        raise ValueError("state_dict를 찾지 못했습니다(RVC 체크포인트가 아닌 듯)")

    vector, tensors, params = [], 0, 0
    name_hash = hashlib.sha256()
    for name in sorted(state.keys()):
        tensor = state[name]
        if not hasattr(tensor, "detach"):
            continue
        values = tensor.detach().float().reshape(-1)
        if values.numel() == 0:
            continue
        tensors += 1
        params += int(values.numel())
        mean = float(values.mean())
        std = float(values.std(unbiased=False))
        vector.extend([mean, std])
        name_hash.update(f"{name}:{tuple(tensor.shape)};".encode())

    version = str(optional.get("version", "unknown")).lower()
    if version not in ("v1", "v2"):
        version = "v2" if any(k.startswith("enc_p.") for k in state.keys()) else "unknown"
    sample_rate = int(optional.get("sr", 0) or 0)
    f0 = bool(optional.get("f0", True))
    summary = np.array(vector, dtype=float)
    signature = hashlib.sha256(
        (name_hash.hexdigest() + np.round(summary, 6).tobytes().hex()).encode()
    ).hexdigest()[:16]
    return CheckpointInfo(path=str(file), version=version, sample_rate=sample_rate,
                          f0=f0, tensors=tensors, parameter_count=params,
                          signature=signature,
                          signature_vector=np.round(summary, 6).tolist())


# ---------------------------------------------------- 스크립트 생성(실행용)
MERGE_SCRIPT = '''"""RVC ckpt 병합 실행기 — voice-blend가 생성한 레시피를 그대로 적용한다.

사용:
    python merge_recipe.py plan/recipes/<recipe_id>.json --out models/

레시피는 트리 구조이며, 각 노드는 "자식들을 weights로 선형 결합"한다.
RVC의 ckpt merge와 동일한 연산(가중 선형 결합)이므로, 트리를 리프까지 전개한
최종 가중치(final_weights)와 수학적으로 같은 결과를 낸다. 다만 중간 노드를
디스크에 남기지 않으므로 여기서는 **전개된 최종 가중치를 한 번에** 적용한다
(반복 병합에 따른 dtype 반올림 누적을 피하는 편이 재현에 유리).
"""
import argparse
import json
import pathlib

import torch


def load_state(path):
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    return payload, state


def merge(recipe_path, assets, out_dir):
    recipe = json.loads(pathlib.Path(recipe_path).read_text(encoding="utf-8"))
    weights = recipe["final_weights"]
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("가중치 합이 1이 아닙니다: %s" % total)

    members = []
    for voice_id, weight in weights.items():
        ckpt = pathlib.Path(recipe["checkpoints"][voice_id])
        if not ckpt.is_absolute() and assets:
            ckpt = pathlib.Path(assets) / ckpt
        payload, state = load_state(ckpt)
        members.append((voice_id, float(weight), payload, state))

    base_payload, base_state = members[0][2], members[0][3]
    merged = {}
    for key, tensor in base_state.items():
        if not hasattr(tensor, "detach"):
            continue
        acc = None
        for _, weight, _, state in members:
            if key not in state:
                raise SystemExit("텐서 키 불일치: %s" % key)
            piece = state[key].float() * weight
            acc = piece if acc is None else acc + piece
        if tuple(acc.shape) != tuple(tensor.shape):
            raise SystemExit("형상 불일치 %s: %s vs %s"
                             % (key, tuple(acc.shape), tuple(tensor.shape)))
        merged[key] = acc.to(tensor.dtype)

    out = dict(base_payload) if isinstance(base_payload, dict) else {}
    out["model"] = merged
    out.setdefault("info", recipe.get("name", "merged"))
    out["voice_blend"] = {
        "recipe_id": recipe.get("recipe_id"),
        "final_weights": weights,
        "gate": recipe.get("gate"),
    }
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ("%s.pth" % recipe.get("recipe_id", "merged"))
    torch.save(out, target)
    print("saved %s (%d tensors)" % (target, len(merged)))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", help="plan/recipes/<recipe_id>.json")
    parser.add_argument("--assets", default="", help="체크포인트 상대경로 기준 디렉터리")
    parser.add_argument("--out", default="merged")
    args = parser.parse_args()
    merge(args.recipe, args.assets, args.out)


if __name__ == "__main__":
    main()
'''

INFER_SCRIPT = '''"""RVC 추론 실행기 — 병합 모델로 대상 음원을 변환한다.

전제: RVC WebUI(또는 Applio 등 RVC 호환 포크)가 설치되어 있고, --repo로 경로를 준다.
이 스크립트는 **모델 로드는 RVC 쪽 구현에 위임**하고, 우리는 입출력 규약만 고정한다.
"""
import argparse
import json
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="RVC WebUI 루트")
    parser.add_argument("--model", required=True, help="병합된 .pth")
    parser.add_argument("--index", default="", help="선택: added_*.index")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--f0-method", default="rmvpe")
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.repo)))
    try:
        from infer.modules.vc.modules import VC  # type: ignore
        from configs.config import Config  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "RVC WebUI를 찾지 못했습니다. --repo를 RVC 루트로 지정하고, "
            "해당 환경에서 이 스크립트를 실행하세요. (원인: %s)" % exc
        )

    config = Config()
    vc = VC(config)
    vc.get_vc(args.model)
    _, wav_opt = vc.vc_single(
        0, args.input, args.pitch, None, args.f0_method,
        args.index, None, args.index_rate, 3, None, "0", 0.33,
    )
    data = wav_opt[1][1] if isinstance(wav_opt, tuple) else wav_opt
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        handle.write(data)
    print(json.dumps({"output": args.output, "bytes": len(data)}))


if __name__ == "__main__":
    main()
'''


def write_merge_script(directory: str | pathlib.Path) -> pathlib.Path:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "merge_recipe.py"
    path.write_text(MERGE_SCRIPT, encoding="utf-8")
    return path


def write_infer_script(directory: str | pathlib.Path) -> pathlib.Path:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "render_track.py"
    path.write_text(INFER_SCRIPT, encoding="utf-8")
    return path


def check_assets(assets_dir: str | pathlib.Path, sr: int) -> list[str]:
    """RVC 학습 전제 파일(사전학습 가중치)이 있는지 확인한다."""
    assets = pathlib.Path(assets_dir)
    folder = assets / ("pretrained_v2" if assets.joinpath("pretrained_v2").exists() else "pretrained")
    missing = []
    for prefix in ("f0G", "f0D", "G", "D"):
        name = folder / f"{prefix}{sr // 1000}.pth"
        if not name.exists():
            missing.append(str(name))
    hubert = assets / "hubert" / "hubert_base.pt"
    if not hubert.exists():
        missing.append(str(hubert))
    return missing

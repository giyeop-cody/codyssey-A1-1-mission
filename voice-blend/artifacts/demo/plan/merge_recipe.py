"""RVC ckpt 병합 실행기 — voice-blend가 생성한 레시피를 그대로 적용한다.

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

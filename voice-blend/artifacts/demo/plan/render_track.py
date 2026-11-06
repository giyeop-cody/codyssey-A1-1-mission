"""RVC 추론 실행기 — 병합 모델로 대상 음원을 변환한다.

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

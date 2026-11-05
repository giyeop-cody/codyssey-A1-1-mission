import json
import pathlib
import re
import subprocess

import numpy as np
import pandas as pd

from scripts.prepare_hm_data import customer_key

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_customer_hash_selection_is_stable_and_seed_sensitive():
    ids = ["000abc", "001def", "002ghi"]
    first = sorted(ids, key=lambda value: (customer_key(value, 20261105), value))
    second = sorted(ids, key=lambda value: (customer_key(value, 20261105), value))
    changed = sorted(ids, key=lambda value: (customer_key(value, 42), value))
    assert first == second
    assert len(customer_key(ids[0], 20261105)) == 64
    assert customer_key(ids[0], 20261105) != customer_key(ids[0], 42)
    assert changed != first


def test_committed_metrics_describe_real_hm_data_not_synthetic_results():
    metrics = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    provenance = metrics["data_provenance"]
    assert provenance["synthetic"] is False
    assert provenance["source_kind"] in {"github_mirror_tail_window", "kaggle_original_user_supplied"}
    assert provenance["source_transaction_rows"] >= 531967
    assert len(provenance["source_transaction_sha256"]) == 64
    assert "not KRW" in provenance["price_unit"]
    assert metrics["schema"]["n_products"] >= 1000
    assert metrics["schema"]["n_images"] == metrics["schema"]["n_products"]
    assert metrics["actual_missing"]["description_after"] == 0
    requirements_check = json.loads((ROOT / "reports/requirements_check.json").read_text(encoding="utf-8"))
    assert requirements_check["all_pass"] is True
    assert requirements_check["passed"] == requirements_check["total"] >= 70
    check_names = {record["name"] for record in requirements_check["checks"]}
    assert "상관 2쌍 각각 해석" in check_names
    assert "Kaggle 브라우저 OAuth 로그인 강제" in check_names
    reproducibility = json.loads((ROOT / "reports/data_reproducibility.json").read_text(encoding="utf-8"))
    assert reproducibility["all_matched"] is True
    assert all(len(record["sha256"]) == 64 and record["matched_independent_preparation"] is True
               for record in reproducibility["files"].values())


def test_correlation_and_spec_curve_are_realized_and_finite():
    metrics = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    spec = pd.read_csv(ROOT / "reports/spec_curve.csv")
    assert metrics["correlation"]["n_pairs"] == metrics["schema"]["n_products"]
    assert np.isfinite(metrics["correlation"]["price__img_mean"])
    assert np.isfinite(metrics["correlation"]["price__transaction_count"])
    assert len(spec) == 60
    assert np.isfinite(spec["r"]).all()
    assert spec["r"].min() <= metrics["correlation"]["price__img_mean"] <= spec["r"].max()


def test_rfm_aggregate_has_direction_evidence_without_customer_rows():
    metrics = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    segments = pd.read_csv(ROOT / "reports/rfm_segments.csv")
    assert len(segments) == 4
    assert metrics["rfm"]["gate_VIP_recency_lower"] is True
    assert metrics["rfm"]["gate_naive_reversal_is_worse"] is True
    assert not (ROOT / "reports/rfm.csv").exists()
    assert "customer_id" not in segments.columns
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    business = readme.split("## 비즈니스 인사이트", 1)[1].split("\n## 실행", 1)[0]
    assert business.count("대상 세그먼트") >= 3
    assert business.count("기대 효과") >= 3
    assert "반증" in business


def test_all_seven_figures_and_executed_notebook_exist():
    figures = sorted((ROOT / "figures").glob("*.png"))
    assert len(figures) == 7
    assert all(path.stat().st_size > 10_000 for path in figures)
    notebook = json.loads((ROOT / "notebooks/analysis_report.ipynb").read_text(encoding="utf-8"))
    outputs = [output for cell in notebook["cells"] for output in cell.get("outputs", [])]
    assert outputs
    assert not any(output.get("output_type") == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 6
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [cell["execution_count"] for cell in code_cells] == list(range(1, 7))
    exploration = "".join("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                          for cell in code_cells)
    assert all(token in exploration for token in ["products.info()", "products.describe", ".head(3)"])
    markdown = "".join("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                       for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    assert all(term in markdown for term in ["구조 탐색 해석", "통계·두 변수 쌍의 상관 해석",
                                             "가격–거래행 수", "그룹별 특징과 시사점"])
    non_image_outputs = []
    for output in outputs:
        safe = {key: value for key, value in output.items() if key != "data"}
        if "data" in output:
            safe["data"] = {key: value for key, value in output["data"].items() if key != "image/png"}
        non_image_outputs.append(safe)
    assert not re.search(r"\b[0-9a-f]{64}\b", json.dumps(non_image_outputs))


def test_raw_or_customer_level_data_is_not_tracked_in_git():
    repo_root = pathlib.Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()).resolve()
    data_relative = (ROOT / "data").resolve().relative_to(repo_root)
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(data_relative)], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert tracked == []

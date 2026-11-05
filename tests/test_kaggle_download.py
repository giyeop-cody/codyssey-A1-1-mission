import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fetch_kaggle_original", ROOT / "scripts/fetch_kaggle_original.py")
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)
RUN_ALL_SPEC = importlib.util.spec_from_file_location("run_all", ROOT / "scripts/run_all.py")
run_all = importlib.util.module_from_spec(RUN_ALL_SPEC)
RUN_ALL_SPEC.loader.exec_module(run_all)


class KaggleDownloadTests(unittest.TestCase):
    def test_default_cli_is_csv_only(self):
        args = fetch.build_parser().parse_args([])
        self.assertEqual(fetch.COMPETITION, "h-and-m-personalized-fashion-recommendations")
        self.assertFalse(args.include_images)
        self.assertFalse(args.force)
        self.assertEqual(fetch.build_login_command("/usr/bin/kaggle"),
                         ["/usr/bin/kaggle", "auth", "login", "--force"])
        command = fetch.build_download_command("/usr/bin/kaggle", Path("/tmp/download"))
        self.assertEqual(command[1:4], ["competitions", "download", fetch.COMPETITION])
        self.assertNotIn("-c", command)
        with mock.patch.dict(fetch.os.environ, {"KAGGLE_API_TOKEN": "secret", "KAGGLE_KEY": "secret"}), \
                mock.patch.object(fetch.subprocess, "run") as run:
            fetch.run_kaggle(["kaggle", "--version"])
        environment = run.call_args.kwargs["env"]
        self.assertNotIn("KAGGLE_API_TOKEN", environment)
        self.assertNotIn("KAGGLE_KEY", environment)
        self.assertNotIn("secret", str(run.call_args.args))

    def test_zip_slip_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../outside.txt", "blocked")
            with self.assertRaisesRegex(ValueError, "unsafe zip member path"):
                fetch.extract_zip(archive, root / "out")
            self.assertFalse((root / "outside.txt").exists())

    def test_csv_download_and_manifest_with_mocked_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fake_cli(args):
                calls.append(args)
                if args[1:] == ["auth", "login", "--force"]:
                    return
                filename = args[args.index("-f") + 1]
                with zipfile.ZipFile(root / "_downloads" / f"{filename}.zip", "w") as zf:
                    zf.writestr(filename, "id\n1\n")

            with mock.patch.object(fetch.shutil, "which", return_value="/usr/bin/kaggle"), \
                    mock.patch.object(fetch, "run_kaggle", side_effect=fake_cli):
                downloaded = fetch.fetch_csv_files(root, force=False, keep_zip=False, test_zip=True)
                fetch.write_manifest(root, downloaded, include_images=False, test_zip=True)
            self.assertEqual({path.name for path in downloaded},
                             {"articles.csv", "customers.csv", "transactions_train.csv"})
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][1:], ["auth", "login", "--force"])
            manifest = json.loads((root / "kaggle_original_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["include_images"])
            self.assertIn("browser OAuth", manifest["authentication"])
            self.assertEqual(manifest["image_count"], 0)
            for path in downloaded:
                self.assertEqual(len(manifest["files"][path.name]["sha256"]), 64)
                self.assertTrue(path.exists())

    def test_full_bundle_safe_extract_with_mocked_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fake_cli(args):
                calls.append(args)
                if args[1:] == ["auth", "login", "--force"]:
                    return
                self.assertEqual(args[3], fetch.COMPETITION)
                self.assertNotIn("-c", args)
                self.assertNotIn("-f", args)
                archive = root / "_downloads" / "h-and-m-personalized-fashion-recommendations.zip"
                with zipfile.ZipFile(archive, "w") as zf:
                    for filename in fetch.CSV_FILES:
                        zf.writestr(filename, "id\n1\n")
                    for index in range(1000):
                        zf.writestr(f"images/010/{index:010d}.jpg", b"not-a-real-jpeg-but-path-is-tested")

            with mock.patch.object(fetch.shutil, "which", return_value="/usr/bin/kaggle"), \
                    mock.patch.object(fetch, "run_kaggle", side_effect=fake_cli):
                downloaded = fetch.fetch_full_bundle(root, force=False, keep_zip=False, test_zip=True)
                fetch.write_manifest(root, downloaded, include_images=True, test_zip=True)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1:], ["auth", "login", "--force"])
            self.assertTrue((root / "images/010/0000000000.jpg").exists())
            self.assertFalse((root / "_downloads/h-and-m-personalized-fashion-recommendations.zip").exists())
            manifest = json.loads((root / "kaggle_original_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["include_images"])
            self.assertEqual(manifest["image_count"], 1000)
            self.assertEqual({path.name for path in downloaded}, set(fetch.CSV_FILES))

    def test_run_all_mirror_also_runs_pytest(self):
        with mock.patch.object(run_all, "execute") as execute, \
                mock.patch.object(run_all.subprocess, "run") as run:
            run_all.main(["--download-mirror"])
        self.assertEqual([call.args[0] for call in execute.call_args_list], [
            "prepare_hm_data.py", "run_analysis.py", "spec_curve.py",
            "build_notebook.py", "check_requirements.py",
        ])
        run.assert_called_once_with([run_all.sys.executable, "-m", "pytest", "-q"],
                                    cwd=run_all.ROOT, check=True)

    def test_run_all_original_downloads_before_preparation_and_runs_pytest(self):
        with mock.patch.object(run_all, "execute") as execute, \
                mock.patch.object(run_all.subprocess, "run") as run:
            run_all.main(["--download-original", "--n-customers", "1000"])
        scripts = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(scripts, [
            "fetch_kaggle_original.py", "prepare_hm_data.py", "run_analysis.py",
            "spec_curve.py", "build_notebook.py", "check_requirements.py",
        ])
        self.assertIn("--include-images", execute.call_args_list[0].args)
        self.assertIn("--image-dir", execute.call_args_list[1].args)
        run.assert_called_once_with([run_all.sys.executable, "-m", "pytest", "-q"],
                                    cwd=run_all.ROOT, check=True)


if __name__ == "__main__":
    unittest.main()

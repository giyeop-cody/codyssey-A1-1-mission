.PHONY: demo test clean

demo:            ## GPU 없이 전 과정 데모(합성 음성) + 리포트
	python3 scripts/demo_plan.py

test:
	python3 -m pytest -q

clean:
	rm -rf .pytest_cache **/__pycache__ artifacts/demo/audio

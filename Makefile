.PHONY: setup start test analyze

setup:
	pip install -r requirements.txt

start:
	python3 app.py

test:
	python3 -m pytest test_app.py -q

analyze:
	python3 analyze.py

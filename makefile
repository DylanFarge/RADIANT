.DEFAULT_GOAL := default
.PHONY: build-venv dependencies run dev test test-view test-all monkey monkey-analysis monkey-downloads default

build-venv:
	python3 -m venv env
dependencies:
	pip install -r requirements.txt
run:
	python3 src/app.py 0
debug:
	python3 src/app.py 1
dev:
	python3 src/app.py devTools $(filter-out $@,$(MAKECMDGOALS))
test:
	python3 -m coverage run --data-file=src/tests/.coverage --omit=/usr/* -m unittest discover src/tests/files
	python3 -m coverage html --data-file=src/tests/.coverage -d src/tests/html
	python3 -m coverage report --data-file=src/tests/.coverage
test-view:
	python3 -m coverage report --data-file=src/tests/.coverage
	open src/tests/html/index.html
test-all:
	python3 -m coverage run --data-file=src/tests/.coverage --omit=/usr/* -m unittest discover src/tests/files
	python3 -m coverage html --data-file=src/tests/.coverage -d src/tests/html
	python3 -m coverage report --data-file=src/tests/.coverage
	open src/tests/html/index.html
monkey:
	python3 src/tests/monkey.py
monkey-analysis:
	python3 src/tests/monkey.py analysis
monkey-downloads:
	python3 src/tests/monkey.py downloads
monkeys:
	python3 src/tests/monkey.py
	python3 src/tests/monkey.py analysis
	python3 src/tests/monkey.py downloads
scalability:
	python3 src/tests/scalability_analysis.py
default:
	@echo "--------------<OPTIONS>---------------------------------------------------------"
	@#echo "'build-venv'		: builds a virtual environment."
	@#echo "'dependencies'		: downloads all dependencies required for the app to function."
	@echo "'run'		: starts up the app on localhost to be views in a browser."
	@echo "'debug'		: starts up the app on localhost to be views in a browser with debug mode on."
	@echo "'dev'		: Runs developer utils without starting the app."
	@#echo "'test'			: runs all coverage tests and generates a coverage report."
	@#echo "'test-view'		: views the coverage report in a browser."
	@#echo "'test-all'		: runs all coverage tests and generates a coverage report, then views it."
	@#echo "'monkey'		: runs the monkey test on the Home page."
	@#echo "'monkey-analysis'	: runs the monkey test on the Analysis page."
	@#echo "'monkey-downloads'	: runs the monkey test on the Downloads page."	
	@#echo "'monkeys'		: runs all monkey tests."
	@#echo "'scalability-analysis'	: runs the scalability tests."
	@echo "--------------------------------------------------------------------------------"
	@# @echo "AUTO START APP . . ."
	@# @echo "--------------------------------------------------------------------------------"
	@# @python3 src/app.py

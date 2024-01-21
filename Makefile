export PIPENV_VENV_IN_PROJECT := 1
export PIPENV_VERBOSITY := -1

environment:
	@echo "Building Python environment"
	python3 -m pip install --upgrade pip
	pip3 install --upgrade pipenv
	pipenv install --python 3.10
	pipenv run pip install -r requirements.txt
	$(shell mkdir -p Model)

stop_docker:
	@echo "To stop all running docker containers run"
	@echo "You need to type the next command in a terminal"
	@echo "docker stop $(docker ps -a -q)"

clean:
	@echo "Cleaning"
	pipenv --rm

deactivate_environment:
	deactivate
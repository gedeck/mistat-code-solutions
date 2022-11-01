SRC=src
RUN=docker run -it --rm -v $(PWD):/code -v $(PWD)/src:/src 

jupyter-m: 
	@ $(RUN) -p 8895:8895 mistat.m.jupyter jupyter notebook --allow-root --port=8895 --ip 0.0.0.0 --no-browser

bash-m:
	@ $(RUN) mistat.m.python  bash

# tests:
# 	@ $(RUN) super.python pytest -p no:cacheprovider

# watch-tests:
# 	rm -f .testmondata
# 	@ $(RUN) supersar.python ptw --runner "pytest -o cache_dir=/tmp --testmon --quiet -rP"


# Docker container
images:
	docker build -t mistat.m.python -f docker/Dockerfile.m.python .
	docker build -t mistat.m.jupyter -f docker/Dockerfile.m.jupyter .

# github-pages:
# # 	# docker run -it --rm -v $(PWD):/usr/src/app -p "4000:4000" starefossen/github-pages
# # 	docker run --rm -v $(PWD):/srv/jekyll -p 4000:4000 -it jekyll/jekyll:builder bash
# 	# -e "INPUT_SOURCE" -e "INPUT_DESTINATION" -e "INPUT_FUTURE" -e "INPUT_BUILD_REVISION" -e "INPUT_VERBOSE" -e "INPUT_TOKEN" 
# 	# -e "HOME" -e "GITHUB_JOB" -e "GITHUB_REF" -e "GITHUB_SHA" -e "GITHUB_REPOSITORY" -e "GITHUB_REPOSITORY_OWNER" -e "GITHUB_RUN_ID" 
# 	# -e "GITHUB_RUN_NUMBER" -e "GITHUB_RETENTION_DAYS" -e "GITHUB_RUN_ATTEMPT" -e "GITHUB_ACTOR" -e "GITHUB_TRIGGERING_ACTOR" 
# 	# -e "GITHUB_WORKFLOW" -e "GITHUB_HEAD_REF" -e "GITHUB_BASE_REF" -e "GITHUB_EVENT_NAME" -e "GITHUB_SERVER_URL" -e "GITHUB_API_URL" 
# 	# -e "GITHUB_GRAPHQL_URL" -e "GITHUB_REF_NAME" -e "GITHUB_REF_PROTECTED" -e "GITHUB_REF_TYPE" -e "GITHUB_WORKSPACE" -e "GITHUB_ACTION" 
# 	# -e "GITHUB_EVENT_PATH" -e "GITHUB_ACTION_REPOSITORY" -e "GITHUB_ACTION_REF" -e "GITHUB_PATH" -e "GITHUB_ENV" -e "GITHUB_STEP_SUMMARY" 
# 	# -e "GITHUB_STATE" -e "GITHUB_OUTPUT" -e "RUNNER_OS" -e "RUNNER_ARCH" -e "RUNNER_NAME" -e "RUNNER_TOOL_CACHE" -e "RUNNER_TEMP" 
# 	# -e "RUNNER_WORKSPACE" -e "ACTIONS_RUNTIME_URL" -e "ACTIONS_RUNTIME_TOKEN" -e "ACTIONS_CACHE_URL" -e "ACTIONS_ID_TOKEN_REQUEST_URL" 
# 	# -e "ACTIONS_ID_TOKEN_REQUEST_TOKEN" -e GITHUB_ACTIONS=true -e CI=true 
# 	# -v "/var/run/docker.sock":"/var/run/docker.sock" 
# 	# -v "/home/runner/work/_temp/_github_home":"/github/home" 
# 	# -v "/home/runner/work/_temp/_github_workflow":"/github/workflow" 
# 	# -v "/home/runner/work/_temp/_runner_file_commands":"/github/file_commands" 
# 	docker run --workdir /github/workspace --rm \
# 	-v $(PWD):"/github/workspace" ghcr.io/actions/jekyll-build-pages:v1.0.4

# # linter and mypy
# MYPY_IMAGE=mypy
# PYLINT_IMAGE=supersar-pylint
# ISORT_IMAGE=supersar-isort

# mypy:
# 	docker build -t $(MYPY_IMAGE) -f docker/Dockerfile.mypy .
# 	@ docker run -it --rm -v $(PWD)/src:/src:ro $(MYPY_IMAGE) cdd

# pylint:
# 	docker build -t $(PYLINT_IMAGE) -f docker/Dockerfile.pylint .
# 	@ docker run -it --rm -v $(PWD)/src:/src:ro $(PYLINT_IMAGE) cdd
# 	@ docker run -it --rm -v $(PWD):/src:ro $(PYLINT_IMAGE) bin

# isort:
# 	docker build -t $(ISORT_IMAGE) -f docker/Dockerfile.isort .
# 	@ docker run -it --rm -v $(PWD)/src:/src $(ISORT_IMAGE) cdd
# 	@ docker run -it --rm -v $(PWD):/src $(ISORT_IMAGE) bin

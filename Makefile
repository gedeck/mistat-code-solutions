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
# 	# docker run -it --rm -v $(PWD):/usr/src/app -p "4000:4000" starefossen/github-pages
# 	docker run --rm -v $(PWD):/srv/jekyll -p 4000:4000 -it jekyll/jekyll:builder bash

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

.PHONY: test
DOCKER=docker
OPTSRUN?= -it -d
IMAGENAME?=interfacesolver
PYTHON=python3
ALLIMAGES=$(shell eval sudo docker ps -q --filter ancestor=$(IMAGENAME))
PATH?=/home/examples/turekhron/Result/
all:
	$(DOCKER) image build -t $(IMAGENAME) .
	$(DOCKER) run -it $(IMAGENAME) bash

runbash:
	$(DOCKER) run -it $(IMAGENAME) bash

buildimage:
	$(DOCKER) image build -t $(IMAGENAME) .

runimage:
	@echo New docker image:
	@$(DOCKER) run $(OPTSRUN) $(IMAGENAME)

dockerdeploy:
	@if [ -z "$(ALLIMAGES)" ] ; then \
		echo you need to run docker first: make runimage; \
	else \
		$(DOCKER) exec -it $(firstword $(ALLIMAGES)) ls && cd examples && python3 nonlinear_parabolic.py; \
	fi;

stopimage:
	@if [ -z "$(ALLIMAGES)" ] ; then \
		echo Already empty; \
	else \
		echo stopping $(ALLIMAGES); \
		$(DOCKER) stop $(ALLIMAGES); \
		$(DOCKER) rm $(ALLIMAGES); \
	fi;


copyresult:
	if [ -z "$(ALLIMAGES)" ] ; then \
		echo No image running; \
	else \
		$(DOCKER) cp $(firstword $(ALLIMAGES)):$(PATH) ./; \
	fi;
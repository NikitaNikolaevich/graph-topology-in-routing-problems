# Makefile for Sphinx documentation

SPHINXBUILD = sphinx-build
SPHINXOPTS = -q
SPHINXBUILDOPTS = -b html
SOURCEDIR = source
BUILDDIR = build

.PHONY: help clean html

help:
	@echo "Makefile for Sphinx documentation"
	@echo "Usage:"
	@echo "  make clean    - remove all build files"
	@echo "  make html     - build HTML documentation"

clean:
	rm -rf $(BUILDDIR)/*

html:
	python3 generate_rst.py
	$(SPHINXBUILD) $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)
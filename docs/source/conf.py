# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

project = 'ride'
copyright = '2024, Nikita Zakharenko'
author = 'Nikita Zakharenko'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
       'sphinx.ext.autodoc',  # Убедитесь, что это расширение включено
       'sphinx.ext.viewcode',
       'sphinx.ext.napoleon',  # Для поддержки Google и NumPy стиля docstrings
   ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_show_sphinx = False
html_static_path = ['_static']
html_css_files = ['_static/custom.css']
html_js_files = ['_static/custom.js']

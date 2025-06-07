# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import xarray_model

project = 'xarray-model'
copyright = '2025, Mike Blackett'
author = 'Mike Blackett'
release = xarray_model.__version__
version = xarray_model.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

source_suffix = ['.rst', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_baseurl = os.environ.get('READTHEDOCS_CANONICAL_URL', '/')
html_static_path = ['_static']
html_title = 'xarray-model'
html_theme = 'pydata_sphinx_theme'

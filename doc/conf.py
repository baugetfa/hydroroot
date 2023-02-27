# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('../src'))

# Get the project root dir, which is the parent dir of this
cwd = os.getcwd()
project_root = os.path.dirname(cwd)

# Insert the project root dir as the first element in the PYTHONPATH.
# This lets us ensure that the source package is imported, and that its
# version is used.
sys.path.insert(0, os.path.join(project_root, 'src'))


# -- Project information -----------------------------------------------------

project = 'HydroRoot'
copyright = '2022, Yann Boursiac, Christophe Pradal, Fabrice Bauget, Mikaël Lucas, Christophe Godin, Christophe Maurel'
author = 'Christophe Pradal, Yann Boursiac, Fabrice Bauget, Mikael Lucas, Christophe Godin, Christophe Maurel'

# The full version, including alpha/beta/rc tags
# release = '1.0.0'
# find version number in /src/$pkg_pth/version.py
pkgver = {}
with open("../src/openalea/mtg/version.py") as fp:
    exec(fp.read(), pkgver)

# The short X.Y version.
version = pkgver["__version__"]
# The full version, including alpha/beta/rc tags.
release = pkgver["__version__"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'openalea.hydroroot'
copyright = u'2022, openalea.hydroroot'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

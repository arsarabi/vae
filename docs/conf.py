# Documentation build configuration file

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc'
]

# See https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# Paths that contain templates
templates_path = ['templates']

# Generate stub pages for autosummary directives
autosummary_generate = True

# Suffix for source files
source_suffix = '.rst'

# The master toctree document
master_doc = 'index'

# General information about the project
project = 'vae'
copyright = '2018, Armin Sarabi'
author = 'Armin Sarabi'

# Version information for the project
version = '0.1.0'
release = '0.1.0'

# Patterns for directories to ignore when looking for source files
exclude_patterns = ['build']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -----------------------------------------------------------------------------
# Options for HTML output
# -----------------------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# -*- coding: utf-8 -*-

# Project information
project = '多模态大模型技术原理和实战'
author = 'Xuyang Cao'
copyright = '2026, Xuyang Cao'

# Extensions
extensions = [
    'myst_parser',           # 支持 Markdown
    'sphinx.ext.mathjax',    # 数学公式支持
    'sphinx_copybutton',     # 代码复制按钮
]

# Markdown support
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Master document
master_doc = 'index'

# MyST-Parser settings
myst_enable_extensions = [
    'dollarmath',      # $...$ 和 $$...$$ 数学公式
    'amsmath',         # 支持 LaTeX amsmath 环境 (align, equation 等)
    'colon_fence',     # ::: 代码块
    'deflist',         # 定义列表
    'tasklist',        # 任务列表
]
myst_heading_anchors = 3  # 为标题生成锚点

# dollarmath 配置：允许行内公式前后无空格
myst_dmath_double_inline = True

# MathJax 配置
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': True,
        'processEnvironments': True,
        'packages': {'[+]': ['ams', 'newcommand', 'configmacros']},
    },
    'options': {
        'ignoreHtmlClass': 'tex2jax_ignore',
        'processHtmlClass': 'tex2jax_process',
    },
}

# Theme
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

# Language
language = 'zh_CN'

# Static files
html_static_path = ['assets', '_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# LaTeX engine for Unicode support
latex_engine = 'xelatex'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

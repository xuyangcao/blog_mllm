# -*- coding: utf-8 -*-

# Project information
project = '多模态与大模型技术原理'
author = 'xuyang'
copyright = '2024, xuyang'

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
    'colon_fence',     # ::: 代码块
    'deflist',         # 定义列表
    'tasklist',        # 任务列表
]
myst_heading_anchors = 3  # 为标题生成锚点

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
html_static_path = ['assets']

# LaTeX engine for Unicode support
latex_engine = 'xelatex'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

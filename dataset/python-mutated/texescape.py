"""TeX escaping helper."""
from __future__ import annotations
import re
tex_replacements = [('$', '\\$'), ('%', '\\%'), ('&', '\\&'), ('#', '\\#'), ('_', '\\_'), ('{', '\\{'), ('}', '\\}'), ('\\', '\\textbackslash{}'), ('~', '\\textasciitilde{}'), ('^', '\\textasciicircum{}'), ('[', '{[}'), (']', '{]}'), ('✓', '\\(\\checkmark\\)'), ('✔', '\\(\\pmb{\\checkmark}\\)'), ('✕', '\\(\\times\\)'), ('✖', '\\(\\pmb{\\times}\\)'), ('\ufeff', '{}'), ('⎽', '\\_'), ('ℯ', 'e'), ('ⅈ', 'i')]
ascii_tex_replacements = [('-', '\\sphinxhyphen{}'), ("'", '\\textquotesingle{}'), ('`', '\\textasciigrave{}'), ('<', '\\textless{}'), ('>', '\\textgreater{}')]
unicode_tex_replacements = [('¶', '\\P{}'), ('§', '\\S{}'), ('€', '\\texteuro{}'), ('∞', '\\(\\infty\\)'), ('±', '\\(\\pm\\)'), ('→', '\\(\\rightarrow\\)'), ('‣', '\\(\\rightarrow\\)'), ('–', '\\textendash{}'), ('⁰', '\\(\\sp{\\text{0}}\\)'), ('¹', '\\(\\sp{\\text{1}}\\)'), ('²', '\\(\\sp{\\text{2}}\\)'), ('³', '\\(\\sp{\\text{3}}\\)'), ('⁴', '\\(\\sp{\\text{4}}\\)'), ('⁵', '\\(\\sp{\\text{5}}\\)'), ('⁶', '\\(\\sp{\\text{6}}\\)'), ('⁷', '\\(\\sp{\\text{7}}\\)'), ('⁸', '\\(\\sp{\\text{8}}\\)'), ('⁹', '\\(\\sp{\\text{9}}\\)'), ('₀', '\\(\\sb{\\text{0}}\\)'), ('₁', '\\(\\sb{\\text{1}}\\)'), ('₂', '\\(\\sb{\\text{2}}\\)'), ('₃', '\\(\\sb{\\text{3}}\\)'), ('₄', '\\(\\sb{\\text{4}}\\)'), ('₅', '\\(\\sb{\\text{5}}\\)'), ('₆', '\\(\\sb{\\text{6}}\\)'), ('₇', '\\(\\sb{\\text{7}}\\)'), ('₈', '\\(\\sb{\\text{8}}\\)'), ('₉', '\\(\\sb{\\text{9}}\\)')]
tex_replace_map: dict[int, str] = {}
_tex_escape_map: dict[int, str] = {}
_tex_escape_map_without_unicode: dict[int, str] = {}
_tex_hlescape_map: dict[int, str] = {}
_tex_hlescape_map_without_unicode: dict[int, str] = {}

def escape(s: str, latex_engine: str | None=None) -> str:
    if False:
        i = 10
        return i + 15
    'Escape text for LaTeX output.'
    if latex_engine in ('lualatex', 'xelatex'):
        return s.translate(_tex_escape_map_without_unicode)
    else:
        return s.translate(_tex_escape_map)

def hlescape(s: str, latex_engine: str | None=None) -> str:
    if False:
        while True:
            i = 10
    'Escape text for LaTeX highlighter.'
    if latex_engine in ('lualatex', 'xelatex'):
        return s.translate(_tex_hlescape_map_without_unicode)
    else:
        return s.translate(_tex_hlescape_map)

def escape_abbr(text: str) -> str:
    if False:
        i = 10
        return i + 15
    'Adjust spacing after abbreviations. Works with @ letter or other.'
    return re.sub('\\.(?=\\s|$)', '.\\@{}', text)

def init() -> None:
    if False:
        i = 10
        return i + 15
    for (a, b) in tex_replacements:
        _tex_escape_map[ord(a)] = b
        _tex_escape_map_without_unicode[ord(a)] = b
        tex_replace_map[ord(a)] = '_'
    for (a, b) in ascii_tex_replacements:
        _tex_escape_map[ord(a)] = b
    _tex_escape_map_without_unicode[ord('-')] = '\\sphinxhyphen{}'
    for (a, b) in unicode_tex_replacements:
        _tex_escape_map[ord(a)] = b
        tex_replace_map[ord(a)] = '_'
    for (a, b) in tex_replacements:
        if a in '[]{}\\':
            continue
        _tex_hlescape_map[ord(a)] = b
        _tex_hlescape_map_without_unicode[ord(a)] = b
    for (a, b) in unicode_tex_replacements:
        _tex_hlescape_map[ord(a)] = b
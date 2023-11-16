"""Test the LaTeX writer"""
import pytest
from sphinx.writers.latex import rstdim_to_latexdim

def test_rstdim_to_latexdim():
    if False:
        return 10
    assert rstdim_to_latexdim('160em') == '160em'
    assert rstdim_to_latexdim('160px') == '160\\sphinxpxdimen'
    assert rstdim_to_latexdim('160in') == '160in'
    assert rstdim_to_latexdim('160cm') == '160cm'
    assert rstdim_to_latexdim('160mm') == '160mm'
    assert rstdim_to_latexdim('160pt') == '160bp'
    assert rstdim_to_latexdim('160pc') == '160pc'
    assert rstdim_to_latexdim('30%') == '0.300\\linewidth'
    assert rstdim_to_latexdim('160') == '160\\sphinxpxdimen'
    assert rstdim_to_latexdim('160.0em') == '160.0em'
    assert rstdim_to_latexdim('.5em') == '.5em'
    with pytest.raises(ValueError, match='could not convert string to float: '):
        rstdim_to_latexdim('unknown')
    assert rstdim_to_latexdim('160.0unknown') == '160.0unknown'
from __future__ import annotations
import pytest
pytest
from bokeh.models import Circle
import bokeh.plotting._renderer as bpr

class Test__pop_visuals:

    def test_basic_prop(self) -> None:
        if False:
            print('Hello World!')
        kwargs = dict(fill_alpha=0.7, line_alpha=0.8, line_color='red')
        ca = bpr.pop_visuals(Circle, kwargs)
        assert ca['fill_alpha'] == 0.7
        assert ca['line_alpha'] == 0.8
        assert ca['line_color'] == 'red'
        assert ca['fill_color'] == '#1f77b4'
        assert set(ca) == {'fill_color', 'hatch_color', 'line_color', 'fill_alpha', 'hatch_alpha', 'line_alpha'}

    def test_basic_trait(self) -> None:
        if False:
            print('Hello World!')
        kwargs = dict(fill_alpha=0.7, alpha=0.8, color='red')
        ca = bpr.pop_visuals(Circle, kwargs)
        assert ca['fill_alpha'] == 0.7
        assert ca['line_alpha'] == 0.8
        assert ca['line_color'] == 'red'
        assert ca['fill_color'] == 'red'
        assert set(ca) == {'fill_color', 'hatch_color', 'line_color', 'fill_alpha', 'hatch_alpha', 'line_alpha'}

    def test_override_defaults_with_prefix(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        glyph_kwargs = dict(fill_alpha=1, line_alpha=1)
        kwargs = dict(alpha=0.6)
        ca = bpr.pop_visuals(Circle, kwargs, prefix='nonselection_', defaults=glyph_kwargs, override_defaults={'alpha': 0.1})
        assert ca['fill_alpha'] == 0.1
        assert ca['hatch_alpha'] == 0.1
        assert ca['line_alpha'] == 0.1

    def test_defaults(self) -> None:
        if False:
            while True:
                i = 10
        kwargs = dict(fill_alpha=0.7, line_alpha=0.8, line_color='red')
        ca = bpr.pop_visuals(Circle, kwargs, defaults=dict(line_color='blue', fill_color='green'))
        assert ca['fill_alpha'] == 0.7
        assert ca['line_alpha'] == 0.8
        assert ca['line_color'] == 'red'
        assert ca['fill_color'] == 'green'
        assert set(ca) == {'fill_color', 'hatch_color', 'line_color', 'fill_alpha', 'hatch_alpha', 'line_alpha'}

    def test_override_defaults(self) -> None:
        if False:
            return 10
        kwargs = dict(fill_alpha=0.7, line_alpha=0.8)
        ca = bpr.pop_visuals(Circle, kwargs, defaults=dict(line_color='blue', fill_color='green'), override_defaults=dict(color='white'))
        assert ca['fill_alpha'] == 0.7
        assert ca['line_alpha'] == 0.8
        assert ca['line_color'] == 'white'
        assert ca['fill_color'] == 'white'
        assert set(ca) == {'fill_color', 'hatch_color', 'line_color', 'fill_alpha', 'hatch_alpha', 'line_alpha'}

class Test_make_glyph:

    def test_null_visuals(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        kwargs = dict(fill_alpha=0.7, line_alpha=0.8, line_color='red')
        hover_visuals = None
        ca = bpr.make_glyph(Circle, kwargs, hover_visuals)
        assert ca is None

    def test_default_mute_glyph_basic_prop(self) -> None:
        if False:
            while True:
                i = 10
        kwargs = dict(fill_alpha=0.7, line_alpha=0.8, line_color='red')
        glyph_visuals = bpr.pop_visuals(Circle, kwargs)
        muted_visuals = bpr.pop_visuals(Circle, kwargs, prefix='muted_', defaults=glyph_visuals, override_defaults={'alpha': 0.2})
        ca = bpr.make_glyph(Circle, kwargs, muted_visuals)
        assert ca.fill_alpha == 0.2
        assert ca.line_alpha == 0.2
        assert isinstance(ca, Circle)

    def test_user_specified_mute_glyph(self) -> None:
        if False:
            return 10
        kwargs = dict(fill_alpha=0.7, line_alpha=0.8, line_color='red', muted_color='blue', muted_alpha=0.4)
        glyph_visuals = bpr.pop_visuals(Circle, kwargs)
        muted_visuals = bpr.pop_visuals(Circle, kwargs, prefix='muted_', defaults=glyph_visuals, override_defaults={'alpha': 0.2})
        ca = bpr.make_glyph(Circle, kwargs, muted_visuals)
        assert ca.fill_alpha == 0.4
        assert ca.line_alpha == 0.4
        assert ca.line_color == 'blue'
        assert ca.fill_color == 'blue'
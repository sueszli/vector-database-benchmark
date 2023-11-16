from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from minijinja import Environment
from litestar.contrib.minijinja import MiniJinjaTemplateEngine
from litestar.exceptions import ImproperlyConfiguredException, TemplateNotFoundException
if TYPE_CHECKING:
    from pathlib import Path

def test_mini_jinja_template_engine_instantiation_error(tmp_path: Path) -> None:
    if False:
        return 10
    with pytest.raises(ImproperlyConfiguredException):
        MiniJinjaTemplateEngine(directory=tmp_path, engine_instance=Environment())
    with pytest.raises(ImproperlyConfiguredException):
        MiniJinjaTemplateEngine()

def test_mini_jinja_template_engine_instantiated_with_engine() -> None:
    if False:
        for i in range(10):
            print('nop')
    engine = Environment()
    template_engine = MiniJinjaTemplateEngine(engine_instance=engine)
    assert template_engine.engine is engine

def test_mini_jinja_template_render_raises_template_not_found(tmp_path: Path) -> None:
    if False:
        while True:
            i = 10
    template_engine = MiniJinjaTemplateEngine(engine_instance=Environment())
    with pytest.raises(TemplateNotFoundException):
        tmpl = template_engine.get_template('not_found.html')
        tmpl.render()

def test_from_environment() -> None:
    if False:
        return 10
    engine = Environment()
    template_engine = MiniJinjaTemplateEngine.from_environment(engine)
    assert template_engine.engine is engine
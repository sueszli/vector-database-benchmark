from pathlib import Path
from typing import Any, Dict
from litestar import Litestar, get
from litestar.contrib.jinja import JinjaTemplateEngine
from litestar.response import Template
from litestar.template.config import TemplateConfig

def my_template_function(ctx: Dict[str, Any]) -> str:
    if False:
        return 10
    return ctx.get('my_context_key', 'nope')

def register_template_callables(engine: JinjaTemplateEngine) -> None:
    if False:
        i = 10
        return i + 15
    engine.register_template_callable(key='check_context_key', template_callable=my_template_function)
template_config = TemplateConfig(directory=Path(__file__).parent / 'templates', engine=JinjaTemplateEngine, engine_callback=register_template_callables)

@get('/', sync_to_thread=False)
def index() -> Template:
    if False:
        while True:
            i = 10
    return Template(template_name='index.html.jinja2')
app = Litestar(route_handlers=[index], template_config=template_config)
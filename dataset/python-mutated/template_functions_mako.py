from pathlib import Path
from typing import Any, Dict
from litestar import Litestar, get
from litestar.contrib.mako import MakoTemplateEngine
from litestar.response import Template
from litestar.template.config import TemplateConfig

def my_template_function(ctx: Dict[str, Any]) -> str:
    if False:
        i = 10
        return i + 15
    return ctx.get('my_context_key', 'nope')

def register_template_callables(engine: MakoTemplateEngine) -> None:
    if False:
        for i in range(10):
            print('nop')
    engine.register_template_callable(key='check_context_key', template_callable=my_template_function)
template_config = TemplateConfig(directory=Path(__file__).parent / 'templates', engine=MakoTemplateEngine, engine_callback=register_template_callables)

@get('/', sync_to_thread=False)
def index() -> Template:
    if False:
        for i in range(10):
            print('nop')
    return Template(template_name='index.html.mako')
app = Litestar(route_handlers=[index], template_config=template_config)
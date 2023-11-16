"""
Jinja2 rendering utils, used to generate new strategy and configurations.
"""

def render_template(templatefile: str, arguments: dict={}) -> str:
    if False:
        for i in range(10):
            print('nop')
    from jinja2 import Environment, PackageLoader, select_autoescape
    env = Environment(loader=PackageLoader('freqtrade', 'templates'), autoescape=select_autoescape(['html', 'xml']))
    template = env.get_template(templatefile)
    return template.render(**arguments)

def render_template_with_fallback(templatefile: str, templatefallbackfile: str, arguments: dict={}) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Use templatefile if possible, otherwise fall back to templatefallbackfile\n    '
    from jinja2.exceptions import TemplateNotFound
    try:
        return render_template(templatefile, arguments)
    except TemplateNotFound:
        return render_template(templatefallbackfile, arguments)
"""Utils for rendering and updating package descriptions (READMEs)."""
from email.message import EmailMessage
from importlib.metadata import distribution
import readme_renderer.markdown
import readme_renderer.rst
import readme_renderer.txt
_RENDERERS = {None: readme_renderer.rst, '': readme_renderer.rst, 'text/plain': readme_renderer.txt, 'text/x-rst': readme_renderer.rst, 'text/markdown': readme_renderer.markdown}

def render(value, content_type=None, use_fallback=True):
    if False:
        for i in range(10):
            print('nop')
    if value is None:
        return value
    if content_type:
        msg = EmailMessage()
        msg['content-type'] = content_type
        content_type = msg.get_content_type()
    renderer = _RENDERERS.get(content_type, readme_renderer.txt)
    rendered = renderer.render(value)
    if content_type == 'text/plain':
        rendered = f'<pre>{rendered}</pre>'
    if use_fallback and rendered is None:
        rendered = readme_renderer.txt.render(value)
    return rendered

def renderer_version():
    if False:
        for i in range(10):
            print('nop')
    return distribution('readme-renderer').version
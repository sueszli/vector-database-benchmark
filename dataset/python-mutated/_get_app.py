from textwrap import dedent
APP = None

def get_app():
    if False:
        i = 10
        return i + 15
    if APP is None:
        raise Exception(dedent('\n                App object is not yet defined.  `app = dash.Dash()` needs to be run\n                before `dash.get_app()` is called and can only be used within apps that use\n                the `pages` multi-page app feature: `dash.Dash(use_pages=True)`.\n\n                `dash.get_app()` is used to get around circular import issues when Python files\n                within the pages/` folder need to reference the `app` object.\n                '))
    return APP
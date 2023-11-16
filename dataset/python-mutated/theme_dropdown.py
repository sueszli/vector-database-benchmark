import os
import pathlib
from gradio.themes.utils import ThemeAsset

def create_theme_dropdown():
    if False:
        while True:
            i = 10
    import gradio as gr
    asset_path = pathlib.Path() / 'themes'
    themes = []
    for theme_asset in os.listdir(str(asset_path)):
        themes.append((ThemeAsset(theme_asset), gr.Theme.load(str(asset_path / theme_asset))))

    def make_else_if(theme_asset):
        if False:
            while True:
                i = 10
        return f"\n        else if (theme == '{str(theme_asset[0].version)}') {{\n            var theme_css = `{theme_asset[1]._get_theme_css()}`\n        }}"
    (head, tail) = (themes[0], themes[1:])
    if_statement = f'''\n        if (theme == "{str(head[0].version)}") {{\n            var theme_css = `{head[1]._get_theme_css()}`\n        }} {' '.join((make_else_if(t) for t in tail))}\n    '''
    latest_to_oldest = sorted([t[0] for t in themes], key=lambda asset: asset.version)[::-1]
    latest_to_oldest = [str(t.version) for t in latest_to_oldest]
    component = gr.Dropdown(choices=latest_to_oldest, value=latest_to_oldest[0], render=False, label='Select Version').style(container=False)
    return (component, f"\n        (theme) => {{\n            if (!document.querySelector('.theme-css')) {{\n                var theme_elem = document.createElement('style');\n                theme_elem.classList.add('theme-css');\n                document.head.appendChild(theme_elem);\n            }} else {{\n                var theme_elem = document.querySelector('.theme-css');\n            }}\n            {if_statement}\n            theme_elem.innerHTML = theme_css;\n        }}\n    ")
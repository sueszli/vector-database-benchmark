import os
import re
import shutil
from typing import List, Optional, Tuple
from lightning.app.utilities.app_helpers import Logger
logger = Logger(__name__)

def app(app_name: str) -> None:
    if False:
        return 10
    if app_name is None:
        app_name = _capture_valid_app_component_name(resource_type='app')
    (new_resource_name, name_for_files) = _make_resource(resource_dir='app-template', resource_name=app_name)
    m = f'\n    ⚡ Lightning app template created! ⚡\n    {new_resource_name}\n\n    run your app with:\n        lightning run app {app_name}/app.py\n\n    run it on the cloud to share with your collaborators:\n        lightning run app {app_name}/app.py --cloud\n    '
    logger.info(m)

def _make_resource(resource_dir: str, resource_name: str) -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    path = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(path, resource_dir)
    name_for_files = re.sub('-', '_', resource_name)
    new_resource_name = os.path.join(os.getcwd(), resource_name)
    logger.info(f'laying out component template at {new_resource_name}')
    shutil.copytree(template_dir, new_resource_name)
    os.rename(os.path.join(new_resource_name, 'placeholdername'), os.path.join(new_resource_name, name_for_files))
    trouble_names = {'.DS_Store'}
    files = _ls_recursively(new_resource_name)
    for bad_file in files:
        if bad_file.split('/')[-1] in trouble_names:
            continue
        with open(bad_file) as fo:
            content = fo.read().replace('placeholdername', name_for_files)
        with open(bad_file, 'w') as fw:
            fw.write(content)
    for file_name in files:
        new_file = re.sub('placeholdername', name_for_files, file_name)
        os.rename(file_name, new_file)
    return (new_resource_name, name_for_files)

def _ls_recursively(dir_name: str) -> List[str]:
    if False:
        return 10
    fname = []
    for (root, d_names, f_names) in os.walk(dir_name):
        for f in f_names:
            if '__pycache__' not in root:
                fname.append(os.path.join(root, f))
    return fname

def _capture_valid_app_component_name(value: Optional[str]=None, resource_type: str='app') -> str:
    if False:
        print('Hello World!')
    prompt = f'\n    ⚡ Creating Lightning {resource_type} ⚡\n    '
    logger.info(prompt)
    try:
        if value is None:
            value = input(f'\nName your Lightning {resource_type} (example: the-{resource_type}-name) >  ')
        value = value.strip().lower()
        unsafe_chars = set(re.findall('[^a-z0-9\\-]', value))
        if len(unsafe_chars) > 0:
            m = f"\n            Error: your Lightning {resource_type} name:\n            {value}\n\n            contains the following unsupported characters:\n            {unsafe_chars}\n\n            A Lightning {resource_type} name can only contain letters (a-z) numbers (0-9) and the '-' character\n\n            valid example:\n            lightning-{resource_type}\n            "
            raise SystemExit(m)
    except KeyboardInterrupt:
        raise SystemExit(f'\n        ⚡ {resource_type} init aborted! ⚡\n        ')
    return value

def component(component_name: str) -> None:
    if False:
        while True:
            i = 10
    if component_name is None:
        component_name = _capture_valid_app_component_name(resource_type='component')
    (new_resource_name, name_for_files) = _make_resource(resource_dir='component-template', resource_name=component_name)
    m = f"\n    ⚡ Lightning component template created! ⚡\n    {new_resource_name}\n\n    ⚡ To use your component, first pip install it (with these 3 commands): ⚡\n    cd {component_name}\n    pip install -r requirements.txt\n    pip install -e .\n\n    ⚡ Use the component inside an app: ⚡\n\n    from {name_for_files} import TemplateComponent\n    import lightning.app as la\n\n    class LitApp(la.LightningFlow):\n        def __init__(self) -> None:\n            super().__init__()\n            self.{name_for_files} = TemplateComponent()\n\n        def run(self):\n            print('this is a simple Lightning app to verify your component is working as expected')\n            self.{name_for_files}.run()\n\n    app = la.LightningApp(LitApp())\n\n    ⚡ Checkout the demo app with your {component_name} component: ⚡\n    lightning run app {component_name}/app.py\n\n    ⚡ Tip: Publish your component to the Lightning Gallery to enable users to install it like so:\n    lightning install component YourLightningUserName/{component_name}\n\n    so the Lightning community can use it like:\n    from {name_for_files} import TemplateComponent\n\n    "
    logger.info(m)
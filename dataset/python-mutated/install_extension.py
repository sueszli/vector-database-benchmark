import os
import jupyterlab_jupytext
from jupyterlab.federated_labextensions import build_labextension, develop_labextension

def main():
    if False:
        return 10
    'Create symlink in sys.prefix based on name of extension'
    labexts = jupyterlab_jupytext._jupyter_labextension_paths()
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jupyterlab_jupytext')
    for labext in labexts:
        src = os.path.join(base_path, labext['src'])
        dest = labext['dest']
        print(f'Installing {src} -> {dest}')
        if not os.path.exists(src):
            build_labextension(base_path)
        full_dest = develop_labextension(src, overwrite=True, symlink=True, user=False, sys_prefix=True, labextensions_dir='', destination=dest)
        print(f'Creating symlink at {full_dest}')
if __name__ == '__main__':
    main()
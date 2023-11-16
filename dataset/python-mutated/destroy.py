from dvc.ignore import destroy as destroy_dvcignore
from dvc.utils.fs import remove
from . import locked

@locked
def _destroy_stages(repo):
    if False:
        return 10
    for stage in repo.index.stages:
        stage.unprotect_outs()
        stage.dvcfile.remove(force=True)

def destroy(repo):
    if False:
        i = 10
        return i + 15
    _destroy_stages(repo)
    repo.close()
    destroy_dvcignore(repo.root_dir)
    remove(repo.dvc_dir)
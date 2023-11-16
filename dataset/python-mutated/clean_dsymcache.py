from sentry.bgtasks.api import bgtask
from sentry.models.debugfile import ProjectDebugFile

@bgtask()
def clean_dsymcache():
    if False:
        for i in range(10):
            print('nop')
    ProjectDebugFile.difcache.clear_old_entries()
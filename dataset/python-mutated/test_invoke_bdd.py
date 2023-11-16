import pytest_bdd as bdd
bdd.scenarios('invoke.feature')

@bdd.when(bdd.parsers.parse('I spawn a new window'))
def invoke_with(quteproc):
    if False:
        while True:
            i = 10
    'Spawn a new window via IPC call.'
    quteproc.log_summary('Create a new window')
    quteproc.send_ipc([], target_arg='window')
    quteproc.wait_for(category='init', module='app', function='_open_startpage', message='Opening start pages')
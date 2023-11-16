import os.path
import pytest_bdd as bdd
bdd.scenarios('javascript.feature')

@bdd.then('the window sizes should be the same')
def check_window_sizes(quteproc):
    if False:
        for i in range(10):
            print('nop')
    hidden = quteproc.wait_for_js('hidden window size: *')
    quteproc.send_cmd(':jseval --world main updateText("visible")')
    visible = quteproc.wait_for_js('visible window size: *')
    hidden_size = hidden.message.split()[-1]
    visible_size = visible.message.split()[-1]
    assert hidden_size == visible_size
test_gm_script = '\n// ==UserScript==\n// @name qutebrowser test userscript\n// @namespace invalid.org\n// @include http://localhost:*/data/hints/iframe.html\n// @include http://localhost:*/data/hints/html/wrapped.html\n// @exclude ???\n// @run-at {stage}\n// {frames}\n// ==/UserScript==\nconsole.log("Script is running on " + window.location.pathname);\n'

@bdd.when(bdd.parsers.parse('I have a GreaseMonkey file saved for {stage} with noframes {frameset}'))
def create_greasemonkey_file(quteproc, stage, frameset):
    if False:
        i = 10
        return i + 15
    script_path = os.path.join(quteproc.basedir, 'data', 'greasemonkey')
    try:
        os.mkdir(script_path)
    except FileExistsError:
        pass
    file_path = os.path.join(script_path, 'test.user.js')
    if frameset == 'set':
        frames = '@noframes'
    elif frameset == 'unset':
        frames = ''
    else:
        raise ValueError('noframes can only be set or unset, not {}'.format(frameset))
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(test_gm_script.format(stage=stage, frames=frames))
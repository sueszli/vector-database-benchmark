"""Creating hard links for tutorials in each individual topics."""
import os
import re
HEADER = '.. THIS FILE IS A COPY OF {} WITH MODIFICATIONS.\n.. TO MAKE ONE TUTORIAL APPEAR IN MULTIPLE PLACES.\n\n'

def flatten_filename(filename):
    if False:
        print('Hello World!')
    return filename.replace('/', '_').replace('.', '_')

def copy_tutorials(app):
    if False:
        print('Hello World!')
    print('[tutorial links] copy tutorials...')
    for (src, tar) in app.config.tutorials_copy_list:
        target_path = os.path.join(app.srcdir, tar)
        content = open(os.path.join(app.srcdir, src)).read()
        content = HEADER.format(src) + content
        label_map = {}
        for (prefix, label_name) in list(re.findall('(\\.\\.\\s*_)(.*?)\\:\\s*\\n', content)):
            label_map[label_name] = flatten_filename(tar) + '_' + label_name
            content = content.replace(prefix + label_name + ':', prefix + label_map[label_name] + ':')
            content = content.replace(f':ref:`{label_name}`', f':ref:`{label_map[label_name]}')
            content = re.sub('(\\:ref\\:`.*?\\<)' + label_name + '(\\>`)', '\\1' + label_map[label_name] + '\\2', content)
        open(target_path, 'w').write(content)

def setup(app):
    if False:
        i = 10
        return i + 15
    app.connect('builder-inited', copy_tutorials)
    app.add_config_value('tutorials_copy_list', [], True, [list])
'''
This module renders highstate configuration into a more human readable format.

How it works:

`highstate or lowstate` data is parsed with a `processor` this defaults to `highstate_doc.processor_markdown`.
The processed data is passed to a `jinja` template that builds up the document content.


configuration: Pillar

.. code-block:: none

    # the following defaults can be overridden
    highstate_doc.config:

        # list of regex of state names to ignore in `highstate_doc.process_lowstates`
        filter_id_regex:
            - '.*!doc_skip$'

        # list of regex of state functions to ignore in `highstate_doc.process_lowstates`
        filter_state_function_regex:
            - 'file.accumulated'

        # dict of regex to replace text after `highstate_doc.render`. (remove passwords)
        text_replace_regex:
            'password:.*^': '[PASSWORD]'

        # limit size of files that can be included in doc (10000 bytes)
        max_render_file_size: 10000

        # advanced option to set a custom lowstate processor
        processor: highstate_doc.processor_markdown


State example

.. code-block:: yaml

    {{sls}} note:
        highstate_doc.note:
            - name: example
            - order: 0
            - contents: |
                example `highstate_doc.note`
                ------------------
                This state does not do anything to the system! It is only used by a `processor`
                you can use `requisites` and `order` to move your docs around the rendered file.

    {{sls}} a file we don't want in the doc !doc_skip:
        file.managed:
            - name: /root/passwords
            - contents: 'password: sadefgq34y45h56q'
            # also could use `highstate_doc.config: text_replace_regex` to replace
            # password string. `password:.*^': '[PASSWORD]`


To create the help document build a State that uses `highstate_doc.render`.
For performance it's advised to not included this state in your `top.sls` file.

.. code-block:: yaml

    # example `salt://makereadme.sls`
    make helpfile:
        file.managed:
            - name: /root/README.md
            - contents: {{salt.highstate_doc.render()|json}}
            - show_diff: {{opts['test']}}
            - mode: '0640'
            - order: last

Run our `makereadme.sls` state to create `/root/README.md`.

.. code-block:: bash

    # first ensure `highstate` return without errors or changes
    salt-call state.highstate
    salt-call state.apply makereadme
    # or if you don't want the extra `make helpfile` state
    salt-call --out=newline_values_only salt.highstate_doc.render > /root/README.md ; chmod 0600 /root/README.md


Creating a document collection
------------------------------

From the master we can run the following script to
creates a collection of all your minion documents.

.. code-block:: bash

    salt '*' state.apply makereadme

.. code-block:: python

    #!/bin/python
    import os
    import salt.client
    s = salt.client.LocalClient()
    # NOTE: because of issues with `cp.push` use `highstate_doc.read_file`
    o = s.cmd('*', 'highstate_doc.read_file', ['/root/README.md'])
    for m in o:
        d = o.get(m)
        if d and not d.endswith('is not available.'):
            # mkdir m
            #directory = os.path.dirname(file_path)
            if not os.path.exists(m):
                os.makedirs(m)
            with open(m + '/README.md','wb') as f:
                f.write(d)
            print('ADDED: ' + m + '/README.md')


Once the master has a collection of all the README files.
You can use pandoc to create HTML versions of the markdown.

.. code-block:: bash

    # process all the readme.md files to readme.html
    if which pandoc; then echo "Found pandoc"; else echo "** Missing pandoc"; exit 1; fi
    if which gs; then echo "Found gs"; else echo "** Missing gs(ghostscript)"; exit 1; fi
    readme_files=$(find $dest -type f -path "*/README.md" -print)
    for f in $readme_files ; do
        ff=${f#$dest/}
        minion=${ff%%/*}
        echo "process: $dest/${minion}/$(basename $f)"
        cat $dest/${minion}/$(basename $f) |             pandoc --standalone --from markdown_github --to html             --include-in-header $dest/style.html             > $dest/${minion}/$(basename $f).html
    done

It is also nice to put the help files in source control.

    # git init
    git add -A
    git commit -am 'updated docs'
    git push -f


Other hints
-----------

If you wish to customize the document format:

.. code-block:: none

    # you could also create a new `processor` for perhaps reStructuredText
    # highstate_doc.config:
    #     processor: doc_custom.processor_rst

    # example `salt://makereadme.jinja`
    """
    {{opts['id']}}
    ==========================================

    {# lowstates is set from highstate_doc.render() #}
    {# if lowstates is missing use salt.highstate_doc.process_lowstates() #}
    {% for s in lowstates %}
    {{s.id}}
    -----------------------------------------------------------------
    {{s.function}}

    {{s.markdown.requisite}}
    {{s.markdown.details}}

    {%- endfor %}
    """

    # example `salt://makereadme.sls`
    {% import_text "makereadme.jinja" as makereadme %}
    {{sls}} or:
        file.managed:
            - name: /root/README_other.md
            - contents: {{salt.highstate_doc.render(jinja_template_text=makereadme)|json}}
            - mode: '0640'


Some `replace_text_regex` values that might be helpful::

    CERTS
    -----

    ``'-----BEGIN RSA PRIVATE KEY-----[\\r\\n\\t\\f\\S]{0,2200}': 'XXXXXXX'``
    ``'-----BEGIN CERTIFICATE-----[\\r\\n\\t\\f\\S]{0,2200}': 'XXXXXXX'``
    ``'-----BEGIN DH PARAMETERS-----[\\r\\n\\t\\f\\S]{0,2200}': 'XXXXXXX'``
    ``'-----BEGIN PRIVATE KEY-----[\\r\\n\\t\\f\\S]{0,2200}': 'XXXXXXX'``
    ``'-----BEGIN OPENSSH PRIVATE KEY-----[\\r\\n\\t\\f\\S]{0,2200}': 'XXXXXXX'``
    ``'ssh-rsa .* ': 'ssh-rsa XXXXXXX '``
    ``'ssh-dss .* ': 'ssh-dss XXXXXXX '``

    DB
    --

    ``'DB_PASS.*': 'DB_PASS = XXXXXXX'``
    ``'5432:*:*:.*': '5432:*:XXXXXXX'``
    ``"'PASSWORD': .*": "'PASSWORD': 'XXXXXXX',"``
    ``" PASSWORD '.*'": " PASSWORD 'XXXXXXX'"``
    ``'PGPASSWORD=.* ': 'PGPASSWORD=XXXXXXX'``
    ``"_replication password '.*'":  "_replication password 'XXXXXXX'"``

    OTHER
    -----

    ``'EMAIL_HOST_PASSWORD =.*': 'EMAIL_HOST_PASSWORD =XXXXXXX'``
    ``"net ads join -U '.*@MFCFADS.MATH.EXAMPLE.CA.* ": "net ads join -U '.*@MFCFADS.MATH.EXAMPLE.CA%XXXXXXX "``
    ``"net ads join -U '.*@NEXUS.EXAMPLE.CA.* ": "net ads join -U '.*@NEXUS.EXAMPLE.CA%XXXXXXX "``
    ``'install-uptrack .* --autoinstall': 'install-uptrack XXXXXXX --autoinstall'``
    ``'accesskey = .*': 'accesskey = XXXXXXX'``
    ``'auth_pass .*': 'auth_pass XXXXXXX'``
    ``'PSK "0x.*': 'PSK "0xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'``
    ``'SECRET_KEY.*': 'SECRET_KEY = XXXXXXX'``
    ``"password=.*": "password=XXXXXXX"``
    ``'<password>.*</password>': '<password>XXXXXXX</password>'``
    ``'<salt>.*</salt>': '<salt>XXXXXXX</salt>'``
    ``'application.secret = ".*"': 'application.secret = "XXXXXXX"'``
    ``'url = "postgres://.*"': 'url = "postgres://XXXXXXX"'``
    ``'PASS_.*_PASS': 'PASS_XXXXXXX_PASS'``

    HTACCESS
    --------

    ``':{PLAIN}.*': ':{PLAIN}XXXXXXX'``

'''
import logging
import re
import salt.utils.files
import salt.utils.stringutils
import salt.utils.templates as tpl
import salt.utils.yaml
__virtualname__ = 'highstate_doc'
log = logging.getLogger(__name__)
markdown_basic_jinja_template_txt = '\n{% for s in lowstates %}\n`{{s.id_full}}`\n-----------------------------------------------------------------\n * state: {{s.state_function}}\n * name: `{{s.name}}`\n\n{{s.markdown.requisites}}\n{{s.markdown.details}}\n\n{%- endfor %}\n'
markdown_default_jinja_template_txt = "\nConfiguration Managment\n===============================================================================\n\n```\n####################################################\nfqdn: {{grains.get('fqdn')}}\nos: {{grains.get('os')}}\nosfinger: {{grains.get('osfinger')}}\nmem_total: {{grains.get('mem_total')}}MB\nnum_cpus: {{grains.get('num_cpus')}}\nipv4: {{grains.get('ipv4')}}\nmaster: {{opts.get('master')}}\n####################################################\n```\n\nThis system is fully or partly managed using Salt.\n\nThe following sections are a rendered view of what the configuration management system\ncontrolled on this system. Each item is handled in order from top to bottom unless some\nrequisites like `require` force other ordering.\n\n" + markdown_basic_jinja_template_txt
markdown_advanced_jinja_template_txt = markdown_default_jinja_template_txt + '\n\n{% if vars.get(\'doc_other\', True) -%}\nOther information\n=====================================================================================\n\n```\n\nsalt grain: ip_interfaces\n-----------------------------------------------------------------\n{{grains[\'ip_interfaces\']|dictsort}}\n\n\nsalt grain: hwaddr_interfaces\n-----------------------------------------------------------------\n{{grains[\'hwaddr_interfaces\']|dictsort}}\n\n{% if not grains[\'os\'] == \'Windows\' %}\n\n{% if salt[\'cmd.has_exec\'](\'ip\') -%}\n# ip address show\n-----------------------------------------------------------------\n{{salt[\'cmd.run\'](\'ip address show | sed "/valid_lft/d"\')}}\n\n\n# ip route list table all\n-----------------------------------------------------------------\n{{salt[\'cmd.run\'](\'ip route list table all\')}}\n{% endif %}\n\n{% if salt[\'cmd.has_exec\'](\'iptables\') %}\n{%- if salt[\'cmd.has_exec\'](\'iptables-save\') -%}\n# iptables-save\n-----------------------------------------------------------------\n{{salt[\'cmd.run\']("iptables --list > /dev/null; iptables-save | \\grep -v -F \'#\' | sed \'/^:/s@\\[[0-9]\\{1,\\}:[0-9]\\{1,\\}\\]@[0:0]@g\'")}}\n\n\n# ip6tables-save\n-----------------------------------------------------------------\n{{salt[\'cmd.run\']("ip6tables --list > /dev/null; ip6tables-save | \\grep -v -F \'#\' | sed \'/^:/s@\\[[0-9]\\{1,\\}:[0-9]\\{1,\\}\\]@[0:0]@g\'")}}\n{%- else -%}\n# iptables --list-rules\n-----------------------------------------------------------------\n{{salt[\'cmd.run\'](\'iptables --list-rules\')}}\n\n\n# ip6tables --list-rules\n-----------------------------------------------------------------\n{{salt[\'cmd.run\'](\'ip6tables --list-rules\')}}\n{% endif %}\n{% endif %}\n\n{% if salt[\'cmd.has_exec\'](\'firewall-cmd\') -%}\n# firewall-cmd --list-all\n-----------------------------------------------------------------\n{{salt[\'cmd.run\'](\'firewall-cmd --list-all\')}}\n{% endif %}\n\n# mount\n-----------------------------------------------------------------\n{{salt[\'cmd.run\'](\'mount\')}}\n\n{% endif %}\n'

def markdown_basic_jinja_template(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return text for a simple markdown jinja template\n\n    This function can be used from the `highstate_doc.render` modules `jinja_template_function` option.\n    '
    return markdown_basic_jinja_template_txt

def markdown_default_jinja_template(**kwargs):
    if False:
        while True:
            i = 10
    '\n    Return text for a markdown jinja template that included a header\n\n    This function can be used from the `highstate_doc.render` modules `jinja_template_function` option.\n    '
    return markdown_default_jinja_template_txt

def markdown_full_jinja_template(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return text for an advanced markdown jinja template\n\n    This function can be used from the `highstate_doc.render` modules `jinja_template_function` option.\n    '
    return markdown_advanced_jinja_template_txt

def _get_config(**kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Return configuration\n    '
    config = {'filter_id_regex': ['.*!doc_skip'], 'filter_function_regex': [], 'replace_text_regex': {}, 'processor': 'highstate_doc.processor_markdown', 'max_render_file_size': 10000, 'note': None}
    if '__salt__' in globals():
        config_key = '{}.config'.format(__virtualname__)
        config.update(__salt__['config.get'](config_key, {}))
    for k in set(config.keys()) & set(kwargs.keys()):
        config[k] = kwargs[k]
    return config

def read_file(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    output the contents of a file:\n\n    this is a workaround if the cp.push module does not work.\n    https://github.com/saltstack/salt/issues/37133\n\n    help the master output the contents of a document\n    that might be saved on the minions filesystem.\n\n    .. code-block:: python\n\n        #!/bin/python\n        import os\n        import salt.client\n        s = salt.client.LocalClient()\n        o = s.cmd('*', 'highstate_doc.read_file', ['/root/README.md'])\n        for m in o:\n            d = o.get(m)\n            if d and not d.endswith('is not available.'):\n                # mkdir m\n                #directory = os.path.dirname(file_path)\n                if not os.path.exists(m):\n                    os.makedirs(m)\n                with open(m + '/README.md','wb') as fin:\n                    fin.write(d)\n                print('ADDED: ' + m + '/README.md')\n    "
    out = ''
    try:
        with salt.utils.files.fopen(name, 'r') as f:
            out = salt.utils.stringutils.to_unicode(f.read())
    except Exception as ex:
        log.error(ex)
        return None
    return out

def render(jinja_template_text=None, jinja_template_function='highstate_doc.markdown_default_jinja_template', **kwargs):
    if False:
        while True:
            i = 10
    '\n    Render highstate to a text format (default Markdown)\n\n    if `jinja_template_text` is not set, `jinja_template_function` is used.\n\n    jinja_template_text: jinja text that the render uses to create the document.\n    jinja_template_function: a salt module call that returns template text.\n\n    :options:\n        highstate_doc.markdown_basic_jinja_template\n        highstate_doc.markdown_default_jinja_template\n        highstate_doc.markdown_full_jinja_template\n\n    '
    config = _get_config(**kwargs)
    lowstates = process_lowstates(**kwargs)
    context = {'saltenv': None, 'config': config, 'lowstates': lowstates, 'salt': __salt__, 'pillar': __pillar__, 'grains': __grains__, 'opts': __opts__, 'kwargs': kwargs}
    template_text = jinja_template_text
    if template_text is None and jinja_template_function:
        template_text = __salt__[jinja_template_function](**kwargs)
    if template_text is None:
        raise Exception('No jinja template text')
    txt = tpl.render_jinja_tmpl(template_text, context, tmplpath=None)
    rt = config.get('replace_text_regex')
    for r in rt:
        txt = re.sub(r, rt[r], txt)
    return txt

def _blacklist_filter(s, config):
    if False:
        while True:
            i = 10
    ss = s['state']
    sf = s['fun']
    state_function = '{}.{}'.format(s['state'], s['fun'])
    for b in config['filter_function_regex']:
        if re.match(b, state_function):
            return True
    for b in config['filter_id_regex']:
        if re.match(b, s['__id__']):
            return True
    return False

def process_lowstates(**kwargs):
    if False:
        i = 10
        return i + 15
    '\n    return processed lowstate data that was not blacklisted\n\n    render_module_function is used to provide your own.\n    defaults to from_lowstate\n    '
    states = []
    config = _get_config(**kwargs)
    processor = config.get('processor')
    ls = __salt__['state.show_lowstate']()
    if not isinstance(ls, list):
        raise Exception('ERROR: to see details run: [salt-call state.show_lowstate] <-----***-SEE-***')
    elif ls:
        if not isinstance(ls[0], dict):
            raise Exception('ERROR: to see details run: [salt-call state.show_lowstate] <-----***-SEE-***')
    for s in ls:
        if _blacklist_filter(s, config):
            continue
        doc = __salt__[processor](s, config, **kwargs)
        states.append(doc)
    return states

def _state_data_to_yaml_string(data, whitelist=None, blacklist=None):
    if False:
        print('Hello World!')
    '\n    return a data dict in yaml string format.\n    '
    y = {}
    if blacklist is None:
        blacklist = ['__env__', '__id__', '__sls__', 'fun', 'name', 'context', 'order', 'state', 'require', 'require_in', 'watch', 'watch_in']
    kset = set(data.keys())
    if blacklist:
        kset -= set(blacklist)
    if whitelist:
        kset &= set(whitelist)
    for k in kset:
        y[k] = data[k]
    if not y:
        return None
    return salt.utils.yaml.safe_dump(y, default_flow_style=False)

def _md_fix(text):
    if False:
        while True:
            i = 10
    '\n    sanitize text data that is to be displayed in a markdown code block\n    '
    return text.replace('```', '``[`][markdown parse fix]')

def _format_markdown_system_file(filename, config):
    if False:
        while True:
            i = 10
    ret = ''
    file_stats = __salt__['file.stats'](filename)
    y = _state_data_to_yaml_string(file_stats, whitelist=['user', 'group', 'mode', 'uid', 'gid', 'size'])
    if y:
        ret += 'file stat {1}\n```\n{0}```\n'.format(y, filename)
    file_size = file_stats.get('size')
    if file_size <= config.get('max_render_file_size'):
        is_binary = True
        try:
            file_type = __salt__['cmd.shell']("\\file -i '{}'".format(filename))
            if 'charset=binary' not in file_type:
                is_binary = False
        except Exception as ex:
            is_binary = False
        if is_binary:
            file_data = '[[skipped binary data]]'
        else:
            with salt.utils.files.fopen(filename, 'r') as f:
                file_data = salt.utils.stringutils.to_unicode(f.read())
        file_data = _md_fix(file_data)
        ret += 'file data {1}\n```\n{0}\n```\n'.format(file_data, filename)
    else:
        ret += '```\n{}\n```\n'.format('SKIPPED LARGE FILE!\nSet {}:max_render_file_size > {} to render.'.format('{}.config'.format(__virtualname__), file_size))
    return ret

def _format_markdown_link(name):
    if False:
        print('Hello World!')
    link = name
    symbals = '~`!@#$%^&*()+={}[]:;"<>,.?/|\'\\'
    for s in symbals:
        link = link.replace(s, '')
    link = link.replace(' ', '-')
    return link

def _format_markdown_requisite(state, stateid, makelink=True):
    if False:
        while True:
            i = 10
    '\n    format requisite as a link users can click\n    '
    fmt_id = '{}: {}'.format(state, stateid)
    if makelink:
        return ' * [{}](#{})\n'.format(fmt_id, _format_markdown_link(fmt_id))
    else:
        return ' * `{}`\n'.format(fmt_id)

def processor_markdown(lowstate_item, config, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes low state data and returns a dict of processed data\n    that is by default used in a jinja template when rendering a markdown highstate_doc.\n\n    This `lowstate_item_markdown` given a lowstate item, returns a dict like:\n\n    .. code-block:: none\n\n        vars:       # the raw lowstate_item that was processed\n        id:         # the \'id\' of the state.\n        id_full:    # combo of the state type and id "state: id"\n        state:      # name of the salt state module\n        function:   # name of the state function\n        name:       # value of \'name:\' passed to the salt state module\n        state_function:    # the state name and function name\n        markdown:          # text data to describe a state\n            requisites:    # requisite like [watch_in, require_in]\n            details:       # state name, parameters and other details like file contents\n\n    '
    s = lowstate_item
    state_function = '{}.{}'.format(s['state'], s['fun'])
    id_full = '{}: {}'.format(s['state'], s['__id__'])
    requisites = ''
    for (comment, key) in (('run or update after changes in:\n', 'watch'), ('after changes, run or update:\n', 'watch_in'), ('require:\n', 'require'), ('required in:\n', 'require_in')):
        reqs = s.get(key, [])
        if reqs:
            requisites += comment
            for w in reqs:
                requisites += _format_markdown_requisite(*next(iter(w.items())))
    details = ''
    if state_function == 'highstate_doc.note':
        if 'contents' in s:
            details += '\n{}\n'.format(s['contents'])
        if 'source' in s:
            text = __salt__['cp.get_file_str'](s['source'])
            if text:
                details += '\n{}\n'.format(text)
            else:
                details += '\n{}\n'.format('ERROR: opening {}'.format(s['source']))
    if state_function == 'pkg.installed':
        pkgs = s.get('pkgs', s.get('name'))
        details += '\n```\ninstall: {}\n```\n'.format(pkgs)
    if state_function == 'file.recurse':
        details += 'recurse copy of files\n'
        y = _state_data_to_yaml_string(s)
        if y:
            details += '```\n{}\n```\n'.format(y)
        if '!doc_recurse' in id_full:
            findfiles = __salt__['file.find'](path=s.get('name'), type='f')
            if len(findfiles) < 10 or '!doc_recurse_force' in id_full:
                for f in findfiles:
                    details += _format_markdown_system_file(f, config)
            else:
                details += ' > Skipping because more than 10 files to display.\n'
                details += ' > HINT: to force include !doc_recurse_force in state id.\n'
        else:
            details += ' > For more details review logs and Salt state files.\n\n'
            details += ' > HINT: for improved docs use multiple file.managed states or file.archive, git.latest. etc.\n'
            details += ' > HINT: to force doc to show all files in path add !doc_recurse .\n'
    if state_function == 'file.blockreplace':
        if s.get('content'):
            details += 'ensure block of content is in file\n```\n{}\n```\n'.format(_md_fix(s['content']))
        if s.get('source'):
            text = '** source: ' + s.get('source')
            details += 'ensure block of content is in file\n```\n{}\n```\n'.format(_md_fix(text))
    if state_function == 'file.managed':
        details += _format_markdown_system_file(s['name'], config)
    if not details:
        y = _state_data_to_yaml_string(s)
        if y:
            details += '```\n{}```\n'.format(y)
    r = {'vars': lowstate_item, 'state': s['state'], 'name': s['name'], 'function': s['fun'], 'id': s['__id__'], 'id_full': id_full, 'state_function': state_function, 'markdown': {'requisites': requisites.decode('utf-8'), 'details': details.decode('utf-8')}}
    return r
"""
    ticket role
    ~~~~~~~~~~~

    An interpreted text role to link docs to tickets issues.

    :copyright: Copyright 2013 by Daniele Varrazzo.
"""
import re
from docutils import nodes, utils
from docutils.parsers.rst import roles

def ticket_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    if False:
        while True:
            i = 10
    cfg = inliner.document.settings.env.app.config
    if cfg.ticket_url is None:
        msg = inliner.reporter.warning('ticket not configured: please configure ticket_url in conf.py')
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    rv = [nodes.Text(name + ' ')]
    tokens = re.findall('(#?\\d+)|([^\\d#]+)', text)
    for (ticket, noise) in tokens:
        if ticket:
            num = int(ticket.replace('#', ''))
            remap_until = cfg.ticket_remap_until
            remap_offset = cfg.ticket_remap_offset
            if remap_until and remap_offset:
                if num <= remap_until:
                    num += remap_offset
            url = cfg.ticket_url % num
            roles.set_classes(options)
            node = nodes.reference(ticket, utils.unescape(ticket), refuri=url, **options)
            rv.append(node)
        else:
            assert noise
            rv.append(nodes.Text(noise))
    return (rv, [])

def setup(app):
    if False:
        i = 10
        return i + 15
    app.add_config_value('ticket_url', None, 'env')
    app.add_config_value('ticket_remap_until', None, 'env')
    app.add_config_value('ticket_remap_offset', None, 'env')
    app.add_role('ticket', ticket_role)
    app.add_role('tickets', ticket_role)
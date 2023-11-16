from __future__ import unicode_literals, division, absolute_import, print_function
from powerline.theme import requires_segment_info
from powerline.lib.dict import updated
from powerline.bindings.wm import get_i3_connection, get_connected_xrandr_outputs

@requires_segment_info
def output_lister(pl, segment_info):
    if False:
        return 10
    'List all outputs in segment_info format\n\t'
    return ((updated(segment_info, output=output['name']), {'draw_inner_divider': None}) for output in get_connected_xrandr_outputs(pl))

@requires_segment_info
def workspace_lister(pl, segment_info, only_show=None, output=None):
    if False:
        i = 10
        return i + 15
    'List all workspaces in segment_info format\n\n\tSets the segment info values of ``workspace`` and ``output`` to the name of\n\tthe i3 workspace and the ``xrandr`` output respectively and the keys\n\t``"visible"``, ``"urgent"`` and ``"focused"`` to a boolean indicating these\n\tstates.\n\n\t:param list only_show:\n\t\tSpecifies which workspaces to list. Valid entries are ``"visible"``,\n\t\t``"urgent"`` and ``"focused"``. If omitted or ``null`` all workspaces\n\t\tare listed.\n\n\t:param str output:\n\t\tMay be set to the name of an X output. If specified, only workspaces\n\t\ton that output are listed. Overrides automatic output detection by\n\t\tthe lemonbar renderer and bindings. Set to ``false`` to force\n\t\tall workspaces to be shown.\n\t'
    if output == None:
        output = output or segment_info.get('output')
    return ((updated(segment_info, output=w.output, workspace=w), {'draw_inner_divider': None}) for w in get_i3_connection().get_workspaces() if (not only_show or any((getattr(w, typ) for typ in only_show))) and (not output or w.output == output))
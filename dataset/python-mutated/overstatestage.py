"""
Display clean output of an overstate stage
==========================================

This outputter is used to display :ref:`Orchestrate Runner
<orchestrate-runner>` stages, and should not be called directly.
"""
import salt.utils.color

def output(data, **kwargs):
    if False:
        return 10
    '\n    Format the data for printing stage information from the overstate system\n    '
    colors = salt.utils.color.get_colors(__opts__.get('color'), __opts__.get('color_theme'))
    ostr = ''
    for comp in data:
        for (name, stage) in comp.items():
            ostr += '{}{}: {}\n'.format(colors['LIGHT_BLUE'], name, colors['ENDC'])
            for key in sorted(stage):
                ostr += '    {}{}: {}{}\n'.format(colors['LIGHT_BLUE'], key, stage[key], colors['ENDC'])
    return ostr
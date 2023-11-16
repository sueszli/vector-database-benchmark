from __future__ import unicode_literals, division, absolute_import, print_function
from copy import copy
from powerline.lib.unicode import unicode
DEFAULT_MODE_KEY = None
ATTR_BOLD = 1
ATTR_ITALIC = 2
ATTR_UNDERLINE = 4

def get_attrs_flag(attrs):
    if False:
        i = 10
        return i + 15
    'Convert an attribute array to a renderer flag.'
    attrs_flag = 0
    if 'bold' in attrs:
        attrs_flag |= ATTR_BOLD
    if 'italic' in attrs:
        attrs_flag |= ATTR_ITALIC
    if 'underline' in attrs:
        attrs_flag |= ATTR_UNDERLINE
    return attrs_flag

def pick_gradient_value(grad_list, gradient_level):
    if False:
        for i in range(10):
            print('nop')
    'Given a list of colors and gradient percent, return a color that should be used.\n\n\tNote: gradient level is not checked for being inside [0, 100] interval.\n\t'
    return grad_list[int(round(gradient_level * (len(grad_list) - 1) / 100))]

class Colorscheme(object):

    def __init__(self, colorscheme_config, colors_config):
        if False:
            return 10
        'Initialize a colorscheme.'
        self.colors = {}
        self.gradients = {}
        self.groups = colorscheme_config['groups']
        self.translations = colorscheme_config.get('mode_translations', {})
        for (color_name, color) in colors_config['colors'].items():
            try:
                self.colors[color_name] = (color[0], int(color[1], 16))
            except TypeError:
                self.colors[color_name] = (color, cterm_to_hex[color])
        for (gradient_name, gradient) in colors_config['gradients'].items():
            if len(gradient) == 2:
                self.gradients[gradient_name] = (gradient[0], [int(color, 16) for color in gradient[1]])
            else:
                self.gradients[gradient_name] = (gradient[0], [cterm_to_hex[color] for color in gradient[0]])

    def get_gradient(self, gradient, gradient_level):
        if False:
            return 10
        if gradient in self.gradients:
            return tuple((pick_gradient_value(grad_list, gradient_level) for grad_list in self.gradients[gradient]))
        else:
            return self.colors[gradient]

    def get_group_props(self, mode, trans, group, translate_colors=True):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(group, (str, unicode)):
            try:
                group_props = trans['groups'][group]
            except KeyError:
                try:
                    group_props = self.groups[group]
                except KeyError:
                    return None
                else:
                    return self.get_group_props(mode, trans, group_props, True)
            else:
                return self.get_group_props(mode, trans, group_props, False)
        elif translate_colors:
            group_props = copy(group)
            try:
                ctrans = trans['colors']
            except KeyError:
                pass
            else:
                for key in ('fg', 'bg'):
                    try:
                        group_props[key] = ctrans[group_props[key]]
                    except KeyError:
                        pass
            return group_props
        else:
            return group

    def get_highlighting(self, groups, mode, gradient_level=None):
        if False:
            print('Hello World!')
        trans = self.translations.get(mode, {})
        for group in groups:
            group_props = self.get_group_props(mode, trans, group)
            if group_props:
                break
        else:
            raise KeyError('Highlighting groups not found in colorscheme: ' + ', '.join(groups))
        if gradient_level is None:
            pick_color = self.colors.__getitem__
        else:
            pick_color = lambda gradient: self.get_gradient(gradient, gradient_level)
        return {'fg': pick_color(group_props['fg']), 'bg': pick_color(group_props['bg']), 'attrs': get_attrs_flag(group_props.get('attrs', []))}
cterm_to_hex = (0, 12582912, 32768, 8404992, 192, 12583104, 32896, 12632256, 8421504, 16736352, 65280, 16776960, 8421631, 16728319, 65535, 16777215, 0, 95, 135, 175, 215, 255, 24320, 24415, 24455, 24495, 24535, 24575, 34560, 34655, 34695, 34735, 34775, 34815, 44800, 44895, 44935, 44975, 45015, 45055, 55040, 55135, 55175, 55215, 55255, 55295, 65280, 65375, 65415, 65455, 65495, 65535, 6225920, 6226015, 6226055, 6226095, 6226135, 6226175, 6250240, 6250335, 6250375, 6250415, 6250455, 6250495, 6260480, 6260575, 6260615, 6260655, 6260695, 6260735, 6270720, 6270815, 6270855, 6270895, 6270935, 6270975, 6280960, 6281055, 6281095, 6281135, 6281175, 6281215, 6291200, 6291295, 6291335, 6291375, 6291415, 6291455, 8847360, 8847455, 8847495, 8847535, 8847575, 8847615, 8871680, 8871775, 8871815, 8871855, 8871895, 8871935, 8881920, 8882015, 8882055, 8882095, 8882135, 8882175, 8892160, 8892255, 8892295, 8892335, 8892375, 8892415, 8902400, 8902495, 8902535, 8902575, 8902615, 8902655, 8912640, 8912735, 8912775, 8912815, 8912855, 8912895, 11468800, 11468895, 11468935, 11468975, 11469015, 11469055, 11493120, 11493215, 11493255, 11493295, 11493335, 11493375, 11503360, 11503455, 11503495, 11503535, 11503575, 11503615, 11513600, 11513695, 11513735, 11513775, 11513815, 11513855, 11523840, 11523935, 11523975, 11524015, 11524055, 11524095, 11534080, 11534175, 11534215, 11534255, 11534295, 11534335, 14090240, 14090335, 14090375, 14090415, 14090455, 14090495, 14114560, 14114655, 14114695, 14114735, 14114775, 14114815, 14124800, 14124895, 14124935, 14124975, 14125015, 14125055, 14135040, 14135135, 14135175, 14135215, 14135255, 14135295, 14145280, 14145375, 14145415, 14145455, 14145495, 14145535, 14155520, 14155615, 14155655, 14155695, 14155735, 14155775, 16711680, 16711775, 16711815, 16711855, 16711895, 16711935, 16736000, 16736095, 16736135, 16736175, 16736215, 16736255, 16746240, 16746335, 16746375, 16746415, 16746455, 16746495, 16756480, 16756575, 16756615, 16756655, 16756695, 16756735, 16766720, 16766815, 16766855, 16766895, 16766935, 16766975, 16776960, 16777055, 16777095, 16777135, 16777175, 16777215, 526344, 1184274, 1842204, 2500134, 3158064, 3815994, 4473924, 5131854, 5789784, 6447714, 7105644, 7763574, 8421504, 9079434, 9737364, 10395294, 11053224, 11711154, 12369084, 13027014, 13684944, 14342874, 15000804, 15658734)
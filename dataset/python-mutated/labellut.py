from colorsys import rgb_to_yiq

class LabelLUT:
    """The class to manage look-up table for assigning colors to labels."""

    class Label:

        def __init__(self, name, value, color):
            if False:
                for i in range(10):
                    print('nop')
            self.name = name
            self.value = value
            self.color = color
    Colors = [[0.0, 0.0, 0.0], [0.96078431, 0.58823529, 0.39215686], [0.96078431, 0.90196078, 0.39215686], [0.58823529, 0.23529412, 0.11764706], [0.70588235, 0.11764706, 0.31372549], [1.0, 0.0, 0.0], [0.11764706, 0.11764706, 1.0], [0.78431373, 0.15686275, 1.0], [0.35294118, 0.11764706, 0.58823529], [1.0, 0.0, 1.0], [1.0, 0.58823529, 1.0], [0.29411765, 0.0, 0.29411765], [0.29411765, 0.0, 0.68627451], [0.0, 0.78431373, 1.0], [0.19607843, 0.47058824, 1.0], [0.0, 0.68627451, 0.0], [0.0, 0.23529412, 0.52941176], [0.31372549, 0.94117647, 0.58823529], [0.58823529, 0.94117647, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.25], [0.5, 1.0, 0.25], [0.25, 1.0, 0.25], [0.25, 1.0, 0.5], [0.25, 1.0, 1.25], [0.25, 0.5, 1.25], [0.25, 0.25, 1.0], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.375, 0.375, 0.375], [0.5, 0.5, 0.5], [0.625, 0.625, 0.625], [0.75, 0.75, 0.75], [0.875, 0.875, 0.875]]

    def __init__(self, label_to_names=None):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            label_to_names: Initialize the colormap with this mapping from\n                labels (int) to class names (str).\n        '
        self._next_color = 0
        self.labels = {}
        if label_to_names is not None:
            for val in sorted(label_to_names.keys()):
                self.add_label(label_to_names[val], val)

    def add_label(self, name, value, color=None):
        if False:
            i = 10
            return i + 15
        "Adds a label to the table.\n\n        Example:\n            The following sample creates a LUT with 3 labels::\n\n                lut = ml3d.vis.LabelLUT()\n                lut.add_label('one', 1)\n                lut.add_label('two', 2)\n                lut.add_label('three', 3, [0,0,1]) # use blue for label 'three'\n\n        Args:\n            name: The label name as string.\n            value: The value associated with the label.\n            color: Optional RGB color. E.g., [0.2, 0.4, 1.0].\n        "
        if color is None:
            if self._next_color >= len(self.Colors):
                color = [0.85, 1.0, 1.0]
            else:
                color = self.Colors[self._next_color]
                self._next_color += 1
        self.labels[value] = self.Label(name, value, color)

    @classmethod
    def get_colors(self, name='default', mode=None):
        if False:
            return 10
        'Return full list of colors in the lookup table.\n\n        Args:\n            name (str): Name of lookup table colormap. Only \'default\' is\n                supported.\n            mode (str): Colormap mode. May be None (return as is), \'lightbg" to\n                move the dark colors earlier in the list or \'darkbg\' to move\n                them later in the list. This will provide better visual\n                discrimination for the earlier classes.\n\n        Returns:\n            List of colors (R, G, B) in the LUT.\n        '
        if mode is None:
            return self.Colors
        dark_colors = list(filter(lambda col: rgb_to_yiq(*col)[0] < 0.5, self.Colors))
        light_colors = list(filter(lambda col: rgb_to_yiq(*col)[0] >= 0.5, self.Colors))
        if mode == 'lightbg':
            return dark_colors + light_colors
        if mode == 'darkbg':
            return light_colors + dark_colors
"""
Configuration for the order of gallery sections and examples.
Paths are relative to the conf.py file.
"""
from sphinx_gallery.sorting import ExplicitOrder
UNSORTED = 'unsorted'
examples_order = ['../galleries/examples/lines_bars_and_markers', '../galleries/examples/images_contours_and_fields', '../galleries/examples/subplots_axes_and_figures', '../galleries/examples/statistics', '../galleries/examples/pie_and_polar_charts', '../galleries/examples/text_labels_and_annotations', '../galleries/examples/color', '../galleries/examples/shapes_and_collections', '../galleries/examples/style_sheets', '../galleries/examples/pyplots', '../galleries/examples/axes_grid1', '../galleries/examples/axisartist', '../galleries/examples/showcase', UNSORTED, '../galleries/examples/userdemo']
tutorials_order = ['../galleries/tutorials/introductory', '../galleries/tutorials/intermediate', '../galleries/tutorials/advanced', UNSORTED, '../galleries/tutorials/provisional']
plot_types_order = ['../galleries/plot_types/basic', '../galleries/plot_types/stats', '../galleries/plot_types/arrays', '../galleries/plot_types/unstructured', '../galleries/plot_types/3D', UNSORTED]
folder_lists = [examples_order, tutorials_order, plot_types_order]
explicit_order_folders = [fd for folders in folder_lists for fd in folders[:folders.index(UNSORTED)]]
explicit_order_folders.append(UNSORTED)
explicit_order_folders.extend([fd for folders in folder_lists for fd in folders[folders.index(UNSORTED):]])

class MplExplicitOrder(ExplicitOrder):
    """For use within the 'subsection_order' key."""

    def __call__(self, item):
        if False:
            return 10
        'Return a string determining the sort order.'
        if item in self.ordered_list:
            return f'{self.ordered_list.index(item):04d}'
        else:
            return f'{self.ordered_list.index(UNSORTED):04d}{item}'
list_all = ['quick_start', 'pyplot', 'images', 'lifecycle', 'customizing', 'artists', 'legend_guide', 'color_cycle', 'constrainedlayout_guide', 'tight_layout_guide', 'text_intro', 'text_props', 'colors', 'color_demo', 'pie_features', 'pie_demo2', 'plot', 'scatter_plot', 'bar', 'stem', 'step', 'fill_between', 'imshow', 'pcolormesh', 'contour', 'contourf', 'barbs', 'quiver', 'streamplot', 'hist_plot', 'boxplot_plot', 'errorbar_plot', 'violin', 'eventplot', 'hist2d', 'hexbin', 'pie', 'tricontour', 'tricontourf', 'tripcolor', 'triplot', 'spines', 'spine_placement_demo', 'spines_dropped', 'multiple_yaxis_with_spines', 'centered_spines_with_arrows']
explicit_subsection_order = [item + '.py' for item in list_all]

class MplExplicitSubOrder:
    """For use within the 'within_subsection_order' key."""

    def __init__(self, src_dir):
        if False:
            i = 10
            return i + 15
        self.src_dir = src_dir
        self.ordered_list = explicit_subsection_order

    def __call__(self, item):
        if False:
            return 10
        'Return a string determining the sort order.'
        if item in self.ordered_list:
            return f'{self.ordered_list.index(item):04d}'
        else:
            return 'zzz' + item
sectionorder = MplExplicitOrder(explicit_order_folders)
subsectionorder = MplExplicitSubOrder
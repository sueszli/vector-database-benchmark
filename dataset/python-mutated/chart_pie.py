from warnings import warn
from . import chart

class ChartPie(chart.Chart):
    """
    A class for writing the Excel XLSX Pie charts.


    """

    def __init__(self, options=None):
        if False:
            while True:
                i = 10
        '\n        Constructor.\n\n        '
        super(ChartPie, self).__init__()
        self.vary_data_color = 1
        self.rotation = 0
        self.label_position_default = 'best_fit'
        self.label_positions = {'center': 'ctr', 'inside_end': 'inEnd', 'outside_end': 'outEnd', 'best_fit': 'bestFit'}

    def set_rotation(self, rotation):
        if False:
            return 10
        '\n        Set the Pie/Doughnut chart rotation: the angle of the first slice.\n\n        Args:\n            rotation: First segment angle: 0 <= rotation <= 360.\n\n        Returns:\n            Nothing.\n\n        '
        if rotation is None:
            return
        if rotation < 0 or rotation > 360:
            warn('Chart rotation %d outside Excel range: 0 <= rotation <= 360' % rotation)
            return
        self.rotation = int(rotation)

    def _write_chart_type(self, args):
        if False:
            i = 10
            return i + 15
        self._write_pie_chart(args)

    def _write_pie_chart(self, args):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('c:pieChart')
        self._write_vary_colors()
        for data in self.series:
            self._write_ser(data)
        self._write_first_slice_ang()
        self._xml_end_tag('c:pieChart')

    def _write_plot_area(self):
        if False:
            for i in range(10):
                print('nop')
        self._xml_start_tag('c:plotArea')
        self._write_layout(self.plotarea.get('layout'), 'plot')
        self._write_chart_type(None)
        second_chart = self.combined
        if second_chart:
            if second_chart.is_secondary:
                second_chart.id = 1000 + self.id
            else:
                second_chart.id = self.id
            second_chart.fh = self.fh
            second_chart.series_index = self.series_index
            second_chart._write_chart_type(None)
        self._write_sp_pr(self.plotarea)
        self._xml_end_tag('c:plotArea')

    def _write_legend(self):
        if False:
            for i in range(10):
                print('nop')
        legend = self.legend
        position = legend.get('position', 'right')
        font = legend.get('font')
        delete_series = []
        overlay = 0
        if legend.get('delete_series') and isinstance(legend['delete_series'], list):
            delete_series = legend['delete_series']
        if position.startswith('overlay_'):
            position = position.replace('overlay_', '')
            overlay = 1
        allowed = {'right': 'r', 'left': 'l', 'top': 't', 'bottom': 'b', 'top_right': 'tr'}
        if position == 'none':
            return
        if position not in allowed:
            return
        position = allowed[position]
        self._xml_start_tag('c:legend')
        self._write_legend_pos(position)
        for index in delete_series:
            self._write_legend_entry(index)
        self._write_layout(legend.get('layout'), 'legend')
        if overlay:
            self._write_overlay()
        self._write_sp_pr(legend)
        self._write_tx_pr_legend(None, font)
        self._xml_end_tag('c:legend')

    def _write_tx_pr_legend(self, horiz, font):
        if False:
            i = 10
            return i + 15
        if font and font.get('rotation'):
            rotation = font['rotation']
        else:
            rotation = None
        self._xml_start_tag('c:txPr')
        self._write_a_body_pr(rotation, horiz)
        self._write_a_lst_style()
        self._write_a_p_legend(font)
        self._xml_end_tag('c:txPr')

    def _write_a_p_legend(self, font):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('a:p')
        self._write_a_p_pr_legend(font)
        self._write_a_end_para_rpr()
        self._xml_end_tag('a:p')

    def _write_a_p_pr_legend(self, font):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('rtl', 0)]
        self._xml_start_tag('a:pPr', attributes)
        self._write_a_def_rpr(font)
        self._xml_end_tag('a:pPr')

    def _write_vary_colors(self):
        if False:
            return 10
        attributes = [('val', 1)]
        self._xml_empty_tag('c:varyColors', attributes)

    def _write_first_slice_ang(self):
        if False:
            i = 10
            return i + 15
        attributes = [('val', self.rotation)]
        self._xml_empty_tag('c:firstSliceAng', attributes)

    def _write_show_leader_lines(self):
        if False:
            print('Hello World!')
        attributes = [('val', 1)]
        self._xml_empty_tag('c:showLeaderLines', attributes)
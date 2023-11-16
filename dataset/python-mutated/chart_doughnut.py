from warnings import warn
from . import chart_pie

class ChartDoughnut(chart_pie.ChartPie):
    """
    A class for writing the Excel XLSX Doughnut charts.


    """

    def __init__(self, options=None):
        if False:
            while True:
                i = 10
        '\n        Constructor.\n\n        '
        super(ChartDoughnut, self).__init__()
        self.vary_data_color = 1
        self.rotation = 0
        self.hole_size = 50

    def set_hole_size(self, size):
        if False:
            return 10
        '\n        Set the Doughnut chart hole size.\n\n        Args:\n            size: 10 <= size <= 90.\n\n        Returns:\n            Nothing.\n\n        '
        if size is None:
            return
        if size < 10 or size > 90:
            warn('Chart hole size %d outside Excel range: 10 <= size <= 90' % size)
            return
        self.hole_size = int(size)

    def _write_chart_type(self, args):
        if False:
            print('Hello World!')
        self._write_doughnut_chart(args)

    def _write_doughnut_chart(self, args):
        if False:
            i = 10
            return i + 15
        self._xml_start_tag('c:doughnutChart')
        self._write_vary_colors()
        for data in self.series:
            self._write_ser(data)
        self._write_first_slice_ang()
        self._write_c_hole_size()
        self._xml_end_tag('c:doughnutChart')

    def _write_c_hole_size(self):
        if False:
            print('Hello World!')
        attributes = [('val', self.hole_size)]
        self._xml_empty_tag('c:holeSize', attributes)
import io
from django.http import HttpResponse
from django.views.generic import View
import xlsxwriter

def get_simple_table_data():
    if False:
        for i in range(10):
            print('nop')
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

class MyView(View):

    def get(self, request):
        if False:
            print('Hello World!')
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output)
        worksheet = workbook.add_worksheet()
        data = get_simple_table_data()
        for (row_num, columns) in enumerate(data):
            for (col_num, cell_data) in enumerate(columns):
                worksheet.write(row_num, col_num, cell_data)
        workbook.close()
        output.seek(0)
        filename = 'django_simple.xlsx'
        response = HttpResponse(output, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = 'attachment; filename=%s' % filename
        return response
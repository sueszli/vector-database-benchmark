import chartify

class TestBaseAxes:

    def test_datetime_callouts(self):
        if False:
            return 10
        data = chartify.examples.example_data()
        price_by_date = data.groupby('date')['total_price'].sum().reset_index()
        ch = chartify.Chart(blank_labels=True, x_axis_type='datetime')
        ch.plot.line(data_frame=price_by_date.sort_values('date'), x_column='date', y_column='total_price')
        ch.callout.line('2017-08-01', orientation='height', line_width=10)
        ch.callout.line_segment('2017-08-01', 10, '2017-09-05', 20)
        ch.callout.box(10, 0, '2017-05-01', '2017-05-05')
        ch.callout.text('text', '2017-05-01', 10)
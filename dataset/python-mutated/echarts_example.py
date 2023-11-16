"""
Example demonstrating how to integrate famous eChart into flexx.
"""
from flexx import flx
flx.assets.associate_asset(__name__, 'https://cdnjs.cloudflare.com/ajax/libs/echarts/4.6.0/echarts.min.js')

class EchartWidget(flx.Widget):
    echart_id = flx.StringProp('_echart_id', settable=True, doc='The unique id to distinguish this echart from other echarts')
    echart_options = flx.DictProp({}, settable=True, doc='the echarts options')
    echart_div_style = flx.StringProp('width: 800px;height:400px;', settable=True, doc='The style for the div which contain echart')

    def get_options(self):
        if False:
            for i in range(10):
                print('nop')
        global window
        myChart = f'myChart_{self.echart_id}'
        return window[myChart]

    @flx.action
    def update_options(self, my_option):
        if False:
            print('Hello World!')
        myChart = self.get_options()
        self._mutate_echart_options(my_option)
        myChart.setOption(self.echart_options)

    @flx.action
    def show_chart(self):
        if False:
            return 10
        myChart = self.get_options()
        myChart.setOption(self.echart_options)

    def _render_dom(self):
        if False:
            for i in range(10):
                print('nop')
        global window
        node = [flx.create_element('div', {'id': self.echart_id, 'style': self.echart_div_style}, ''), flx.create_element('script', {'type': 'text/javascript'}, f"\n                    // based on prepared DOM, initialize echarts instance\n                    var myChart_{self.echart_id} = echarts.init(\n                        document.getElementById('{self.echart_id}'));\n                ")]
        self.show_chart()
        return node
if __name__ == '__main__':

    class MyApp(flx.Widget):
        option1 = {'title': {'text': 'ECharts entry example'}, 'tooltip': {}, 'legend': {'data': ['Sales']}, 'xAxis': {'data': ['shirt', 'cardign', 'chiffon', 'pants', 'heels', 'socks']}, 'yAxis': {}, 'series': [{'name': 'Sales', 'type': 'bar', 'data': [5, 2, 3, 1, 10, 8]}]}
        option2 = {'series': [{'name': 'Customer Source', 'type': 'pie', 'radius': '55%', 'data': [{'value': 235, 'name': 'Video Ad.'}, {'value': 274, 'name': 'Alignment Ad.'}, {'value': 310, 'name': 'Email Marketing'}, {'value': 335, 'name': 'Direct'}, {'value': 400, 'name': 'Search Engine'}], 'roseType': 'angle', 'itemStyle': {'normal': {'shadowColor': 'rgba(0, 0, 0, 0.5)'}}}]}
        option3 = {'tooltip': {'formatter': '{a} <br/>{b} : {c}%'}, 'toolbox': {'feature': {'restore': {}, 'saveAsImage': {}}}, 'series': [{'name': 'Business KPI', 'type': 'gauge', 'detail': {'formatter': '{value}%'}, 'data': [{'value': 50, 'name': 'Complete Rate'}]}]}

        @flx.reaction('btn_change_chart.pointer_click')
        def change_chart(self):
            if False:
                while True:
                    i = 10
            print('click button')
            my_option = self.option1
            my_option['series'][0]['data'] = [1, 2, 3, 4, 5, 6]
            self.echart[0].update_options(my_option)

        def tick(self):
            if False:
                return 10
            global Math, window
            my_option2 = self.option2
            option2_value = (Math.random() * 400).toFixed(2) - 0
            my_option2['series'][0]['data'][0]['value'] = option2_value
            option2_value = (Math.random() * 400).toFixed(2) - 0
            my_option2['series'][0]['data'][1]['value'] = option2_value
            option2_value = (Math.random() * 400).toFixed(2) - 0
            my_option2['series'][0]['data'][3]['value'] = option2_value
            self.echart2[0].update_options(my_option2)
            option3_value = (Math.random() * 100).toFixed(2) - 0
            print('tick...', option3_value)
            my_option3 = self.option3
            my_option3['series'][0]['data'][0]['value'] = option3_value
            self.echart3[0].update_options(my_option3)
            window.setTimeout(self.tick, 1000)

        def init(self):
            if False:
                i = 10
                return i + 15
            super().init()
            with self:
                with flx.VBox():
                    self.echart = (EchartWidget(echart_id='any_unique_id', echart_options=self.option1),)
                    with flx.HBox():
                        self.btn_change_chart = flx.Button(text='Change Chart1')
                    with flx.HBox():
                        self.echart2 = (EchartWidget(echart_id='any_unique_id_2', echart_div_style='width: 400px;height:400px;', echart_options=self.option2),)
                        self.echart3 = (EchartWidget(echart_id='any_unique_id_3', echart_div_style='width: 400px;height:400px;', echart_options=self.option3),)
            self.tick()
    app = flx.App(MyApp)
    app.launch('chrome-app')
    flx.run()
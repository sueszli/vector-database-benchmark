"""
@project: PyCharm
@file: app_callback.py
@author: Shengqiang Zhang
@time: 2019/8/10 01:15
@mail: sqzhang77@gmail.com
"""
import dash
from app_configuration import app
from app_plot import *
from history_data import *
import random
import base64
import time
from os.path import exists
from os import makedirs

def app_callback_function():
    if False:
        return 10

    @app.callback(dash.dependencies.Output('graph_website_count_rank', 'figure'), [dash.dependencies.Input('input_website_count_rank', 'value'), dash.dependencies.Input('store_memory_history_data', 'data')])
    def update(value, store_memory_history_data):
        if False:
            i = 10
            return i + 15
        if store_memory_history_data:
            history_data = store_memory_history_data['history_data']
            figure = plot_bar_website_count_rank(value, history_data)
            return figure
        else:
            raise dash.exceptions.PreventUpdate('cancel the callback')

    @app.callback(dash.dependencies.Output('graph_day_count_rank', 'figure'), [dash.dependencies.Input('store_memory_history_data', 'data')])
    def update(store_memory_history_data):
        if False:
            return 10
        if store_memory_history_data:
            history_data = store_memory_history_data['history_data']
            figure = plot_scatter_website_count_rank(history_data)
            return figure
        else:
            raise dash.exceptions.PreventUpdate('cancel the callback')

    @app.callback(dash.dependencies.Output('table_url_count_rank', 'data'), [dash.dependencies.Input('store_memory_history_data', 'data')])
    def update(store_memory_history_data):
        if False:
            return 10
        if store_memory_history_data:
            history_data = store_memory_history_data['history_data']
            table_data = table_data_url_count_rank(history_data)
            return table_data
        else:
            raise dash.exceptions.PreventUpdate('cancel the callback')

    @app.callback(dash.dependencies.Output('table_url_time_rank', 'data'), [dash.dependencies.Input('store_memory_history_data', 'data')])
    def update(store_memory_history_data):
        if False:
            for i in range(10):
                print('nop')
        if store_memory_history_data:
            history_data = store_memory_history_data['history_data']
            table_data = table_data_url_time_rank(history_data)
            return table_data
        else:
            raise dash.exceptions.PreventUpdate('cancel the callback')

    @app.callback(dash.dependencies.Output('dropdown_time_1', 'options'), [dash.dependencies.Input('store_memory_history_data', 'data')])
    def update(store_memory_history_data):
        if False:
            while True:
                i = 10
        if store_memory_history_data:
            history_data = store_memory_history_data['history_data']
            result_ist = get_history_date_time(history_data)
            result_options = []
            for data in result_ist:
                result_options.append({'label': data, 'value': data})
            return result_options
        else:
            raise dash.exceptions.PreventUpdate('cancel the callback')

    @app.callback(dash.dependencies.Output('dropdown_time_1', 'value'), [dash.dependencies.Input('dropdown_time_1', 'options')])
    def update(available_options):
        if False:
            i = 10
            return i + 15
        if available_options:
            return available_options[0]['value']
        else:
            raise dash.exceptions.PreventUpdate('cancel the callback')

    @app.callback(dash.dependencies.Output('graph_day_diff_time_count', 'figure'), [dash.dependencies.Input('dropdown_time_1', 'value'), dash.dependencies.Input('store_memory_history_data', 'data')])
    def update(date_time_value, store_memory_history_data):
        if False:
            return 10
        if date_time_value:
            if store_memory_history_data:
                history_data = store_memory_history_data['history_data']
                figure = plot_scatter_website_diff_time(date_time_value, history_data)
                return figure
        else:
            raise dash.exceptions.PreventUpdate('cancel the callback')
        raise dash.exceptions.PreventUpdate('cancel the callback')

    @app.callback([dash.dependencies.Output('graph_search_word_count_rank', 'figure'), dash.dependencies.Output('graph_search_engine_count_rank', 'figure')], [dash.dependencies.Input('store_memory_history_data', 'data')])
    def update(store_memory_history_data):
        if False:
            print('Hello World!')
        if store_memory_history_data is not None:
            search_word = store_memory_history_data['search_word']
            (figure_1, figure_2) = plot_bar_search_word_count_rank(search_word)
            return (figure_1, figure_2)
        else:
            raise dash.exceptions.PreventUpdate('cancel the callback')

    @app.callback(dash.dependencies.Output('store_memory_history_data', 'data'), [dash.dependencies.Input('dcc_upload_file', 'contents')])
    def update(contents):
        if False:
            return 10
        if contents is not None:
            (content_type, content_string) = contents.split(',')
            decoded = base64.b64decode(content_string)
            suffix = [str(random.randint(0, 100)) for i in range(10)]
            suffix = ''.join(suffix)
            suffix = suffix + str(int(time.time()))
            file_name = 'History_' + suffix
            if not exists('data'):
                makedirs('data')
            path = 'data' + '/' + file_name
            with open(file=path, mode='wb+') as f:
                f.write(decoded)
            history_data = get_history_data(path)
            search_word = get_search_word(path)
            if history_data != 'error':
                date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print('新接收到一条客户端的数据, 数据正确, 时间:{}'.format(date_time))
                store_data = {'history_data': history_data, 'search_word': search_word}
                return store_data
            else:
                date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print('新接收到一条客户端的数据, 数据错误, 时间:{}'.format(date_time))
                return None
        return None
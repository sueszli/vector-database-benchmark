"""
@project: PyCharm
@file: app_plot.py
@author: Shengqiang Zhang
@time: 2019/8/10 02:03
@mail: sqzhang77@gmail.com
"""
import plotly.graph_objs as go
import time

def url_simplification(url):
    if False:
        print('Hello World!')
    tmp_url = url
    try:
        url = url.split('//')
        url = url[1].split('/', 1)
        url = url[0].replace('www.', '')
        return url
    except IndexError:
        return tmp_url

def get_top_k_from_dict(origin_dict, k):
    if False:
        return 10
    origin_dict_len = len(origin_dict)
    n = k
    if n > origin_dict_len:
        n = origin_dict_len
    new_data = sorted(origin_dict.items(), key=lambda item: item[1], reverse=True)
    new_data = new_data[:n]
    new_dict = {}
    for l in new_data:
        new_dict[l[0]] = l[1]
    return new_dict

def get_top_k_from_dict_value_1(origin_dict, k):
    if False:
        print('Hello World!')
    origin_dict_len = len(origin_dict)
    n = k
    if n > origin_dict_len:
        n = origin_dict_len
    new_data = sorted(origin_dict.items(), key=lambda item: item[1][0], reverse=True)
    new_data = new_data[:n]
    new_dict = {}
    for l in new_data:
        new_dict[l[0]] = l[1]
    return new_dict

def sort_time_dict(origin_dict):
    if False:
        print('Hello World!')
    new_data = sorted(origin_dict.items(), key=lambda item: time.mktime(time.strptime(item[0], '%Y-%m-%d')), reverse=False)
    new_dict = {}
    for l in new_data:
        new_dict[l[0]] = l[1]
    return new_dict

def convert_to_number(value):
    if False:
        for i in range(10):
            print('nop')
    try:
        x = int(value)
    except TypeError:
        return 0
    except ValueError:
        return 0
    except Exception as e:
        return 0
    else:
        return x

def plot_bar_website_count_rank(value, history_data):
    if False:
        print('Hello World!')
    dict_data = {}
    for data in history_data:
        url = data[1]
        key = url_simplification(url)
        if key in dict_data.keys():
            dict_data[key] += 1
        else:
            dict_data[key] = 0
    k = convert_to_number(value)
    top_10_dict = get_top_k_from_dict(dict_data, k)
    figure = go.Figure(data=[go.Bar(x=[i for i in top_10_dict.keys()], y=[i for i in top_10_dict.values()], name='bar', marker=go.bar.Marker(color='rgb(55, 83, 109)'))], layout=go.Layout(showlegend=False, margin=go.layout.Margin(l=40, r=0, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title='网站'), yaxis=dict(title='次数')))
    return figure

def plot_bar_search_word_count_rank(search_word):
    if False:
        while True:
            i = 10
    dict_data = {}
    for data in search_word:
        search_item = data[1]
        key = search_item
        if key in dict_data.keys():
            dict_data[key][0] += 1
        else:
            url_link = data[2]
            url_visit_time = data[3]
            dict_data[key] = [1, url_link, url_visit_time]
    top_10_dict = get_top_k_from_dict_value_1(dict_data, 10)
    figure_1 = go.Figure(data=[go.Bar(x=[key for key in top_10_dict.keys()], y=[value[0] for value in top_10_dict.values()], name='bar', marker=go.bar.Marker(color='rgb(55, 83, 109)'))], layout=go.Layout(showlegend=False, margin=go.layout.Margin(l=40, r=0, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title='关键词'), yaxis=dict(title='次数')))
    search_engine_list = ['www.google.com', 'www.bing.com', 'www.yahoo.com', 'www.baidu.com', 'www.sogou.com', 'www.so.com']
    search_engine_value = [0, 0, 0, 0, 0, 0]
    for (key, value) in dict_data.items():
        for i in range(len(search_engine_list)):
            if search_engine_list[i] in value[1]:
                search_engine_value[i] += 1
                break
    figure_2 = go.Figure(data=[go.Pie(labels=search_engine_list, values=search_engine_value, hole=0.3)])
    return (figure_1, figure_2)

def plot_scatter_website_count_rank(history_data):
    if False:
        i = 10
        return i + 15
    dict_data = {}
    for data in history_data:
        date_time = data[5]
        unix_time_samp = date_time / 1000000 - 11644473600
        unix_time_samp += 28800
        key = time.strftime('%Y-%m-%d', time.gmtime(unix_time_samp))
        if key in dict_data.keys():
            dict_data[key] += 1
        else:
            dict_data[key] = 0
    dict_sort_data = sort_time_dict(dict_data)
    max_value_dict = max([i for i in dict_sort_data.values()])
    figure = go.Figure(data=[go.Scatter(x=[i for i in dict_sort_data.keys()], y=[i for i in dict_sort_data.values()], name='lines+markers', mode='lines+markers', marker_color='rgba(55, 83, 109, .8)', marker=dict(size=[i / max_value_dict * 30 for i in dict_sort_data.values()]), fill='tozeroy')], layout=go.Layout(showlegend=False, margin=go.layout.Margin(l=40, r=0, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title='时间'), yaxis=dict(title='次数')))
    return figure

def table_data_url_count_rank(history_data):
    if False:
        return 10
    dict_data = {}
    for data in history_data:
        url_id = data[0]
        key = url_id
        if key in dict_data.keys():
            dict_data[key][0] += 1
            dict_data[key][1] = data[1]
            dict_data[key][2] = data[2]
        else:
            dict_data[key] = [0, '', '']
    top_k_dict = get_top_k_from_dict_value_1(dict_data, 100)
    table_data = []
    for (index, item) in enumerate(top_k_dict.items()):
        table_data.append({'id': index + 1, 'url': item[1][1], 'title': item[1][2], 'count': item[1][0]})
    return table_data

def table_data_url_time_rank(history_data):
    if False:
        print('Hello World!')
    dict_data = {}
    for data in history_data:
        url_id = data[0]
        key = url_id
        if key in dict_data.keys():
            dict_data[key][0] += round(data[8] / 1000000 / 3600, 2)
            dict_data[key][1] = data[1]
            dict_data[key][2] = data[2]
        else:
            dict_data[key] = [0.0, '', '']
    top_k_dict = get_top_k_from_dict_value_1(dict_data, 100)
    table_data = []
    for (index, item) in enumerate(top_k_dict.items()):
        table_data.append({'id': index + 1, 'url': item[1][1], 'title': item[1][2], 'count': item[1][0]})
    return table_data

def get_history_date_time(history_data):
    if False:
        while True:
            i = 10
    list_date_time = []
    for data in history_data:
        date_time = data[5]
        unix_time_samp = date_time / 1000000 - 11644473600
        unix_time_samp += 28800
        list_date_time.append(unix_time_samp)
    for i in range(len(list_date_time)):
        unix_time_samp = list_date_time[i]
        list_date_time[i] = time.strftime('%Y-%m-%d', time.gmtime(unix_time_samp))
    list_unique = list(set(list_date_time))
    list_unique_sort = sorted(list_unique)
    return list_unique_sort

def plot_scatter_website_diff_time(date_time_value, history_data):
    if False:
        while True:
            i = 10
    if date_time_value is None:
        return {}
    dict_data = {}
    for i in range(0, 24):
        dict_data[i] = 0
    for data in history_data:
        date_time = data[5]
        unix_time_samp = date_time / 1000000 - 11644473600
        unix_time_samp += 28800
        current_day = time.strftime('%Y-%m-%d', time.gmtime(unix_time_samp))
        if date_time_value == current_day:
            key = time.strftime('%H', time.gmtime(unix_time_samp))
            key = int(key)
            if key in dict_data.keys():
                dict_data[key] += 1
    max_value_dict = max([i for i in dict_data.values()])
    if max_value_dict == 0:
        return {}
    figure = go.Figure(data=[go.Scatter(x=[i for i in dict_data.keys()], y=[i for i in dict_data.values()], name='lines+markers', mode='lines+markers', marker_color='rgba(55, 83, 109, .8)', marker=dict(size=[i / max_value_dict * 30 for i in dict_data.values()]), fill='tozeroy')], layout=go.Layout(showlegend=False, margin=go.layout.Margin(l=40, r=0, t=40, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title='时刻(24小时制)'), yaxis=dict(title='次数')))
    return figure
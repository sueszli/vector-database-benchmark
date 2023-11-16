import os
import pandas
from sklearn.linear_model import LinearRegression

def linear_model_main(_distances, _press_times, target_distance):
    if False:
        for i in range(10):
            print('nop')
    regr = LinearRegression()
    regr.fit(_distances, _press_times)
    predict_press_time = regr.predict(target_distance)
    result = {}
    result['intercept'] = regr.intercept_
    result['coefficient'] = regr.coef_
    result['value'] = predict_press_time
    return result

def computing_k_b_v(target_distance):
    if False:
        for i in range(10):
            print('nop')
    result = linear_model_main(distances, press_times, target_distance)
    b = result['intercept']
    k = result['coefficient']
    v = result['value']
    return (k[0], b[0], v[0])

def add_data(distance, press_time):
    if False:
        for i in range(10):
            print('nop')
    distances.append([distance])
    press_times.append([press_time])
    save_data('./jump_range.csv', distances, press_times)

def save_data(file_name, distances, press_times):
    if False:
        for i in range(10):
            print('nop')
    pf = pandas.DataFrame({'Distance': distances, 'Press_time': press_times})
    pf.to_csv(file_name, index=False, sep=',')

def get_data(file_name):
    if False:
        while True:
            i = 10
    data = pandas.read_csv(file_name)
    distance_array = []
    press_time_array = []
    for (distance, press_time) in zip(data['Distance'], data['Press_time']):
        distance_array.append([float(distance.strip().strip('[]'))])
        press_time_array.append([float(press_time.strip().strip('[]'))])
    return (distance_array, press_time_array)

def init():
    if False:
        return 10
    global distances, press_times
    distances = []
    press_times = []
    if os.path.exists('./jump_range.csv'):
        (distances, press_times) = get_data('./jump_range.csv')
    else:
        save_data('./jump_range.csv', [], [])
        return 0

def get_result_len():
    if False:
        for i in range(10):
            print('nop')
    return len(distances)
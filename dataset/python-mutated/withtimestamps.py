def time_tuple2unix_time():
    if False:
        for i in range(10):
            print('nop')
    import time
    time_tuple = time.strptime('2020-03-19 20:50:00', '%Y-%m-%d %H:%M:%S')
    unix_time = time.mktime(time_tuple)
    return unix_time

def datetime2unix_time():
    if False:
        i = 10
        return i + 15
    import time
    import datetime
    now = datetime.datetime.now()
    time_tuple = now.timetuple()
    unix_time = time.mktime(time_tuple)
    return unix_time
from pyecharts import Bar

def paint(count, data1, data2):
    if False:
        for i in range(10):
            print('nop')
    columns = []
    for i in range(1, count + 1):
        columns.append(i)
    bar = Bar('模型训练时长对比图')
    bar.add('单相电子式电能表', columns, data1, mark_line=['average'], mark_point=['max', 'min'])
    bar.add('电子电流表', columns, data2, mark_line=['average'], mark_point=['max', 'min'])
    bar.render('templates/bar.html')
if __name__ == '__main__':
    paint()
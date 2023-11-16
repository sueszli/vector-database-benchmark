from pyecharts import Line
from pyecharts import Bar

def paint(count, data1, data2):
    if False:
        for i in range(10):
            print('nop')
    columns = []
    for i in range(1, count + 1):
        columns.append(i)
    line = Line('模型训练准确率对比图')
    line.add('电子式电能表', columns, data1, is_label_show=True)
    line.add('单相电子式电流表', columns, data2, is_label_show=True)
    line.render('templates/line.html')
    print()
if __name__ == '__main__':
    paint()
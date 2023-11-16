from pyecharts import Line
from pyecharts import Bar
# 利用折线图绘制图表 反映模型训练性能
def paint(count,data1, data2):
    columns = []
    for i in range(1, count+1):
        columns.append(i)
    # data3=["46.234","45.325","65.765","47.125"]
    # data4=["65.234","67.980","65.786","67.459"]
    line = Line("模型训练准确率对比图")
    # is_label_show是设置每个数据点的上方数据是否进行显示
    line.add("电子式电能表", columns, data1, is_label_show=True)
    line.add("单相电子式电流表", columns, data2, is_label_show=True)
    line.render('templates/line.html')
    print()
    # bar.render()

if __name__ == '__main__':
    paint()
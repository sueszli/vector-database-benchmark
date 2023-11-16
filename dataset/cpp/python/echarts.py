from pyecharts import Bar
# 利用柱状图绘制图表 反映模型训练性能
def paint(count,data1, data2):
    columns = []
    for i in range(1, count+1):
        columns.append(i)
    bar = Bar("模型训练时长对比图")
    # 添加柱状图的数据及配置项
    bar.add("单相电子式电能表", columns, data1, mark_line=["average"], mark_point=["max", "min"])
    bar.add("电子电流表", columns, data2, mark_line=["average"], mark_point=["max", "min"])
    bar.render('templates/bar.html')
    # bar.render()

if __name__ == '__main__':
    paint()
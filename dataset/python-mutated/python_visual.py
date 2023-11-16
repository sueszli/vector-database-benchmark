"""
python_visual.py by xianhu
"""
import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
myfont = fm.FontProperties(fname='/Library/Fonts/Songti.ttc', size=14)
matplotlib.rcParams['axes.unicode_minus'] = False

def simple_plot():
    if False:
        i = 10
        return i + 15
    '\n    simple plot\n    '
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    (y_cos, y_sin) = (np.cos(x), np.sin(x))
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('简单曲线图', fontproperties=myfont)
    plt.grid(True)
    plt.xlabel('X轴', fontproperties=myfont)
    plt.xlim(-4.0, 4.0)
    plt.xticks(np.linspace(-4, 4, 9, endpoint=True))
    plt.ylabel('Y轴', fontproperties=myfont)
    plt.ylim(-1.0, 1.0)
    plt.yticks(np.linspace(-1, 1, 9, endpoint=True))
    plt.plot(x, y_cos, 'b--', linewidth=2.0, label='cos示例')
    plt.plot(x, y_sin, 'g-', linewidth=2.0, label='sin示例')
    plt.legend(loc='upper left', prop=myfont, shadow=True)
    plt.show()
    return

def simple_advanced_plot():
    if False:
        for i in range(10):
            print('nop')
    '\n    simple advanced plot\n    '
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    (y_cos, y_sin) = (np.cos(x), np.sin(x))
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('复杂曲线图', fontproperties=myfont)
    plt.grid(True)
    ax_1 = plt.subplot(111)
    ax_1.plot(x, y_cos, color='blue', linewidth=2.0, linestyle='--', label='左cos')
    ax_1.legend(loc='upper left', prop=myfont, shadow=True)
    ax_1.set_ylabel('左cos的y轴', fontproperties=myfont)
    ax_1.set_ylim(-1.0, 1.0)
    ax_1.set_yticks(np.linspace(-1, 1, 9, endpoint=True))
    ax_2 = ax_1.twinx()
    ax_2.plot(x, y_sin, color='green', linewidth=2.0, linestyle='-', label='右sin')
    ax_2.legend(loc='upper right', prop=myfont, shadow=True)
    ax_2.set_ylabel('右sin的y轴', fontproperties=myfont)
    ax_2.set_ylim(-2.0, 2.0)
    ax_2.set_yticks(np.linspace(-2, 2, 9, endpoint=True))
    ax_1.set_xlabel('x轴', fontproperties=myfont)
    ax_1.set_xlim(-4.0, 4.0)
    ax_1.set_xticks(np.linspace(-4, 4, 9, endpoint=True))
    plt.show()
    return

def subplot_plot():
    if False:
        print('Hello World!')
    '\n    subplot plot\n    '
    style_list = ['g+-', 'r*-', 'b.-', 'yo-']
    for num in range(4):
        x = np.linspace(0.0, 2 + num, num=10 * (num + 1))
        y = np.sin((5 - num) * np.pi * x)
        plt.subplot(2, 2, num + 1)
        plt.title('子图 %d' % (num + 1), fontproperties=myfont)
        plt.plot(x, y, style_list[num])
    plt.show()
    return

def bar_plot():
    if False:
        i = 10
        return i + 15
    '\n    bar plot\n    '
    means_men = (20, 35, 30, 35, 27)
    means_women = (25, 32, 34, 20, 25)
    plt.title('柱状图', fontproperties=myfont)
    index = np.arange(len(means_men))
    bar_width = 0.35
    plt.bar(index, means_men, width=bar_width, alpha=0.2, color='b', label='男生')
    plt.bar(index + bar_width, means_women, width=bar_width, alpha=0.8, color='r', label='女生')
    plt.legend(loc='upper right', prop=myfont, shadow=True)
    for (x, y) in zip(index, means_men):
        plt.text(x, y + 0.3, y, ha='center', va='bottom')
    for (x, y) in zip(index, means_women):
        plt.text(x + bar_width, y + 0.3, y, ha='center', va='bottom')
    plt.ylim(0, 45)
    plt.xlabel('分组Group', fontproperties=myfont)
    plt.ylabel('得分Scores', fontproperties=myfont)
    plt.xticks(index + bar_width / 2, ('A组', 'B组', 'C组', 'D组', 'E组'), fontproperties=myfont)
    plt.show()
    return

def barh_plot():
    if False:
        i = 10
        return i + 15
    '\n    barh plot\n    '
    means_men = (20, 35, 30, 35, 27)
    means_women = (25, 32, 34, 20, 25)
    plt.title('横向柱状图', fontproperties=myfont)
    index = np.arange(len(means_men))
    bar_height = 0.35
    plt.barh(index, means_men, height=bar_height, alpha=0.2, color='b', label='Men')
    plt.barh(index + bar_height, means_women, height=bar_height, alpha=0.8, color='r', label='Women')
    plt.legend(loc='upper right', shadow=True)
    for (x, y) in zip(index, means_men):
        plt.text(y + 0.3, x, y, ha='left', va='center')
    for (x, y) in zip(index, means_women):
        plt.text(y + 0.3, x + bar_height, y, ha='left', va='center')
    plt.xlim(0, 45)
    plt.xlabel('Scores')
    plt.ylabel('Group')
    plt.yticks(index + bar_height / 2, ('A', 'B', 'C', 'D', 'E'))
    plt.show()
    return

def bar_advanced_plot():
    if False:
        while True:
            i = 10
    '\n    bar advanced plot\n    '
    means_men = np.array((20, 35, 30, 35, 27, 25, 32, 34, 20, 25))
    means_women = np.array((25, 32, 34, 20, 25, 20, 35, 30, 35, 27))
    plt.title('高级柱状图', fontproperties=myfont)
    index = np.arange(len(means_men))
    bar_width = 0.8
    plt.bar(index, means_men, width=bar_width, alpha=0.4, color='b', label='Men')
    plt.bar(index, -means_women, width=bar_width, alpha=0.4, color='r', label='Women')
    plt.plot(index, means_men, marker='o', linestyle='-', color='r', label='Men line')
    plt.plot(index, -means_women, marker='.', linestyle='--', color='b', label='Women line')
    for (x, y) in zip(index, means_men):
        plt.text(x, y + 1, y, ha='center', va='bottom')
    for (x, y) in zip(index, means_women):
        plt.text(x, -y - 1, y, ha='center', va='top')
    plt.ylim(-45, 80)
    plt.legend(loc='upper left', shadow=True)
    plt.show()
    return

def table_plot():
    if False:
        while True:
            i = 10
    '\n    table plot\n    '
    data = np.array([[1, 4, 2, 5, 2], [2, 1, 1, 3, 6], [5, 3, 6, 4, 1]])
    plt.title('层次柱状图', fontproperties=myfont)
    index = np.arange(len(data[0]))
    color_index = ['r', 'g', 'b']
    bottom = np.array([0, 0, 0, 0, 0])
    for i in range(len(data)):
        plt.bar(index, data[i], width=0.5, color=color_index[i], bottom=bottom, alpha=0.7, label='标签 %d' % i)
        bottom += data[i]
    plt.legend(loc='upper left', prop=myfont, shadow=True)
    plt.show()
    return

def histograms_plot():
    if False:
        i = 10
        return i + 15
    '\n    histograms plot\n    '
    (mu, sigma) = (100, 15)
    x = mu + sigma * np.random.randn(10000)
    plt.title('直方图', fontproperties=myfont)
    (n, bins, patches) = plt.hist(x, bins=50, normed=1, cumulative=False, color='green', alpha=0.6, label='直方图')
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', label='线条')
    plt.legend(loc='upper left', prop=myfont, shadow=True)
    plt.show()
    return

def pie_plot():
    if False:
        while True:
            i = 10
    '\n    pie plot\n    '
    sizes = [15, 30, 45, 10]
    labels = ['Frogs', '中文', 'Dogs', 'Logs']
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    plt.title('饼图', fontproperties=myfont)
    explode = [0, 0.05, 0, 0]
    (patches, l_text, p_text) = plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    for text in l_text:
        text.set_fontproperties(myfont)
    plt.axis('equal')
    plt.show()
    return

def scatter_plot():
    if False:
        return 10
    '\n    scatter plot\n    '
    point_count = 1000
    x_index = np.random.random(point_count)
    y_index = np.random.random(point_count)
    plt.title('散点图', fontproperties=myfont)
    color_list = np.random.random(point_count)
    scale_list = np.random.random(point_count) * 100
    plt.scatter(x_index, y_index, s=scale_list, c=color_list, marker='o')
    plt.show()
    return

def fill_plot():
    if False:
        return 10
    '\n    fill plot\n    '
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000, endpoint=True)
    y = np.sin(x)
    plt.title('填充图', fontproperties=myfont)
    plt.plot(x, y, color='blue', alpha=1.0)
    plt.fill_between(x, 0, y, where=y > 0, color='blue', alpha=0.25)
    plt.fill_between(x, 0, y, where=y < 0, color='red', alpha=0.25)
    plt.show()
    return

def radar_plot():
    if False:
        i = 10
        return i + 15
    '\n    radar plot\n    '
    labels = np.array(['A组', 'B组', 'C组', 'D组', 'E组', 'F组'])
    data = np.array([68, 83, 90, 77, 89, 73])
    theta = np.linspace(0, 2 * np.pi, len(data), endpoint=False)
    data = np.concatenate((data, [data[0]]))
    theta = np.concatenate((theta, [theta[0]]))
    plt.subplot(111, polar=True)
    plt.title('雷达图', fontproperties=myfont)
    plt.thetagrids(theta * (180 / np.pi), labels=labels, fontproperties=myfont)
    plt.rgrids(np.arange(20, 100, 20), labels=np.arange(20, 100, 20), angle=0)
    plt.ylim(0, 100)
    plt.plot(theta, data, 'bo-', linewidth=2)
    plt.fill(theta, data, color='red', alpha=0.25)
    plt.show()
    return

def three_dimension_scatter():
    if False:
        print('Hello World!')
    '\n    3d scatter plot\n    '
    x = np.random.random(100)
    y = np.random.random(100)
    z = np.random.random(100)
    color = np.random.random(100)
    scale = np.random.random(100) * 100
    fig = plt.figure()
    fig.suptitle('三维散点图', fontproperties=myfont)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=scale, c=color, marker='.')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.show()
    return

def three_dimension_line():
    if False:
        while True:
            i = 10
    '\n    3d line plot\n    '
    x = np.linspace(0, 1, 1000)
    y = np.linspace(0, 1, 1000)
    z = np.sin(x * 2 * np.pi) / (y + 0.1)
    fig = plt.figure()
    ax = fig.gca(projection='3d', title='plot title')
    ax.plot(x, y, z, color='red', linestyle='-')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    return

def three_dimension_bar():
    if False:
        return 10
    '\n    3d bar plot\n    '
    xpos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ypos = [2, 3, 4, 5, 1, 6, 2, 1, 7, 2]
    zpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dx = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dy = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dz = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fig = plt.figure()
    ax = fig.gca(projection='3d', title='plot title')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    return
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential, regularizers
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False
OUTPUT_DIR = 'output_dir'
N_EPOCHS = 500

def load_dataset():
    if False:
        i = 10
        return i + 15
    N_SAMPLES = 1000
    TEST_SIZE = None
    (X, y) = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=100)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    return (X, y, X_train, X_test, y_train, y_test)

def make_plot(X, y, plot_name, file_name, XX=None, YY=None, preds=None, dark=False, output_dir=OUTPUT_DIR):
    if False:
        while True:
            i = 10
    if dark:
        plt.style.use('dark_background')
    else:
        sns.set_style('whitegrid')
    axes = plt.gca()
    axes.set_xlim([-2, 3])
    axes.set_ylim([-1.5, 2])
    axes.set(xlabel='$x_1$', ylabel='$x_2$')
    plt.title(plot_name, fontsize=20, fontproperties='SimHei')
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(right=0.8)
    if XX is not None and YY is not None and (preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.08, cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[0.5], cmap='Greys', vmin=0, vmax=0.6)
    markers = ['o' if i == 1 else 's' for i in y.ravel()]
    mscatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, cmap=plt.cm.Spectral, edgecolors='none', m=markers, ax=axes)
    plt.savefig(output_dir + '/' + file_name)
    plt.close()

def mscatter(x, y, ax=None, m=None, **kw):
    if False:
        print('Hello World!')
    import matplotlib.markers as mmarkers
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if m is not None and len(m) == len(x):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def network_layers_influence(X_train, y_train):
    if False:
        i = 10
        return i + 15
    for n in range(5):
        model = Sequential()
        model.add(layers.Dense(8, input_dim=2, activation='relu'))
        for _ in range(n):
            model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
        xx = np.arange(-2, 3, 0.01)
        yy = np.arange(-1.5, 2, 0.01)
        (XX, YY) = np.meshgrid(xx, yy)
        preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
        title = '网络层数：{0}'.format(2 + n)
        file = '网络容量_%i.png' % (2 + n)
        make_plot(X_train, y_train, title, file, XX, YY, preds, output_dir=OUTPUT_DIR + '/network_layers')

def dropout_influence(X_train, y_train):
    if False:
        for i in range(10):
            print('nop')
    for n in range(5):
        model = Sequential()
        model.add(layers.Dense(8, input_dim=2, activation='relu'))
        counter = 0
        for _ in range(5):
            model.add(layers.Dense(64, activation='relu'))
        if counter < n:
            counter += 1
            model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
        xx = np.arange(-2, 3, 0.01)
        yy = np.arange(-1.5, 2, 0.01)
        (XX, YY) = np.meshgrid(xx, yy)
        preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
        title = '无Dropout层' if n == 0 else '{0}层 Dropout层'.format(n)
        file = 'Dropout_%i.png' % n
        make_plot(X_train, y_train, title, file, XX, YY, preds, output_dir=OUTPUT_DIR + '/dropout')

def build_model_with_regularization(_lambda):
    if False:
        for i in range(10):
            print('nop')
    model = Sequential()
    model.add(layers.Dense(8, input_dim=2, activation='relu'))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_weights_matrix(model, layer_index, plot_name, file_name, output_dir=OUTPUT_DIR):
    if False:
        return 10
    weights = model.layers[layer_index].get_weights()[0]
    shape = weights.shape
    X = np.array(range(shape[1]))
    Y = np.array(range(shape[0]))
    (X, Y) = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.title(plot_name, fontsize=20, fontproperties='SimHei')
    ax.plot_surface(X, Y, weights, cmap=plt.get_cmap('rainbow'), linewidth=0)
    ax.set_xlabel('网格x坐标', fontsize=16, rotation=0, fontproperties='SimHei')
    ax.set_ylabel('网格y坐标', fontsize=16, rotation=0, fontproperties='SimHei')
    ax.set_zlabel('权值', fontsize=16, rotation=90, fontproperties='SimHei')
    plt.savefig(output_dir + '/' + file_name + '.svg')
    plt.close(fig)

def regularizers_influence(X_train, y_train):
    if False:
        i = 10
        return i + 15
    for _lambda in [1e-05, 0.001, 0.1, 0.12, 0.13]:
        model = build_model_with_regularization(_lambda)
        model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
        layer_index = 2
        plot_title = '正则化系数：{}'.format(_lambda)
        file_name = '正则化网络权值_' + str(_lambda)
        plot_weights_matrix(model, layer_index, plot_title, file_name, output_dir=OUTPUT_DIR + '/regularizers')
        xx = np.arange(-2, 3, 0.01)
        yy = np.arange(-1.5, 2, 0.01)
        (XX, YY) = np.meshgrid(xx, yy)
        preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
        title = '正则化系数：{}'.format(_lambda)
        file = '正则化_%g.svg' % _lambda
        make_plot(X_train, y_train, title, file, XX, YY, preds, output_dir=OUTPUT_DIR + '/regularizers')

def main():
    if False:
        while True:
            i = 10
    (X, y, X_train, X_test, y_train, y_test) = load_dataset()
    make_plot(X, y, None, '月牙形状二分类数据集分布.svg')
    network_layers_influence(X_train, y_train)
    dropout_influence(X_train, y_train)
    regularizers_influence(X_train, y_train)
if __name__ == '__main__':
    main()
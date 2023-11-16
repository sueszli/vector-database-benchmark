from Util.Import import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import statsmodels.api as sm

def plot3d(df):
    if False:
        print('Hello World!')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    Y = [0.25, 0.2, 0.15, 0.1, 0.05, 0.025]
    (X, Y) = np.meshgrid(X, Y)
    Z = [i for i in df['AvgTopicLen']][::-1]
    Z = np.reshape(Z, X.shape)
    T = [i for i in df['Time']][::-1]
    T = np.reshape(T, X.shape)
    print(T)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.RdBu, linewidth=0, antialiased=True)
    surf2 = ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.RdBu, linewidth=0, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def getDensity(numTweets, desired=30):
    if False:
        print('Hello World!')
    constant = 94.569726
    density = constant / numTweets
    if density > 1:
        density = 1
    elif density < 0.01:
        density = 0.01
    return density
if __name__ == '__main__':
    df = pd.read_csv('../GridResults/gridResults2.csv', index_col=0)
    df['Intercept'] = 1
    plot3d(df)
    X = df[[0, 1, 4]]
    y = df[[2]]
    model = sm.OLS(y, X)
    model = model.fit()
    print(model.params)
from pandas.core.config_init import pc_line_width_deprecation_warning

__author__ = 'nastra'

import scipy as sp
import os
import matplotlib.pyplot as plt


def init_and_cleanup_data(path, delimiter):
    data = sp.genfromtxt(path, delimiter=delimiter)
    hours = data[:, 0] # contains the hours
    webhits = data[:, 1] # contains the number of web hits at a particular hour
    hours = hours[~sp.isnan(webhits)]
    webhits = webhits[~sp.isnan(webhits)]
    return (hours, webhits)


def plot_data(x, y):
    plt.scatter(x, y)
    plt.title("Web traffic over last month")
    plt.xlabel("Time")
    plt.ylabel("Web Hits per Hour")
    plt.xticks([w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])
    plt.autoscale(tight=True)
    plt.grid()
    plt.show()


def squared_error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


def plot_trained_model(func):
    fx = sp.linspace(0, x[-1], 1000) # generate x-values for plotting
    plt.plot(fx, func(fx), linewidth=4)
    plt.legend(["d=%i" % func.order], loc="upper left")

def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks(
        [w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

# all examples will have three classes in this file
colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']

(x, y) = init_and_cleanup_data("data/web_traffic.tsv", "\t")
#plot_data(x, y)

# full=True gives additional info on the fitting process
# fp1 contains the parameters of the fitted model function Theta0 and Theta1

# The model parameters are [2.59619213  989.02487106], meaning that the best straight line fit is:
# f(x) = 2.59619213 * x + 989.02487106
# which is nothing more that doing curve fitting
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)

# f(x) =0.0 105 * x**2 - 5.2654 * x + 1974.7608
fp2 = sp.polyfit(x, y, 2) # 2-degree polynomial

fp3 = sp.polyfit(x, y, 3)
fp10 = sp.polyfit(x, y, 10)
fp100 = sp.polyfit(x, y, 100)

# poly1d creates a model function from the model parameters
f1 = sp.poly1d(fp1)
f2 = sp.poly1d(fp2)
f3 = sp.poly1d(fp3)
f10 = sp.poly1d(fp10)
f100 = sp.poly1d(fp100)
models = [f1, f2, f3, f10, f100]

#models f10 and f100 result in overfitting
plot_models(x, y, models, None)


# the data shows us that there is an inflection point between weeks 3 and 4
inflection = 3.5*7*24

# data before the inflection point
xa = x[:inflection]
ya = y[:inflection]

# data after the inflection point
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

fa_error = squared_error(fa, xa, ya)
fb_error = squared_error(fb, xb, yb)

plot_models(x, y, [fa, fb], None)

print("Errors for the complete data set:")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, squared_error(f, x, y)))
print("-------------------------------------------------------------")
print("Errors for only the time after inflection point")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, squared_error(f, xb, yb)))
print("-------------------------------------------------------------")
print("Error inflection=%f" % (squared_error(fa, xa, ya) + squared_error(fb, xb, yb)))

# extrapolating into the future
plot_models(x, y, [f1, f2, f3, f10, f100], None, mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100), ymax=10000, xmin=0 * 7 * 24)


print("-------------------------------------------------------------")
print("Trained only on data after inflection point")
fb1 = fb
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))
fb3 = sp.poly1d(sp.polyfit(xb, yb, 3))
fb10 = sp.poly1d(sp.polyfit(xb, yb, 10))
fb100 = sp.poly1d(sp.polyfit(xb, yb, 100))
print("-------------------------------------------------------------")
print("Errors for only the time after inflection point")
for f in [fb1, fb2, fb3, fb10, fb100]:
    print("Error d=%i: %f" % (f.order, squared_error(f, xb, yb)))

plot_models(
    x, y, [fb1, fb2, fb3, fb10, fb100], None,
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)



# separating training from testing data
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train], 100))
test_models = [fbt1, fbt2, fbt3, fbt10, fbt100]
print("-------------------------------------------------------------")
print("Test errors for only the time after inflection point")
for f in test_models:
    print("Error d=%i: %f" % (f.order, squared_error(f, xb[test], yb[test])))

plot_models(x, y, test_models, None,  mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),  ymax=10000, xmin=0 * 7 * 24)

print(fbt2)
print(fbt2 - 100000)
print("-------------------------------------------------------------")
from scipy.optimize import fsolve
reached_max = fsolve(fbt2 - 100000, 800)/(7*24)
print("100,000 hits/hour expected at week %f" % reached_max[0])
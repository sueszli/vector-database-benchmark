import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def partial_plot_test():
    if False:
        print('Hello World!')
    iris = h2o.import_file(pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    iris['class'] = iris['class'].asfactor()
    iris['random_cat'] = iris['class']
    predictors = iris.col_names[:-1]
    response = 'class'
    (train, valid) = iris.split_frame(ratios=[0.8], seed=1234)
    model = H2OGeneralizedLinearEstimator(family='multinomial')
    model.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    cols = ['petal_len']
    targets = ['Iris-setosa']
    pdp_petal_len_se = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=False, plot=True, server=True)
    print(pdp_petal_len_se)
    pdp_petal_len_se_std = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=True, plot=True, server=True)
    print(pdp_petal_len_se_std)
    targets = ['Iris-setosa', 'Iris-virginica']
    pdp_petal_len_se_vi = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=False, plot=True, server=True)
    print(pdp_petal_len_se_vi)
    pdp_petal_len_se_vi_std = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=True, plot=True, server=True)
    print(pdp_petal_len_se_vi_std)
    targets = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
    pdp_petal_len_se_vi_ve_std = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=True, plot=True, server=True)
    print(pdp_petal_len_se_vi_ve_std)
    cols = ['sepal_len', 'petal_len']
    pdp_petal_len_sepal_len_se_vi_ve_std = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=True, plot=True, server=True)
    print(pdp_petal_len_sepal_len_se_vi_ve_std)
    cols = ['sepal_len', 'petal_len', 'sepal_wid']
    pdp_petal_len_sepal_len_sepal_wid_se_vi_ve = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=False, plot=True, server=True)
    print(pdp_petal_len_sepal_len_sepal_wid_se_vi_ve)
    pdp_petal_len_sepal_len_sepal_wid_se_vi_ve_std = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=True, plot=True, server=True)
    print(pdp_petal_len_sepal_len_sepal_wid_se_vi_ve_std)
    cols = ['random_cat']
    targets = ['Iris-setosa']
    pdp_petal_len_cat = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=False, plot=True, server=True)
    print(pdp_petal_len_cat)
    targets = ['Iris-setosa', 'Iris-versicolor']
    pdp_petal_len_cat_std = model.partial_plot(frame=iris, cols=cols, targets=targets, plot_stddev=True, plot=True, server=True)
    print(pdp_petal_len_cat_std)
if __name__ == '__main__':
    pyunit_utils.standalone_test(partial_plot_test)
else:
    partial_plot_test()
import os
from tests_pytorch.helpers.advanced_models import ParityModuleMNIST, ParityModuleRNN
from parity_pytorch.measure import measure_loops
NUM_EPOCHS = 20
NUM_RUNS = 50
MODEL_CLASSES = (ParityModuleRNN, ParityModuleMNIST)
PATH_HERE = os.path.dirname(__file__)
FIGURE_EXTENSION = '.png'

def _main():
    if False:
        for i in range(10):
            print('nop')
    import matplotlib.pylab as plt
    import pandas as pd
    (fig, axarr) = plt.subplots(nrows=len(MODEL_CLASSES))
    for (i, cls_model) in enumerate(MODEL_CLASSES):
        path_csv = os.path.join(PATH_HERE, f'dump-times_{cls_model.__name__}.csv')
        if os.path.isfile(path_csv):
            df_time = pd.read_csv(path_csv, index_col=0)
        else:
            vanilla = measure_loops(cls_model, kind='Vanilla PT', num_epochs=NUM_EPOCHS, num_runs=NUM_RUNS)
            lightning = measure_loops(cls_model, kind='PT Lightning', num_epochs=NUM_EPOCHS, num_runs=NUM_RUNS)
            df_time = pd.DataFrame({'vanilla PT': vanilla['durations'][1:], 'PT Lightning': lightning['durations'][1:]})
            df_time /= NUM_RUNS
            df_time.to_csv(os.path.join(PATH_HERE, f'dump-times_{cls_model.__name__}.csv'))
        df_time.plot.hist(ax=axarr[i], bins=20, alpha=0.5, title=cls_model.__name__, legend=True, grid=True)
        axarr[i].set(xlabel='time [seconds]')
    path_fig = os.path.join(PATH_HERE, f'figure-parity-times{FIGURE_EXTENSION}')
    fig.tight_layout()
    fig.savefig(path_fig)
if __name__ == '__main__':
    _main()
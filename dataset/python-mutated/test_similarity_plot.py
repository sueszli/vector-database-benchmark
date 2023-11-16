import os
from utils_cv.similarity.data import comparative_set_builder
from utils_cv.similarity.plot import plot_comparative_set, plot_distances, plot_recalls, plot_ranks_distribution

def test_plot_distances(tiny_ic_data_path):
    if False:
        return 10
    im_root_path = os.path.join(tiny_ic_data_path, 'can')
    im_paths = [os.path.join(im_root_path, s) for s in os.listdir(im_root_path)[:3]]
    distances = [(im_path, 1.0) for im_path in im_paths]
    plot_distances(distances, num_rows=1, num_cols=7, figsize=(15, 5))

def test_plot_comparative_set(tiny_ic_databunch):
    if False:
        print('Hello World!')
    comparative_sets = comparative_set_builder(tiny_ic_databunch.valid_ds, num_sets=2, num_negatives=50)
    plot_comparative_set(comparative_sets[1])

def test_plot_recalls():
    if False:
        return 10
    ranks = [1, 2, 3, 2, 1, 5, 3, 5, 4]
    plot_recalls(ranks)

def test_plot_ranks_distribution():
    if False:
        print('Hello World!')
    ranks = [1, 2, 3, 2, 1, 5, 3, 5, 4]
    plot_ranks_distribution(ranks)
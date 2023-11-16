"""
Plot tests.

You must run these tests interactively as follows::

    pytest tests/intensive/plot_tests.py -s -k <test_case>

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import random
import unittest
import numpy as np
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

def test_scatterplot():
    if False:
        print('Hello World!')
    _test_scatterplot('plotly')

def test_scatterplot_mpl():
    if False:
        while True:
            i = 10
    _test_scatterplot('matplotlib')

def _test_scatterplot(backend):
    if False:
        for i in range(10):
            print('nop')
    dataset = foz.load_zoo_dataset('quickstart')
    points = np.random.randn(len(dataset), 2)
    labels = dataset.values('uniqueness')
    sizes = dataset.values(F('ground_truth.detections').length())
    plot = fo.scatterplot(points=points, labels=labels, sizes=sizes, backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_scatterplot_interactive():
    if False:
        i = 10
        return i + 15
    _test_scatterplot_interactive('plotly')

def test_scatterplot_interactive_mpl():
    if False:
        print('Hello World!')
    _test_scatterplot_interactive('matplotlib')

def _test_scatterplot_interactive(backend):
    if False:
        i = 10
        return i + 15
    dataset = foz.load_zoo_dataset('quickstart')
    points = np.random.randn(len(dataset), 2)
    plot = fo.scatterplot(points=points, samples=dataset, labels='uniqueness', sizes=F('ground_truth.detections').length(), backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_scatterplot_by_ids():
    if False:
        return 10
    _test_scatterplot_by_ids('plotly')

def test_scatterplot_by_ids_mpl():
    if False:
        for i in range(10):
            print('nop')
    _test_scatterplot_by_ids('matplotlib')

def _test_scatterplot_by_ids(backend):
    if False:
        i = 10
        return i + 15
    dataset = foz.load_zoo_dataset('quickstart')
    view = dataset.take(51, seed=51)
    points = np.random.randn(len(view), 2)
    ids = view.values('id')
    plot = fo.scatterplot(points=points, samples=dataset, ids=ids, labels='uniqueness', sizes=F('ground_truth.detections').length(), backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_location_scatterplot():
    if False:
        while True:
            i = 10
    _test_location_scatterplot('plotly')

def test_location_scatterplot_mpl():
    if False:
        for i in range(10):
            print('nop')
    _test_location_scatterplot('matplotlib')

def _test_location_scatterplot(backend):
    if False:
        return 10
    dataset = foz.load_zoo_dataset('quickstart-geo')
    fob.compute_uniqueness(dataset)
    locations = dataset.values('location.point.coordinates')
    labels = dataset.values('uniqueness')
    sizes = dataset.values(F('ground_truth.detections').length())
    plot = fo.location_scatterplot(locations=locations, labels=labels, sizes=sizes, backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_location_scatterplot_interactive():
    if False:
        print('Hello World!')
    _test_location_scatterplot_interactive('plotly')

def test_location_scatterplot_interactive_mpl():
    if False:
        while True:
            i = 10
    _test_location_scatterplot_interactive('matplotlib')

def _test_location_scatterplot_interactive(backend):
    if False:
        return 10
    dataset = foz.load_zoo_dataset('quickstart-geo')
    fob.compute_uniqueness(dataset)
    plot = fo.location_scatterplot(locations='location', samples=dataset, labels='uniqueness', sizes=F('ground_truth.detections').length(), backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_location_scatterplot_by_ids():
    if False:
        for i in range(10):
            print('nop')
    _test_location_scatterplot_by_ids('plotly')

def test_location_scatterplot_by_ids_mpl():
    if False:
        return 10
    _test_location_scatterplot_by_ids('matplotlib')

def _test_location_scatterplot_by_ids(backend):
    if False:
        return 10
    dataset = foz.load_zoo_dataset('quickstart-geo')
    view = dataset.take(51, seed=51)
    ids = view.values('id')
    plot = fo.location_scatterplot(samples=dataset, ids=ids, labels='uniqueness', sizes=F('ground_truth.detections').length(), backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_plot_regressions():
    if False:
        print('Hello World!')
    _test_plot_regressions('plotly')

def test_plot_regressions_mpl():
    if False:
        return 10
    _test_plot_regressions('matplotlib')

def _test_plot_regressions(backend):
    if False:
        for i in range(10):
            print('nop')
    dataset = foz.load_zoo_dataset('quickstart').select_fields().clone()
    for (idx, sample) in enumerate(dataset, 1):
        ytrue = random.random() * idx
        ypred = ytrue + np.random.randn() * np.sqrt(ytrue)
        sample['ground_truth'] = fo.Regression(value=ytrue)
        sample['predictions'] = fo.Regression(value=ypred, confidence=random.random())
        sample['weather'] = random.choice(['sunny', 'cloudy', 'rainy'])
        sample.save()
    results = dataset.evaluate_regressions('predictions', gt_field='ground_truth', eval_key='eval')
    plot = results.plot_results(labels='weather', sizes='predictions.confidence', backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_lines():
    if False:
        i = 10
        return i + 15
    _test_lines('plotly')

def test_lines_mpl():
    if False:
        i = 10
        return i + 15
    _test_lines('matplotlib')

def _test_lines(backend):
    if False:
        while True:
            i = 10
    dataset = foz.load_zoo_dataset('quickstart-video')
    view = dataset.filter_labels('frames.detections', F('label') == 'vehicle')
    plot = fo.lines(x='frames.frame_number', y=F('frames.detections.detections').length(), labels='id', samples=view, backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_plot_confusion_matrix():
    if False:
        while True:
            i = 10
    _test_plot_confusion_matrix('plotly')

def test_plot_confusion_matrix_mpl():
    if False:
        print('Hello World!')
    _test_plot_confusion_matrix('matplotlib')

def _test_plot_confusion_matrix(backend):
    if False:
        print('Hello World!')
    dataset = foz.load_zoo_dataset('quickstart')
    results = dataset.evaluate_detections('predictions', gt_field='ground_truth', classwise=False, eval_key='eval')
    counts = dataset.count_values('ground_truth.detections.label')
    classes = sorted(counts, key=counts.get, reverse=True)[:10]
    plot = results.plot_confusion_matrix(classes=classes, backend=backend)
    plot.show()
    input('Press enter to continue...')

def test_view_plot():
    if False:
        while True:
            i = 10
    dataset = foz.load_zoo_dataset('quickstart')
    dataset.compute_metadata()
    plot1 = fo.NumericalHistogram(F('metadata.size_bytes') / 1024, bins=50, xlabel='image size (KB)', init_view=dataset)
    plot1.show()
    plot2 = fo.NumericalHistogram('predictions.detections.confidence', bins=50, range=[0, 1], init_view=dataset)
    plot2.show()
    plot3 = fo.CategoricalHistogram('ground_truth.detections.label', order='frequency', init_view=dataset)
    plot3.show()
    plot4 = fo.CategoricalHistogram('predictions.detections.label', order='frequency', init_view=dataset)
    plot4.show()
    input('Press enter to continue...')
    plot = fo.ViewGrid([plot1, plot2, plot3, plot4], init_view=dataset)
    plot.show()
    input('Press enter to continue...')
    view = dataset.take(51)
    plot.update_view(view)
    plot.show()
    input('Press enter to continue...')
if __name__ == '__main__':
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
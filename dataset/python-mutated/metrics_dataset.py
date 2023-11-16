"""``MetricsDataSet`` saves data to a JSON file using an underlying
filesystem (e.g.: local, S3, GCS). It uses native json to handle the JSON file.
The ``MetricsDataSet`` is part of Kedro Experiment Tracking. The dataset is versioned by default
and only takes metrics of numeric values.
"""
import json
from typing import Dict, NoReturn
from kedro.extras.datasets.json import JSONDataSet
from kedro.io.core import DatasetError, get_filepath_str

class MetricsDataSet(JSONDataSet):
    """``MetricsDataSet`` saves data to a JSON file using an underlying
    filesystem (e.g.: local, S3, GCS). It uses native json to handle the JSON file. The
    ``MetricsDataSet`` is part of Kedro Experiment Tracking. The dataset is write-only,
    it is versioned by default and only takes metrics of numeric values.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_:


    .. code-block:: yaml

        cars:
          type: metrics.MetricsDataSet
          filepath: data/09_tracking/cars.json

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_:
    ::

        >>> from kedro.extras.datasets.tracking import MetricsDataSet
        >>>
        >>> data = {'col1': 1, 'col2': 0.23, 'col3': 0.002}
        >>>
        >>> data_set = MetricsDataSet(filepath="test.json")
        >>> data_set.save(data)

    """
    versioned = True

    def _load(self) -> NoReturn:
        if False:
            return 10
        raise DatasetError(f"Loading not supported for '{self.__class__.__name__}'")

    def _save(self, data: Dict[str, float]) -> None:
        if False:
            i = 10
            return i + 15
        'Converts all values in the data from a ``MetricsDataSet`` to float to make sure\n        they are numeric values which can be displayed in Kedro Viz and then saves the dataset.\n        '
        try:
            for (key, value) in data.items():
                data[key] = float(value)
        except ValueError as exc:
            raise DatasetError(f'The MetricsDataSet expects only numeric values. {exc}') from exc
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
            json.dump(data, fs_file, **self._save_args)
        self._invalidate_cache()
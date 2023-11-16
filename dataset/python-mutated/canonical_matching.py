import io
import os
import time
import dedupe
from benchmarks import common

def get_true_dupes(data):
    if False:
        for i in range(10):
            print('nop')
    (data_1, data_2) = data
    all_data = data_1.copy()
    all_data.update(data_2)
    return common.get_true_dupes(all_data)

def make_report(data, clustering):
    if False:
        i = 10
        return i + 15
    true_dupes = get_true_dupes(data)
    predicted_dupes = set((frozenset(pair) for (pair, _) in clustering))
    return common.Report.from_scores(true_dupes, predicted_dupes)

class Matching:
    settings_file = common.DATASETS_DIR / 'canonical_data_matching_learned_settings'
    data_1_file = common.DATASETS_DIR / 'restaurant-1.csv'
    data_2_file = common.DATASETS_DIR / 'restaurant-2.csv'
    params = [{'threshold': 0.5}, {'threshold': 0.5, 'constraint': 'many-to-one'}]
    param_names = ['kwargs']

    def setup(self, kwargs):
        if False:
            return 10
        data_1 = common.load_data(self.data_1_file)
        data_2 = common.load_data(self.data_2_file)
        self.data = (data_1, data_2)
        training_pairs = dedupe.training_data_link(data_1, data_2, 'unique_id', 5000)
        self.training_pairs_filelike = io.StringIO()
        dedupe.serializer.write_training(training_pairs, self.training_pairs_filelike)
        self.training_pairs_filelike.seek(0)

    def run(self, kwargs, use_settings=False):
        if False:
            while True:
                i = 10
        (data_1, data_2) = self.data
        if use_settings and os.path.exists(self.settings_file):
            with open(self.settings_file, 'rb') as f:
                deduper = dedupe.StaticRecordLink(f)
        else:
            variables = [{'field': 'name', 'type': 'String'}, {'field': 'address', 'type': 'String'}, {'field': 'cuisine', 'type': 'String'}, {'field': 'city', 'type': 'String'}]
            deduper = dedupe.RecordLink(variables)
            deduper.prepare_training(data_1, data_2, training_file=self.training_pairs_filelike, sample_size=10000)
            deduper.train()
            with open(self.settings_file, 'wb') as f:
                deduper.write_settings(f)
        return deduper.join(data_1, data_2, **kwargs)

    def make_report(self, clustering):
        if False:
            while True:
                i = 10
        return make_report(self.data, clustering)

    def time_run(self, kwargs):
        if False:
            print('Hello World!')
        return self.run(kwargs)

    def peakmem_run(self, kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.run(kwargs)

    def track_precision(self, kwargs):
        if False:
            return 10
        return self.make_report(self.run(kwargs)).precision

    def track_recall(self, kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.make_report(self.run(kwargs)).recall

def cli():
    if False:
        for i in range(10):
            print('nop')
    common.configure_logging()
    m = Matching()
    for kwargs in m.params:
        m.setup(kwargs)
        print()
        print(f'running with kwargs: {kwargs}')
        t0 = time.time()
        clustering = m.run(kwargs=kwargs, use_settings=True)
        elapsed = time.time() - t0
        print(m.make_report(clustering))
        print(f'ran in {elapsed} seconds')
if __name__ == '__main__':
    cli()
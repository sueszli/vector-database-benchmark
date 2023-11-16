from paddle.distributed import fleet
cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)

class CriteoDataset(fleet.MultiSlotDataGenerator):

    def generate_sample(self, line):
        if False:
            print('Hello World!')
        '\n        Read the data line by line and process it as a dictionary\n        '

        def reader():
            if False:
                print('Hello World!')
            '\n            This function needs to be implemented by the user, based on data format\n            '
            features = line.rstrip('\n').split('\t')
            dense_feature = []
            for idx in continuous_range_:
                if features[idx] == '':
                    dense_feature.append(0.0)
                else:
                    dense_feature.append((float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])
            label = [int(features[0])]
            feature_name = ['dense_feature']
            feature_name.append('label')
            yield list(zip(feature_name, [label] + [dense_feature]))
        return reader
d = CriteoDataset()
d.run_from_stdin()
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
            return 10
        '\n        Read the data line by line and process it as a dictionary\n        '

        def reader():
            if False:
                for i in range(10):
                    print('nop')
            '\n            This function needs to be implemented by the user, based on data format\n            '
            features = line.rstrip('\n').split('\t')
            feature_name = []
            sparse_feature = []
            for idx in categorical_range_:
                sparse_feature.append([hash(str(idx) + features[idx]) % hash_dim_])
            for idx in categorical_range_:
                feature_name.append('C' + str(idx - 13))
            yield list(zip(feature_name, sparse_feature))
        return reader
d = CriteoDataset()
d.run_from_stdin()
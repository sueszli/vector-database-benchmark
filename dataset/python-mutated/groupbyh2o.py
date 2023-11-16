import vaex
from benchmarks.fixtures import generate_numerical
check = False

class GroupBySetup:
    pretty_name = 'Groupby benchmarks - H2O inspired'
    version = '1'
    params = ([10 ** 7, 5 * 10 ** 7, 10 ** 8],)
    param_names = ['N']

    def setup_cache(self):
        if False:
            print('Hello World!')
        generate_numerical()

    def setup(self, N):
        if False:
            while True:
                i = 10
        df = self.df = vaex.open(generate_numerical())[:N]
        df['id1'] = df['i1_100']
        df['id2'] = df['i1_100']
        df['id3'] = df['i4_1M']
        df['id4'] = df['i1_100']
        df['id5'] = df['i1_100']
        df['id6'] = df['i4_1M']
        df['v1'] = df['i1_10']
        df['v2'] = df['i1_10']
        df['v3'] = df['x4']

class GroupbyH2O(GroupBySetup):

    def time_question_01(self, N):
        if False:
            return 10
        df = self.df.groupby(['id1']).agg({'v1': 'sum'})
        if check:
            chk_sum_cols = ['v1']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_02(self, N):
        if False:
            for i in range(10):
                print('nop')
        df = self.df.groupby(['id1', 'id2']).agg({'v1': 'sum'})
        if check:
            chk_sum_cols = ['v1']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_03(self, N):
        if False:
            print('Hello World!')
        df = self.df.groupby(['id3']).agg({'v1': 'sum', 'v3': 'mean'})
        if check:
            chk_sum_cols = ['v1', 'v3']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_04(self, N):
        if False:
            while True:
                i = 10
        df = self.df.groupby(['id4']).agg({'v1': 'mean', 'v2': 'mean', 'v3': 'mean'})
        if check:
            chk_sum_cols = ['v1', 'v2', 'v3']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_05(self, N):
        if False:
            print('Hello World!')
        df = self.df.groupby(['id6']).agg({'v1': 'sum', 'v2': 'sum', 'v3': 'sum'})
        if check:
            chk_sum_cols = ['v1', 'v2', 'v3']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_07(self, N):
        if False:
            print('Hello World!')
        df = self.df.groupby(['id3']).agg({'v1': 'max', 'v2': 'min'})
        df['range_x_y'] = df['v1'] - df['v2']
        if check:
            chk_sum_cols = ['range_v1_v2']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_10(self, N):
        if False:
            print('Hello World!')
        df = self.df.groupby(['id1', 'id2', 'id3', 'id4', 'id5', 'id6']).agg({'v3': 'sum', 'v1': 'count'})
        if check:
            chk_sum_cols = ['v3', 'v1']
            [df[col].sum() for col in chk_sum_cols]
from benchmarks.fixtures import generate_numerical, generate_strings
import vaex

class IsIn:
    pretty_name = 'Performance of isin'
    version = '1'
    params = ([10 ** 7, 5 * 10 ** 7, 10 ** 8], (1, 10, 100, 1000, 1000000))
    param_names = ['N', 'M']

    def setup_cache(self):
        if False:
            i = 10
            return i + 15
        generate_numerical()
        generate_strings()

    def setup(self, N, M):
        if False:
            print('Hello World!')
        self.df_num = vaex.open(generate_numerical())[:N]
        self.df_str = vaex.open(generate_strings())[:N]

    def time_isin_i8_1M(self, N, M):
        if False:
            for i in range(10):
                print('nop')
        df = self.df_num
        values = df.sample(M)['i8_1M'].values
        df['i8_1M'].isin(values).sum()

    def time_isin_str(self, N, M):
        if False:
            return 10
        df = self.df_str
        values = df.sample(M)['s'].values
        df['s'].isin(values).sum()
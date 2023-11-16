import vaex
from benchmarks.fixtures import generate_strings
from benchmarks.fixtures import generate_numerical

class hashmap:
    pretty_name = 'hashmap benchmarks'
    version = '1'

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        dfs = vaex.open(generate_strings())
        Nmax_strings = 1000000
        self.dfs_small = dfs[:Nmax_strings]
        self.hms = self.dfs_small._hash_map_unique('s')

    def time_strings_create(self):
        if False:
            while True:
                i = 10
        self.dfs_small._hash_map_unique('s')

    def time_strings_keys(self):
        if False:
            for i in range(10):
                print('nop')
        self.hms.keys()
if __name__ == '__main__':
    bench = hashmap()
    bench.setup()
    bench.time_strings_create()
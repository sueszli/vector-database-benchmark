from paddle.distributed import fleet

class MyDataset(fleet.MultiSlotDataGenerator):

    def generate_sample(self, line):
        if False:
            return 10

        def data_iter():
            if False:
                for i in range(10):
                    print('nop')
            elements = line.strip().split()[0:]
            output = [('show', [int(elements[0])]), ('click', [int(elements[1])]), ('slot1', [int(elements[2])])]
            yield output
        return data_iter
if __name__ == '__main__':
    d = MyDataset()
    d.run_from_stdin()
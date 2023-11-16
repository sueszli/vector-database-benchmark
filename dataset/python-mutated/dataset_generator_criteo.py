import logging
from paddle.distributed import fleet
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class Reader(fleet.MultiSlotDataGenerator):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        padding = 0
        sparse_slots = 'click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26'
        self.sparse_slots = sparse_slots.strip().split(' ')
        self.dense_slots = ['dense_feature']
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots + self.dense_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding
        logger.info('pipe init success')

    def line_process(self, line):
        if False:
            while True:
                i = 10
        line = line.strip().split(' ')
        output = [(i, []) for i in self.slots]
        for i in line:
            slot_feasign = i.split(':')
            slot = slot_feasign[0]
            if slot not in self.slots:
                continue
            if slot in self.sparse_slots:
                feasign = int(slot_feasign[1])
            else:
                feasign = float(slot_feasign[1])
            output[self.slot2index[slot]][1].append(feasign)
            self.visit[slot] = True
        for i in self.visit:
            slot = i
            if not self.visit[slot]:
                if i in self.dense_slots:
                    output[self.slot2index[i]][1].extend([self.padding] * self.dense_slots_shape[self.slot2index[i]])
                else:
                    output[self.slot2index[i]][1].extend([self.padding])
            else:
                self.visit[slot] = False
        return output

    def generate_sample(self, line):
        if False:
            return 10
        'Dataset Generator'

        def reader():
            if False:
                for i in range(10):
                    print('nop')
            output_dict = self.line_process(line)
            yield output_dict
        return reader
if __name__ == '__main__':
    r = Reader()
    r.init()
    r.run_from_stdin()
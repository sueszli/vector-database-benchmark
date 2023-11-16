import logging
from paddle.distributed import fleet
logging.basicConfig()
logger = logging.getLogger('paddle')
logger.setLevel(logging.INFO)

class DatasetSimnetReader(fleet.MultiSlotDataGenerator):

    def generate_sample(self, line):
        if False:
            for i in range(10):
                print('nop')
        pass
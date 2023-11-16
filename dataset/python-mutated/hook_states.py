class BasicHookState:

    def __init__(self, cref, process_group):
        if False:
            i = 10
            return i + 15
        '\n        A class that holds state information that is needed by the communication hook\n        during the training algorithm.\n        Args:\n            cref (DdpTrainer): reference to the self keyword of the trainer instance\n            process_group (ProcessGroup): distributed process group\n        '
        self.cref = cref
        self.process_group = process_group
        self.batch_number = -1

    def get_key(self, bucket_index):
        if False:
            i = 10
            return i + 15
        '\n        A method that returns an encoded key that represents the current batch and\n        bucket index.\n        Args:\n            bucket_index (int): index of the bucket being processed in backward\n        '
        return f'{self.batch_number},{bucket_index}'

    def next_batch(self):
        if False:
            return 10
        '\n        A method that increments batch_number by 1.\n        '
        self.batch_number += 1
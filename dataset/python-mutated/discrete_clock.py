from datetime import datetime
from tribler.core.components.metadata_store.db.serialization import time2int

class DiscreteClock:

    def __init__(self):
        if False:
            return 10
        self.clock = time2int(datetime.utcnow()) * 1000

    def tick(self):
        if False:
            return 10
        self.clock += 1
        return self.clock
clock = DiscreteClock()
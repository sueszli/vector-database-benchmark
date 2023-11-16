from eventsourcing.domain import Aggregate, event
from loguru import logger

class Person(Aggregate):

    @event('Created')
    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.life = []

    @event('LifeHappened')
    def deal_with_life(self, some_event):
        if False:
            return 10
        self.life.append(some_event)

@logger.catch
def main():
    if False:
        while True:
            i = 10
    ben = Person()
    assert ben.life == []
    ben.deal_with_life('school')
    assert ben.life == ['school']
    assert len(ben.collect_events()) == 2
main()
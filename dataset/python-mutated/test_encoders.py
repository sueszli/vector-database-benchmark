import pytest
import dramatiq

@pytest.fixture
def pickle_encoder():
    if False:
        return 10
    old_encoder = dramatiq.get_encoder()
    new_encoder = dramatiq.PickleEncoder()
    dramatiq.set_encoder(new_encoder)
    yield new_encoder
    dramatiq.set_encoder(old_encoder)

def test_set_encoder_sets_the_global_encoder(pickle_encoder):
    if False:
        i = 10
        return i + 15
    encoder = dramatiq.get_encoder()
    assert encoder == pickle_encoder

def test_pickle_encoder(pickle_encoder, stub_broker, stub_worker):
    if False:
        return 10
    db = []

    @dramatiq.actor
    def add_value(x):
        if False:
            print('Hello World!')
        db.append(x)
    add_value.send(1)
    stub_broker.join(add_value.queue_name)
    stub_worker.join()
    assert db == [1]
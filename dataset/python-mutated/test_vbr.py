import pytest
from apps.wasm.vbr import VerificationResult, Actor, BucketVerifier, NotAllowedError, UnknownActorError, MissingResultsError, AlreadyFinished
actors = [Actor(str(i)) for i in range(20)]

class SimpleComparator:

    def __call__(self, x, y):
        if False:
            while True:
                i = 10
        return x == y

def verdicts_to_dict(verdicts):
    if False:
        for i in range(10):
            print('nop')
    if verdicts is None:
        return None
    d = {}
    for (actor, _, verdict) in verdicts:
        d[actor] = verdict
    return d

def verdict_is_undecided(verdict):
    if False:
        while True:
            i = 10
    return all([v == VerificationResult.UNDECIDED for (_, _, v) in verdict])

def test_r0():
    if False:
        while True:
            i = 10
    verifier = BucketVerifier(0, SimpleComparator(), 0)
    assert verifier.normal_actor_count == 1
    assert verifier.referee_count == 0
    verifier.add_actor(actors[1])
    with pytest.raises(NotAllowedError):
        verifier.add_actor(actors[1])
    with pytest.raises(UnknownActorError):
        verifier.add_result(actors[2], 2)
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    verifier.add_result(actors[1], 1)
    verdicts = verifier.get_verdicts()
    assert len(verdicts) == 1
    d = verdicts_to_dict(verdicts)
    assert d[actors[1]] == VerificationResult.SUCCESS

def test_r1_equal():
    if False:
        for i in range(10):
            print('nop')
    verifier = BucketVerifier(1, SimpleComparator(), 0)
    assert verifier.normal_actor_count == 2
    assert verifier.referee_count == 0
    verifier.add_actor(actors[1])
    with pytest.raises(NotAllowedError):
        verifier.add_actor(actors[1])
    with pytest.raises(UnknownActorError):
        verifier.add_result(actors[2], 1)
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    verifier.add_result(actors[1], 1)
    verifier.add_actor(actors[2])
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    verifier.add_result(actors[2], 1)
    verdicts = verifier.get_verdicts()
    assert verdicts is not None
    assert len(verdicts) == 2
    d = verdicts_to_dict(verdicts)
    assert d[actors[1]] == VerificationResult.SUCCESS
    assert d[actors[2]] == VerificationResult.SUCCESS

def test_r1_different_no_referee():
    if False:
        for i in range(10):
            print('nop')
    verifier = BucketVerifier(1, SimpleComparator(), 0)
    assert verifier.normal_actor_count == 2
    assert verifier.referee_count == 0
    assert verifier.majority == 2
    verifier.add_actor(actors[1])
    with pytest.raises(NotAllowedError):
        verifier.add_actor(actors[1])
    with pytest.raises(UnknownActorError):
        verifier.add_result(actors[2], 1)
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    verifier.add_result(actors[1], 1)
    verifier.add_actor(actors[2])
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    verifier.add_result(actors[2], 2)
    verdicts = verifier.get_verdicts()
    assert verdicts is not None
    assert len(verdicts) == 2
    d = verdicts_to_dict(verdicts)
    assert d[actors[1]] == VerificationResult.UNDECIDED
    assert d[actors[2]] == VerificationResult.UNDECIDED

def test_r1_different_one_referee():
    if False:
        while True:
            i = 10
    verifier = BucketVerifier(1, SimpleComparator(), referee_count=1)
    assert verifier.normal_actor_count == 2
    assert verifier.referee_count == 1
    assert verifier.majority == 2
    assert verifier.more_actors_needed
    verifier.add_actor(actors[1])
    with pytest.raises(NotAllowedError):
        verifier.add_actor(actors[1])
    with pytest.raises(UnknownActorError):
        verifier.add_result(actors[2], 1)
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    assert verifier.more_actors_needed
    verifier.add_result(actors[1], 1)
    assert verifier.more_actors_needed
    verifier.add_actor(actors[2])
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    verifier.add_result(actors[2], 2)
    assert verifier.more_actors_needed
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    assert verifier.more_actors_needed
    verifier.add_actor(actors[3])
    assert not verifier.more_actors_needed
    verifier.add_result(actors[3], 1)
    assert len(verifier.results) == 3
    verdicts = verifier.get_verdicts()
    assert verdicts is not None
    assert len(verdicts) == 3
    d = verdicts_to_dict(verdicts)
    assert d[actors[1]] == VerificationResult.SUCCESS
    assert d[actors[2]] == VerificationResult.FAIL
    assert d[actors[3]] == VerificationResult.SUCCESS

def test_r1_timeout_no_referee():
    if False:
        for i in range(10):
            print('nop')
    verifier = BucketVerifier(1, SimpleComparator(), 0)
    assert verifier.normal_actor_count == 2
    assert verifier.referee_count == 0
    assert verifier.majority == 2
    verifier.add_actor(actors[1])
    with pytest.raises(NotAllowedError):
        verifier.add_actor(actors[1])
    with pytest.raises(UnknownActorError):
        verifier.add_result(actors[2], 1)
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    verifier.add_result(actors[1], 1)
    verifier.add_actor(actors[2])
    verdicts = verifier.get_verdicts()
    assert verdicts is None
    verifier.add_result(actors[2], 2)
    verdicts = verifier.get_verdicts()
    assert verdicts is not None
    assert verdict_is_undecided(verdicts)

def test_r0_sole_timeout():
    if False:
        i = 10
        return i + 15
    verifier = BucketVerifier(0, SimpleComparator(), 0)
    verifier.add_actor(actors[0])
    verifier.add_result(actors[0], None)
    verdicts = verifier.get_verdicts()
    assert verdicts

def test_r1_sole_timeout():
    if False:
        print('Hello World!')
    verifier = BucketVerifier(1, SimpleComparator(), 0)
    verifier.add_actor(actors[0])
    verifier.add_actor(actors[1])
    verifier.add_result(actors[0], None)
    verifier.add_result(actors[1], None)
    verdicts = verifier.get_verdicts()
    assert verdicts

def test_r0_missing_results_error():
    if False:
        print('Hello World!')
    verifier = BucketVerifier(0, SimpleComparator(), 0)
    verifier.add_actor(actors[1])
    with pytest.raises(MissingResultsError):
        verifier.add_actor(actors[2])

def test_r1_missing_results_error():
    if False:
        while True:
            i = 10
    verifier = BucketVerifier(1, SimpleComparator(), 0)
    verifier.add_actor(actors[1])
    verifier.add_actor(actors[2])
    with pytest.raises(MissingResultsError):
        verifier.add_actor(actors[3])
    verifier.add_result(actors[1], 1)
    with pytest.raises(MissingResultsError):
        verifier.add_actor(actors[3])

def test_r1_result_already_added_value_error():
    if False:
        i = 10
        return i + 15
    verifier = BucketVerifier(1, SimpleComparator(), 0)
    verifier.add_actor(actors[1])
    verifier.add_actor(actors[2])
    verifier.add_result(actors[1], 1)
    with pytest.raises(ValueError):
        verifier.add_result(actors[1], 1)

def test_r1_with_referee_all_different():
    if False:
        print('Hello World!')
    verifier = BucketVerifier(1, SimpleComparator(), 1)
    verifier.add_actor(actors[1])
    verifier.add_actor(actors[2])
    verifier.add_result(actors[1], 1)
    verifier.add_result(actors[2], 2)
    verifier.add_actor(actors[3])
    verifier.add_result(actors[3], 3)
    assert verifier.get_verdicts() is not None

def test_r1_with_referee_all_different_with_none():
    if False:
        return 10
    verifier = BucketVerifier(1, SimpleComparator(), 1)
    verifier.add_actor(actors[1])
    verifier.add_actor(actors[2])
    verifier.add_result(actors[1], 1)
    verifier.add_result(actors[2], None)
    verifier.add_actor(actors[3])
    verifier.add_result(actors[3], 3)
    assert verifier.get_verdicts() is not None

def test_r1_with_referee_none_result():
    if False:
        return 10
    verifier = BucketVerifier(1, SimpleComparator(), 1)
    verifier.add_actor(actors[1])
    verifier.add_actor(actors[2])
    verifier.add_result(actors[1], None)
    verifier.add_result(actors[2], 2)
    verifier.add_actor(actors[3])
    verifier.add_result(actors[3], None)
    assert verifier.get_verdicts() is not None

def test_r1_actor_removal():
    if False:
        return 10
    verifier = BucketVerifier(1, SimpleComparator(), 1)
    verifier.add_actor(actors[1])
    verifier.add_actor(actors[2])
    verifier.remove_actor(actors[2])
    verifier.add_actor(actors[3])
    verifier.add_result(actors[1], 1)
    verifier.add_result(actors[3], 1)
    verdicts = verifier.get_verdicts()
    assert verdicts is not None
    for (actor, _, verdict) in verdicts:
        assert actor in (actors[1], actors[3])
        assert verdict == VerificationResult.SUCCESS

def test_r1_actor_removal2():
    if False:
        i = 10
        return i + 15
    verifier = BucketVerifier(1, SimpleComparator(), 1)
    verifier.add_actor(actors[1])
    verifier.add_actor(actors[2])
    verifier.remove_actor(actors[1])
    verifier.remove_actor(actors[2])
    verifier.add_actor(actors[3])
    verifier.add_actor(actors[4])
    verifier.add_result(actors[3], 1)
    verifier.add_result(actors[4], 1)
    verdicts = verifier.get_verdicts()
    assert verdicts is not None
    for (actor, _, verdict) in verdicts:
        assert actor in (actors[3], actors[4])
        assert verdict == VerificationResult.SUCCESS

def test_r1_actor_remove_already_finished():
    if False:
        i = 10
        return i + 15
    verifier = BucketVerifier(1, SimpleComparator(), 1)
    verifier.add_actor(actors[1])
    verifier.add_result(actors[1], 1)
    verifier.add_actor(actors[2])
    with pytest.raises(AlreadyFinished):
        verifier.remove_actor(actors[1])
    verifier.add_result(actors[2], 1)
    verdicts = verifier.get_verdicts()
    for (actor, _, verdict) in verdicts:
        assert actor in (actors[1], actors[2])
        assert verdict == VerificationResult.SUCCESS

def test_r1_actor_removal_raises_finished():
    if False:
        while True:
            i = 10
    verifier = BucketVerifier(1, SimpleComparator(), 1)
    verifier.add_actor(actors[1])
    verifier.add_actor(actors[2])
    verifier.add_result(actors[1], 1)
    verifier.add_result(actors[2], 1)
    verdicts = verifier.get_verdicts()
    assert verdicts is not None
    with pytest.raises(AlreadyFinished):
        verifier.remove_actor(actors[1])
    for (_, _, verdict) in verdicts:
        assert verdict == VerificationResult.SUCCESS
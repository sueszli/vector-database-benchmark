import pytest

@pytest.mark.skip_framework('tensorflow1', 'tensorflow2v1', 'keras', 'non_dl_frameworks', 'mxnet', 'kerastf')
def test_attack_repr(image_dl_estimator):
    if False:
        i = 10
        return i + 15
    from art.attacks.evasion import ProjectedGradientDescentNumpy
    (classifier, _) = image_dl_estimator(from_logits=True)
    attack = ProjectedGradientDescentNumpy(estimator=classifier, targeted=True, decay=0.5)
    print(repr(attack))
    assert repr(attack) == 'ProjectedGradientDescentNumpy(norm=inf, eps=0.3, eps_step=0.1, targeted=True, ' + 'num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=0.5, ' + 'max_iter=100, random_eps=False, verbose=True, )'
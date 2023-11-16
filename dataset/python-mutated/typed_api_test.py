from typedapi import ensure_api_is_typed
import autokeras
HELP_MESSAGE = 'You can also take a look at this issue:\nhttps://github.com/keras-team/autokeras/issues/918'
EXCEPTION_LIST = [autokeras.BayesianOptimization, autokeras.CastToFloat32, autokeras.ExpandLastDim, autokeras.RandomSearch]

def test_api_surface_is_typed():
    if False:
        for i in range(10):
            print('nop')
    ensure_api_is_typed([autokeras], EXCEPTION_LIST, init_only=True, additional_message=HELP_MESSAGE)
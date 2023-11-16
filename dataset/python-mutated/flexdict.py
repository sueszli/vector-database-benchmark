import gymnasium as gym
from ray.rllib.utils.annotations import PublicAPI

@PublicAPI
class FlexDict(gym.spaces.Dict):
    """Gym Dictionary with arbitrary keys updatable after instantiation

    Example:
       space = FlexDict({})
       space['key'] = spaces.Box(4,)
    See also: documentation for gym.spaces.Dict
    """

    def __init__(self, spaces=None, **spaces_kwargs):
        if False:
            for i in range(10):
                print('nop')
        err = 'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
        assert spaces is None or not spaces_kwargs, err
        if spaces is None:
            spaces = spaces_kwargs
        for space in spaces.values():
            self.assertSpace(space)
        super().__init__(spaces=spaces)

    def assertSpace(self, space):
        if False:
            for i in range(10):
                print('nop')
        err = 'Values of the dict should be instances of gym.Space'
        assert issubclass(type(space), gym.spaces.Space), err

    def sample(self):
        if False:
            while True:
                i = 10
        return {k: space.sample() for (k, space) in self.spaces.items()}

    def __getitem__(self, key):
        if False:
            return 10
        return self.spaces[key]

    def __setitem__(self, key, space):
        if False:
            print('Hello World!')
        self.assertSpace(space)
        self.spaces[key] = space

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'FlexDict(' + ', '.join([str(k) + ':' + str(s) for (k, s) in self.spaces.items()]) + ')'
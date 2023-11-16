from pyro.contrib.funsor.handlers.primitives import to_data
from pyro.poutine.replay_messenger import ReplayMessenger as OrigReplayMessenger

class ReplayMessenger(OrigReplayMessenger):
    """
    This version of :class:`~ReplayMessenger` is almost identical to the original version,
    except that it calls :func:`~pyro.contrib.funsor.to_data` on the replayed funsor values.
    This may result in different unpacked shapes, but should produce correct allocations.
    """

    def _pyro_sample(self, msg):
        if False:
            for i in range(10):
                print('nop')
        name = msg['name']
        msg['replay_active'] = True
        if self.trace is None:
            return
        if name in self.trace:
            guide_msg = self.trace.nodes[name]
            msg['funsor'] = {} if 'funsor' not in msg else msg['funsor']
            if guide_msg['type'] != 'sample':
                raise RuntimeError('site {} must be sample in trace'.format(name))
            if guide_msg.get('funsor', {}).get('value', None) is not None:
                msg['value'] = to_data(guide_msg['funsor']['value'])
            else:
                msg['value'] = guide_msg['value']
            msg['infer'] = guide_msg['infer']
            msg['done'] = True
            msg['replay_skipped'] = False
        else:
            msg['replay_skipped'] = msg.get('replay_skipped', True)
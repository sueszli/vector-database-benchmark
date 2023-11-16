"""
Manage the Thorium complex event reaction system
"""
import salt.thorium

def start(grains=False, grain_keys=None, pillar=False, pillar_keys=None):
    if False:
        return 10
    '\n    Execute the Thorium runtime\n    '
    state = salt.thorium.ThorState(__opts__, grains, grain_keys, pillar, pillar_keys)
    state.start_runtime()
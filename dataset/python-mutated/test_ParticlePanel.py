import pytest
Pmw = pytest.importorskip('Pmw')
from direct.tkpanels.ParticlePanel import ParticlePanel

def test_ParticlePanel(base, tk_toplevel):
    if False:
        return 10
    root = Pmw.initialise()
    pp = ParticlePanel()
    base.pp = pp
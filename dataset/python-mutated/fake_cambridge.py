"""
Fake Cambridge device (20 qubit).
"""
import os
from qiskit.providers.fake_provider import fake_qasm_backend, fake_backend

class FakeCambridgeV2(fake_backend.FakeBackendV2):
    """A fake Cambridge backend.

    .. code-block:: text

                  00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
                  ↕                    ↕
                  05                  06
                  ↕                    ↕
        07 ↔ 08 ↔ 09 ↔ 10 ↔ 11 ↔ 12 ↔ 13 ↔ 14 ↔ 15
        ↕                   ↕                    ↕
        16                  17                  18
        ↕                   ↕                    ↕
        19 ↔ 20 ↔ 21 ↔ 22 ↔ 23 ↔ 24 ↔ 25 ↔ 26 ↔ 27
    """
    dirname = os.path.dirname(__file__)
    conf_filename = 'conf_cambridge.json'
    props_filename = 'props_cambridge.json'
    backend_name = 'fake_cambridge'

class FakeCambridge(fake_qasm_backend.FakeQasmBackend):
    """A fake Cambridge backend.

    .. code-block:: text

                  00 ↔ 01 ↔ 02 ↔ 03 ↔ 04
                  ↕                    ↕
                  05                  06
                  ↕                    ↕
        07 ↔ 08 ↔ 09 ↔ 10 ↔ 11 ↔ 12 ↔ 13 ↔ 14 ↔ 15
        ↕                   ↕                    ↕
        16                  17                  18
        ↕                   ↕                    ↕
        19 ↔ 20 ↔ 21 ↔ 22 ↔ 23 ↔ 24 ↔ 25 ↔ 26 ↔ 27
    """
    dirname = os.path.dirname(__file__)
    conf_filename = 'conf_cambridge.json'
    props_filename = 'props_cambridge.json'
    backend_name = 'fake_cambridge'

class FakeCambridgeAlternativeBasis(FakeCambridge):
    """A fake Cambridge backend with alternate 1q basis gates."""
    props_filename = 'props_cambridge_alt.json'

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self._configuration.basis_gates = ['u', 'sx', 'p', 'cx', 'id']
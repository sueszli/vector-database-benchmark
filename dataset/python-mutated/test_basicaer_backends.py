"""BasicAer Backends Test."""
from qiskit import BasicAer
from qiskit.providers.basicaer import BasicAerProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.test import providers

class TestBasicAerBackends(providers.ProviderTestCase):
    """Qiskit BasicAer Backends (Object) Tests."""
    provider_cls = BasicAerProvider
    backend_name = 'qasm_simulator'

    def test_deprecated(self):
        if False:
            return 10
        'Test that deprecated names map the same backends as the new names.'

        def _get_first_available_backend(provider, backend_names):
            if False:
                return 10
            'Gets the first available backend.'
            if isinstance(backend_names, str):
                backend_names = [backend_names]
            for backend_name in backend_names:
                try:
                    return provider.get_backend(backend_name).name()
                except QiskitBackendNotFoundError:
                    pass
            return None
        deprecated_names = BasicAer._deprecated_backend_names()
        for (oldname, newname) in deprecated_names.items():
            expected = "WARNING:qiskit.providers.providerutils:Backend '%s' is deprecated. Use '%s'." % (oldname, newname)
            with self.subTest(oldname=oldname, newname=newname):
                with self.assertLogs('qiskit.providers.providerutils', level='WARNING') as context:
                    resolved_newname = _get_first_available_backend(BasicAer, newname)
                    real_backend = BasicAer.get_backend(resolved_newname)
                    self.assertEqual(BasicAer.backends(oldname)[0], real_backend)
                self.assertEqual(context.output, [expected])

    def test_aliases_fail(self):
        if False:
            i = 10
            return i + 15
        'Test a failing backend lookup.'
        self.assertRaises(QiskitBackendNotFoundError, BasicAer.get_backend, 'bad_name')

    def test_aliases_return_empty_list(self):
        if False:
            return 10
        'Test backends() return an empty list if name is unknown.'
        self.assertEqual(BasicAer.backends('bad_name'), [])
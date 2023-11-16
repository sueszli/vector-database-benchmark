import json
from unittest import TestCase
from golem.diag.service import DiagnosticsOutputFormat
from golem.diag.vm import VMDiagnosticsProvider

class TestVMDiagnosticProvider(TestCase):

    def test_format_outputs(self):
        if False:
            print('Hello World!')
        provider = VMDiagnosticsProvider()
        diag = provider.get_diagnostics(DiagnosticsOutputFormat.string)
        json.dumps(diag)
        diag = provider.get_diagnostics(DiagnosticsOutputFormat.json)
        json.dumps(diag)
        diag = provider.get_diagnostics(DiagnosticsOutputFormat.data)
        json.dumps(diag)
        with self.assertRaises(ValueError):
            provider.get_diagnostics('UNKNOWN FORMAT')
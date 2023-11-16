from .integ_base import IntegBase, TelemetryServer

class TestTelemetryContract(IntegBase):
    """
    Validates the basic tenets/contract Telemetry module needs to adhere to
    """

    def test_must_not_send_metrics_if_disabled_using_envvar(self):
        if False:
            while True:
                i = 10
        '\n        No metrics should be sent if "Enabled via Config file but Disabled via Envvar"\n        '
        self.set_config(telemetry_enabled=True)
        with TelemetryServer() as server:
            process = self.run_cmd(optout_envvar_value='0')
            process.communicate()
            self.assertEqual(process.returncode, 0, 'Command should successfully complete')
            all_requests = server.get_all_requests()
            self.assertEqual(0, len(all_requests), 'No metrics should be sent')
            process = self.run_cmd()
            process.communicate()
            self.assertEqual(process.returncode, 0, 'Command should successfully complete')
            all_requests = server.get_all_requests()
            self.assertEqual(2, len(all_requests), 'Command run and event metrics should be sent')

    def test_must_send_metrics_if_enabled_via_envvar(self):
        if False:
            print('Hello World!')
        '\n        Metrics should be sent if "Disabled via config file but Enabled via Envvar"\n        '
        self.set_config(telemetry_enabled=False)
        with TelemetryServer() as server:
            process = self.run_cmd()
            process.communicate()
            self.assertEqual(process.returncode, 0, 'Command should successfully complete')
            all_requests = server.get_all_requests()
            self.assertEqual(0, len(all_requests), 'No metric should be sent')
            process = self.run_cmd(optout_envvar_value='1')
            process.communicate()
            self.assertEqual(process.returncode, 0, 'Command should successfully complete')
            all_requests = server.get_all_requests()
            self.assertEqual(2, len(all_requests), 'Command run and event metrics must be sent')

    def test_must_not_crash_when_offline(self):
        if False:
            return 10
        '\n        Must not crash the process if internet is not available\n        '
        self.set_config(telemetry_enabled=True)
        process = self.run_cmd()
        process.communicate()
        self.assertEqual(process.returncode, 0, 'Command should successfully complete')
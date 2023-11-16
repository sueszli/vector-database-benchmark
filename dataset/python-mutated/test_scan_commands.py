from sslyze.plugins.scan_commands import ScanCommandsRepository

class TestScanCommands:

    def test_all_commands_are_implemented(self):
        if False:
            for i in range(10):
                print('nop')
        for scan_command in ScanCommandsRepository.get_all_scan_commands():
            assert ScanCommandsRepository.get_implementation_cls(scan_command)
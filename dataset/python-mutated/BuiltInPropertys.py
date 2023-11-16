from robot.libraries.BuiltIn import BuiltIn

class BuiltInPropertys:

    def __init__(self, dry_run=False):
        if False:
            for i in range(10):
                print('nop')
        assert BuiltIn().robot_running is True
        assert BuiltIn().dry_run_active is dry_run

    def keyword(self):
        if False:
            while True:
                i = 10
        assert BuiltIn().robot_running is True
        assert BuiltIn().dry_run_active is False
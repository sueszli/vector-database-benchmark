from __future__ import annotations
from tests.test_utils.system_tests_class import SystemTest

class Terraform(SystemTest):
    TERRAFORM_DIR: str

    def setup_method(self) -> None:
        if False:
            i = 10
            return i + 15
        self.execute_cmd(['terraform', 'init', '-input=false', self.TERRAFORM_DIR])
        self.execute_cmd(['terraform', 'plan', '-input=false', self.TERRAFORM_DIR])
        self.execute_cmd(['terraform', 'apply', '-input=false', '-auto-approve', self.TERRAFORM_DIR])

    def get_tf_output(self, name):
        if False:
            print('Hello World!')
        return ''.join(self.check_output(['terraform', 'output', name]).decode('utf-8').splitlines())

    def teardown_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.execute_cmd(['terraform', 'plan', '-destroy', '-input=false', self.TERRAFORM_DIR])
        self.execute_cmd(['terraform', 'destroy', '-input=false', '-auto-approve', self.TERRAFORM_DIR])
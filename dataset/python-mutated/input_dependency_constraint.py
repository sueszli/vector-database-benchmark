from enum import Enum
from pyflink.java_gateway import get_gateway
__all__ = ['InputDependencyConstraint']

class InputDependencyConstraint(Enum):
    """
    This constraint indicates when a task should be scheduled considering its inputs status.

    :data:`ANY`:

    Schedule the task if any input is consumable.

    :data:`ALL`:

    Schedule the task if all the inputs are consumable.
    """
    ANY = 0
    ALL = 1

    @staticmethod
    def _from_j_input_dependency_constraint(j_input_dependency_constraint) -> 'InputDependencyConstraint':
        if False:
            print('Hello World!')
        return InputDependencyConstraint[j_input_dependency_constraint.name()]

    def _to_j_input_dependency_constraint(self):
        if False:
            i = 10
            return i + 15
        gateway = get_gateway()
        JInputDependencyConstraint = gateway.jvm.org.apache.flink.api.common.InputDependencyConstraint
        return getattr(JInputDependencyConstraint, self.name)
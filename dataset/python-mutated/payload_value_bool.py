from typing import Final
from localstack.services.stepfunctions.asl.component.common.payload.payloadvalue.payloadvaluelit.payload_value_lit import PayloadValueLit

class PayloadValueBool(PayloadValueLit):

    def __init__(self, val: bool):
        if False:
            i = 10
            return i + 15
        self.val: Final[bool] = val
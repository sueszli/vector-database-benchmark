from localstack.services.stepfunctions.asl.component.common.payload.payloadvalue.payloadvaluelit.payload_value_lit import PayloadValueLit

class PayloadValueFloat(PayloadValueLit):
    val: float

    def __init__(self, val: float):
        if False:
            i = 10
            return i + 15
        super().__init__(val=val)
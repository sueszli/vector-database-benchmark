from localstack.testing.pytest import markers
from tests.aws.services.stepfunctions.templates.intrinsicfunctions.intrinsic_functions_templates import IntrinsicFunctionTemplate as IFT
from tests.aws.services.stepfunctions.v2.intrinsic_functions.utils import create_and_test_on_inputs

@markers.snapshot.skip_snapshot_verify(paths=['$..loggingConfiguration', '$..tracingConfiguration'])
class TestEncodeDecode:

    @markers.aws.validated
    def test_base_64_encode(self, create_iam_role_for_sfn, create_state_machine, sfn_snapshot, aws_client):
        if False:
            i = 10
            return i + 15
        input_values = ['', 'Data to encode']
        create_and_test_on_inputs(aws_client.stepfunctions, create_iam_role_for_sfn, create_state_machine, sfn_snapshot, IFT.BASE_64_ENCODE, input_values)

    @markers.aws.validated
    def test_base_64_decode(self, create_iam_role_for_sfn, create_state_machine, sfn_snapshot, aws_client):
        if False:
            return 10
        input_values = ['', 'RGF0YSB0byBlbmNvZGU=']
        create_and_test_on_inputs(aws_client.stepfunctions, create_iam_role_for_sfn, create_state_machine, sfn_snapshot, IFT.BASE_64_DECODE, input_values)
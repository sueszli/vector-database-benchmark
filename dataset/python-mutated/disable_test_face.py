from os.path import dirname, join, realpath
from time import sleep
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import Gender
from msrest.authentication import CognitiveServicesCredentials
from azure_devtools.scenario_tests import ReplayableTest, AzureTestError
from devtools_testutils import mgmt_settings_fake as fake_settings
CWD = dirname(realpath(__file__))

class FaceTest(ReplayableTest):
    FILTER_HEADERS = ReplayableTest.FILTER_HEADERS + ['Ocp-Apim-Subscription-Key']

    def __init__(self, method_name):
        if False:
            return 10
        (self._fake_settings, self._real_settings) = self._load_settings()
        super(FaceTest, self).__init__(method_name)

    @property
    def settings(self):
        if False:
            i = 10
            return i + 15
        if self.is_live:
            if self._real_settings:
                return self._real_settings
            else:
                raise AzureTestError('Need a mgmt_settings_real.py file to run tests live.')
        else:
            return self._fake_settings

    def _load_settings(self):
        if False:
            print('Hello World!')
        try:
            from devtools_testutils import mgmt_settings_real as real_settings
            return (fake_settings, real_settings)
        except ImportError:
            return (fake_settings, None)

    def test_face_detect(self):
        if False:
            i = 10
            return i + 15
        credentials = CognitiveServicesCredentials(self.settings.CS_SUBSCRIPTION_KEY)
        face_client = FaceClient('https://westus2.api.cognitive.microsoft.com', credentials=credentials)
        with open(join(CWD, 'facefindsimilar.queryface.jpg'), 'rb') as face_fd:
            result = face_client.face.detect_with_stream(face_fd, return_face_attributes=['age', 'gender', 'headPose', 'smile', 'facialHair', 'glasses', 'emotion', 'hair', 'makeup', 'occlusion', 'accessories', 'blur', 'exposure', 'noise'])
        detected = result[0]
        self.assertEqual(detected.face_attributes.age, 53.0)
        self.assertEqual(detected.face_attributes.gender, Gender.female)
        self.assertEqual(detected.face_attributes.emotion.happiness, 1.0)

    def test_snapshot(self):
        if False:
            while True:
                i = 10
        credentials = CognitiveServicesCredentials(self.settings.CS_SUBSCRIPTION_KEY)
        face_client = FaceClient('https://westus2.api.cognitive.microsoft.com', credentials=credentials)
        personGroupId = '69ff3e98-2de7-468e-beae-f78aa85200db'
        newPersonGroupId = 'fb644ecf-3ed0-4b25-9270-1d174b980afb'
        try:
            face_client.person_group.delete(personGroupId)
        except Exception:
            pass
        face_client.person_group.create(personGroupId, 'test', 'test')
        apply_scope = [self.settings.SUBSCRIPTION_ID]
        snapshot_type = 'PersonGroup'
        takeSnapshotResponse = face_client.snapshot.take(snapshot_type, personGroupId, apply_scope, raw=True)
        takeOperationId = takeSnapshotResponse.headers['Operation-Location'].split('/')[2]
        getOperationStatusResponse = face_client.snapshot.get_operation_status(takeOperationId)
        operationStatus = getOperationStatusResponse.status
        while operationStatus != 'succeeded' and operationStatus != 'failed':
            getOperationStatusResponse = face_client.snapshot.get_operation_status(takeOperationId)
            operationStatus = getOperationStatusResponse.status
            if self.is_live:
                sleep(1)
        assert operationStatus == 'succeeded'
        snapshotId = getOperationStatusResponse.resource_location.split('/')[2]
        applySnapshotResponse = face_client.snapshot.apply(snapshotId, newPersonGroupId, raw=True)
        applyOperationId = applySnapshotResponse.headers['Operation-Location'].split('/')[2]
        applyOperationStatusResponse = face_client.snapshot.get_operation_status(applyOperationId)
        operationStatus = applyOperationStatusResponse.status
        while operationStatus != 'succeeded' and operationStatus != 'failed':
            applyOperationStatusResponse = face_client.snapshot.get_operation_status(applyOperationId)
            operationStatus = applyOperationStatusResponse.status
            if self.is_live:
                sleep(1)
        assert operationStatus == 'succeeded'
        face_client.snapshot.delete(snapshotId)
        face_client.person_group.delete(personGroupId)
        face_client.person_group.delete(newPersonGroupId)
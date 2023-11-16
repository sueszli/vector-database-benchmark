import logging
from random import randint
from uuid import uuid4
from bigdl.ppml.fl.psi.psi_intersection import PsiIntersection
from bigdl.ppml.fl.nn.generated.psi_service_pb2_grpc import *
from bigdl.ppml.fl.nn.generated.psi_service_pb2 import *

class PSIServiceImpl(PSIServiceServicer):

    def __init__(self, conf) -> None:
        if False:
            while True:
                i = 10
        self.client_salt = None
        self.client_secret = None
        self.client_shuffle_seed = 0
        self.psi_intersection = PsiIntersection(conf['clientNum'])

    def getSalt(self, request, context):
        if False:
            while True:
                i = 10
        if self.client_salt is not None:
            salt = self.client_salt
        else:
            salt = str(uuid4())
            self.client_salt = salt
        if self.client_secret is None:
            self.client_secret = request.secure_code
        elif self.client_secret != request.secure_code:
            salt = ''
        if self.client_shuffle_seed == 0:
            self.client_shuffle_seed = randint(0, 100)
        return SaltReply(salt_reply=salt)

    def uploadSet(self, request, context):
        if False:
            print('Hello World!')
        client_id = request.client_id
        ids = request.hashedID
        self.psi_intersection.add_collection(ids)
        logging.info(f'{len(self.psi_intersection.collection)}-th collection added')
        return UploadSetResponse(status=1)

    def downloadIntersection(self, request, context):
        if False:
            while True:
                i = 10
        intersection = self.psi_intersection.get_intersection()
        return DownloadIntersectionResponse(intersection=intersection)
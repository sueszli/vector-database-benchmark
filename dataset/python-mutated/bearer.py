from fastapi import Response, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from fastapi_users.authentication.transport.base import Transport, TransportLogoutNotSupportedError
from fastapi_users.openapi import OpenAPIResponseType
from fastapi_users.schemas import model_dump

class BearerResponse(BaseModel):
    access_token: str
    token_type: str

class BearerTransport(Transport):
    scheme: OAuth2PasswordBearer

    def __init__(self, tokenUrl: str):
        if False:
            return 10
        self.scheme = OAuth2PasswordBearer(tokenUrl, auto_error=False)

    async def get_login_response(self, token: str) -> Response:
        bearer_response = BearerResponse(access_token=token, token_type='bearer')
        return JSONResponse(model_dump(bearer_response))

    async def get_logout_response(self) -> Response:
        raise TransportLogoutNotSupportedError()

    @staticmethod
    def get_openapi_login_responses_success() -> OpenAPIResponseType:
        if False:
            for i in range(10):
                print('nop')
        return {status.HTTP_200_OK: {'model': BearerResponse, 'content': {'application/json': {'example': {'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiOTIyMWZmYzktNjQwZi00MzcyLTg2ZDMtY2U2NDJjYmE1NjAzIiwiYXVkIjoiZmFzdGFwaS11c2VyczphdXRoIiwiZXhwIjoxNTcxNTA0MTkzfQ.M10bjOe45I5Ncu_uXvOmVV8QxnL-nZfcH96U90JaocI', 'token_type': 'bearer'}}}}}

    @staticmethod
    def get_openapi_logout_responses_success() -> OpenAPIResponseType:
        if False:
            i = 10
            return i + 15
        return {}
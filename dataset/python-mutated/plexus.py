from __future__ import annotations
from typing import Any
import arrow
import jwt
import requests
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.models import Variable

class PlexusHook(BaseHook):
    """
    Used for jwt token generation and storage to make Plexus API calls.

    Requires email and password Airflow variables be created.

    Example:
        - export AIRFLOW_VAR_EMAIL = user@corescientific.com
        - export AIRFLOW_VAR_PASSWORD = *******

    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.__token = None
        self.__token_exp = None
        self.host = 'https://apiplexus.corescientific.com/'
        self.user_id = None

    def _generate_token(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        login = Variable.get('email')
        pwd = Variable.get('password')
        if login is None or pwd is None:
            raise AirflowException('No valid email/password supplied.')
        token_endpoint = self.host + 'sso/jwt-token/'
        response = requests.post(token_endpoint, data={'email': login, 'password': pwd}, timeout=5)
        if not response.ok:
            raise AirflowException(f'Could not retrieve JWT Token. Status Code: [{response.status_code}]. Reason: {response.reason} - {response.text}')
        token = response.json()['access']
        payload = jwt.decode(token, verify=False)
        self.user_id = payload['user_id']
        self.__token_exp = payload['exp']
        return token

    @property
    def token(self) -> Any:
        if False:
            print('Hello World!')
        'Returns users token.'
        if self.__token is not None:
            if not self.__token_exp or arrow.get(self.__token_exp) <= arrow.now():
                self.__token = self._generate_token()
            return self.__token
        else:
            self.__token = self._generate_token()
            return self.__token
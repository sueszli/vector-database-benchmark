import warnings
from base64 import b64decode
from json import JSONDecodeError, loads
from typing import Optional
from dagster import DagsterRun, QueuedRunCoordinator, SubmitRunContext

class CustomRunCoordinator(QueuedRunCoordinator):

    def get_email(self, jwt_claims_header: Optional[str]) -> Optional[str]:
        if False:
            while True:
                i = 10
        if not jwt_claims_header:
            return None
        split_header_tokens = jwt_claims_header.split('.')
        if len(split_header_tokens) < 2:
            return None
        decoded_claims_json_str = b64decode(split_header_tokens[1])
        try:
            claims_json = loads(decoded_claims_json_str)
            return claims_json.get('email')
        except JSONDecodeError:
            return None

    def submit_run(self, context: SubmitRunContext) -> DagsterRun:
        if False:
            for i in range(10):
                print('nop')
        dagster_run = context.dagster_run
        jwt_claims_header = context.get_request_header('X-Amzn-Oidc-Data')
        email = self.get_email(jwt_claims_header)
        if email:
            self._instance.add_run_tags(dagster_run.run_id, {'user': email})
        else:
            warnings.warn(f"Couldn't decode JWT header {jwt_claims_header}")
        return super().submit_run(context)
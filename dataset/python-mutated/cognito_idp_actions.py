"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Cognito to
sign up a user, register a multi-factor authentication (MFA) application, sign in
using an MFA code, and sign in using a tracked device.
"""
import base64
import hashlib
import hmac
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class CognitoIdentityProviderWrapper:
    """Encapsulates Amazon Cognito actions"""

    def __init__(self, cognito_idp_client, user_pool_id, client_id, client_secret=None):
        if False:
            return 10
        '\n        :param cognito_idp_client: A Boto3 Amazon Cognito Identity Provider client.\n        :param user_pool_id: The ID of an existing Amazon Cognito user pool.\n        :param client_id: The ID of a client application registered with the user pool.\n        :param client_secret: The client secret, if the client has a secret.\n        '
        self.cognito_idp_client = cognito_idp_client
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.client_secret = client_secret

    def _secret_hash(self, user_name):
        if False:
            i = 10
            return i + 15
        '\n        Calculates a secret hash from a user name and a client secret.\n\n        :param user_name: The user name to use when calculating the hash.\n        :return: The secret hash.\n        '
        key = self.client_secret.encode()
        msg = bytes(user_name + self.client_id, 'utf-8')
        secret_hash = base64.b64encode(hmac.new(key, msg, digestmod=hashlib.sha256).digest()).decode()
        logger.info('Made secret hash for %s: %s.', user_name, secret_hash)
        return secret_hash

    def sign_up_user(self, user_name, password, user_email):
        if False:
            return 10
        '\n        Signs up a new user with Amazon Cognito. This action prompts Amazon Cognito\n        to send an email to the specified email address. The email contains a code that\n        can be used to confirm the user.\n\n        When the user already exists, the user status is checked to determine whether\n        the user has been confirmed.\n\n        :param user_name: The user name that identifies the new user.\n        :param password: The password for the new user.\n        :param user_email: The email address for the new user.\n        :return: True when the user is already confirmed with Amazon Cognito.\n                 Otherwise, false.\n        '
        try:
            kwargs = {'ClientId': self.client_id, 'Username': user_name, 'Password': password, 'UserAttributes': [{'Name': 'email', 'Value': user_email}]}
            if self.client_secret is not None:
                kwargs['SecretHash'] = self._secret_hash(user_name)
            response = self.cognito_idp_client.sign_up(**kwargs)
            confirmed = response['UserConfirmed']
        except ClientError as err:
            if err.response['Error']['Code'] == 'UsernameExistsException':
                response = self.cognito_idp_client.admin_get_user(UserPoolId=self.user_pool_id, Username=user_name)
                logger.warning('User %s exists and is %s.', user_name, response['UserStatus'])
                confirmed = response['UserStatus'] == 'CONFIRMED'
            else:
                logger.error("Couldn't sign up %s. Here's why: %s: %s", user_name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        return confirmed

    def resend_confirmation(self, user_name):
        if False:
            print('Hello World!')
        '\n        Prompts Amazon Cognito to resend an email with a new confirmation code.\n\n        :param user_name: The name of the user who will receive the email.\n        :return: Delivery information about where the email is sent.\n        '
        try:
            kwargs = {'ClientId': self.client_id, 'Username': user_name}
            if self.client_secret is not None:
                kwargs['SecretHash'] = self._secret_hash(user_name)
            response = self.cognito_idp_client.resend_confirmation_code(**kwargs)
            delivery = response['CodeDeliveryDetails']
        except ClientError as err:
            logger.error("Couldn't resend confirmation to %s. Here's why: %s: %s", user_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return delivery

    def confirm_user_sign_up(self, user_name, confirmation_code):
        if False:
            while True:
                i = 10
        "\n        Confirms a previously created user. A user must be confirmed before they\n        can sign in to Amazon Cognito.\n\n        :param user_name: The name of the user to confirm.\n        :param confirmation_code: The confirmation code sent to the user's registered\n                                  email address.\n        :return: True when the confirmation succeeds.\n        "
        try:
            kwargs = {'ClientId': self.client_id, 'Username': user_name, 'ConfirmationCode': confirmation_code}
            if self.client_secret is not None:
                kwargs['SecretHash'] = self._secret_hash(user_name)
            self.cognito_idp_client.confirm_sign_up(**kwargs)
        except ClientError as err:
            logger.error("Couldn't confirm sign up for %s. Here's why: %s: %s", user_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return True

    def list_users(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of the users in the current user pool.\n\n        :return: The list of users.\n        '
        try:
            response = self.cognito_idp_client.list_users(UserPoolId=self.user_pool_id)
            users = response['Users']
        except ClientError as err:
            logger.error("Couldn't list users for %s. Here's why: %s: %s", self.user_pool_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return users

    def start_sign_in(self, user_name, password):
        if False:
            return 10
        "\n        Starts the sign-in process for a user by using administrator credentials.\n        This method of signing in is appropriate for code running on a secure server.\n\n        If the user pool is configured to require MFA and this is the first sign-in\n        for the user, Amazon Cognito returns a challenge response to set up an\n        MFA application. When this occurs, this function gets an MFA secret from\n        Amazon Cognito and returns it to the caller.\n\n        :param user_name: The name of the user to sign in.\n        :param password: The user's password.\n        :return: The result of the sign-in attempt. When sign-in is successful, this\n                 returns an access token that can be used to get AWS credentials. Otherwise,\n                 Amazon Cognito returns a challenge to set up an MFA application,\n                 or a challenge to enter an MFA code from a registered MFA application.\n        "
        try:
            kwargs = {'UserPoolId': self.user_pool_id, 'ClientId': self.client_id, 'AuthFlow': 'ADMIN_USER_PASSWORD_AUTH', 'AuthParameters': {'USERNAME': user_name, 'PASSWORD': password}}
            if self.client_secret is not None:
                kwargs['AuthParameters']['SECRET_HASH'] = self._secret_hash(user_name)
            response = self.cognito_idp_client.admin_initiate_auth(**kwargs)
            challenge_name = response.get('ChallengeName', None)
            if challenge_name == 'MFA_SETUP':
                if 'SOFTWARE_TOKEN_MFA' in response['ChallengeParameters']['MFAS_CAN_SETUP']:
                    response.update(self.get_mfa_secret(response['Session']))
                else:
                    raise RuntimeError('The user pool requires MFA setup, but the user pool is not configured for TOTP MFA. This example requires TOTP MFA.')
        except ClientError as err:
            logger.error("Couldn't start sign in for %s. Here's why: %s: %s", user_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            response.pop('ResponseMetadata', None)
            return response

    def get_mfa_secret(self, session):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets a token that can be used to associate an MFA application with the user.\n\n        :param session: Session information returned from a previous call to initiate\n                        authentication.\n        :return: An MFA token that can be used to set up an MFA application.\n        '
        try:
            response = self.cognito_idp_client.associate_software_token(Session=session)
        except ClientError as err:
            logger.error("Couldn't get MFA secret. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            response.pop('ResponseMetadata', None)
            return response

    def verify_mfa(self, session, user_code):
        if False:
            i = 10
            return i + 15
        '\n        Verify a new MFA application that is associated with a user.\n\n        :param session: Session information returned from a previous call to initiate\n                        authentication.\n        :param user_code: A code generated by the associated MFA application.\n        :return: Status that indicates whether the MFA application is verified.\n        '
        try:
            response = self.cognito_idp_client.verify_software_token(Session=session, UserCode=user_code)
        except ClientError as err:
            logger.error("Couldn't verify MFA. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            response.pop('ResponseMetadata', None)
            return response

    def respond_to_mfa_challenge(self, user_name, session, mfa_code):
        if False:
            return 10
        '\n        Responds to a challenge for an MFA code. This completes the second step of\n        a two-factor sign-in. When sign-in is successful, it returns an access token\n        that can be used to get AWS credentials from Amazon Cognito.\n\n        :param user_name: The name of the user who is signing in.\n        :param session: Session information returned from a previous call to initiate\n                        authentication.\n        :param mfa_code: A code generated by the associated MFA application.\n        :return: The result of the authentication. When successful, this contains an\n                 access token for the user.\n        '
        try:
            kwargs = {'UserPoolId': self.user_pool_id, 'ClientId': self.client_id, 'ChallengeName': 'SOFTWARE_TOKEN_MFA', 'Session': session, 'ChallengeResponses': {'USERNAME': user_name, 'SOFTWARE_TOKEN_MFA_CODE': mfa_code}}
            if self.client_secret is not None:
                kwargs['ChallengeResponses']['SECRET_HASH'] = self._secret_hash(user_name)
            response = self.cognito_idp_client.admin_respond_to_auth_challenge(**kwargs)
            auth_result = response['AuthenticationResult']
        except ClientError as err:
            if err.response['Error']['Code'] == 'ExpiredCodeException':
                logger.warning('Your MFA code has expired or has been used already. You might have to wait a few seconds until your app shows you a new code.')
            else:
                logger.error("Couldn't respond to mfa challenge for %s. Here's why: %s: %s", user_name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        else:
            return auth_result

    def confirm_mfa_device(self, user_name, device_key, device_group_key, device_password, access_token, aws_srp):
        if False:
            print('Hello World!')
        "\n        Confirms an MFA device to be tracked by Amazon Cognito. When a device is\n        tracked, its key and password can be used to sign in without requiring a new\n        MFA code from the MFA application.\n\n        :param user_name: The user that is associated with the device.\n        :param device_key: The key of the device, returned by Amazon Cognito.\n        :param device_group_key: The group key of the device, returned by Amazon Cognito.\n        :param device_password: The password that is associated with the device.\n        :param access_token: The user's access token.\n        :param aws_srp: A class that helps with Secure Remote Password (SRP)\n                        calculations. The scenario associated with this example uses\n                        the warrant package.\n        :return: True when the user must confirm the device. Otherwise, False. When\n                 False, the device is automatically confirmed and tracked.\n        "
        srp_helper = aws_srp.AWSSRP(username=user_name, password=device_password, pool_id='_', client_id=self.client_id, client_secret=None, client=self.cognito_idp_client)
        device_and_pw = f'{device_group_key}{device_key}:{device_password}'
        device_and_pw_hash = aws_srp.hash_sha256(device_and_pw.encode('utf-8'))
        salt = aws_srp.pad_hex(aws_srp.get_random(16))
        x_value = aws_srp.hex_to_long(aws_srp.hex_hash(salt + device_and_pw_hash))
        verifier = aws_srp.pad_hex(pow(srp_helper.val_g, x_value, srp_helper.big_n))
        device_secret_verifier_config = {'PasswordVerifier': base64.standard_b64encode(bytearray.fromhex(verifier)).decode('utf-8'), 'Salt': base64.standard_b64encode(bytearray.fromhex(salt)).decode('utf-8')}
        try:
            response = self.cognito_idp_client.confirm_device(AccessToken=access_token, DeviceKey=device_key, DeviceSecretVerifierConfig=device_secret_verifier_config)
            user_confirm = response['UserConfirmationNecessary']
        except ClientError as err:
            logger.error("Couldn't confirm mfa device %s. Here's why: %s: %s", device_key, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return user_confirm

    def sign_in_with_tracked_device(self, user_name, password, device_key, device_group_key, device_password, aws_srp):
        if False:
            return 10
        "\n        Signs in to Amazon Cognito as a user who has a tracked device. Signing in\n        with a tracked device lets a user sign in without entering a new MFA code.\n\n        Signing in with a tracked device requires that the client respond to the SRP\n        protocol. The scenario associated with this example uses the warrant package\n        to help with SRP calculations.\n\n        For more information on SRP, see https://en.wikipedia.org/wiki/Secure_Remote_Password_protocol.\n\n        :param user_name: The user that is associated with the device.\n        :param password: The user's password.\n        :param device_key: The key of a tracked device.\n        :param device_group_key: The group key of a tracked device.\n        :param device_password: The password that is associated with the device.\n        :param aws_srp: A class that helps with SRP calculations. The scenario\n                        associated with this example uses the warrant package.\n        :return: The result of the authentication. When successful, this contains an\n                 access token for the user.\n        "
        try:
            srp_helper = aws_srp.AWSSRP(username=user_name, password=device_password, pool_id='_', client_id=self.client_id, client_secret=None, client=self.cognito_idp_client)
            response_init = self.cognito_idp_client.initiate_auth(ClientId=self.client_id, AuthFlow='USER_PASSWORD_AUTH', AuthParameters={'USERNAME': user_name, 'PASSWORD': password, 'DEVICE_KEY': device_key})
            if response_init['ChallengeName'] != 'DEVICE_SRP_AUTH':
                raise RuntimeError(f"Expected DEVICE_SRP_AUTH challenge but got {response_init['ChallengeName']}.")
            auth_params = srp_helper.get_auth_params()
            auth_params['DEVICE_KEY'] = device_key
            response_auth = self.cognito_idp_client.respond_to_auth_challenge(ClientId=self.client_id, ChallengeName='DEVICE_SRP_AUTH', ChallengeResponses=auth_params)
            if response_auth['ChallengeName'] != 'DEVICE_PASSWORD_VERIFIER':
                raise RuntimeError(f"Expected DEVICE_PASSWORD_VERIFIER challenge but got {response_init['ChallengeName']}.")
            challenge_params = response_auth['ChallengeParameters']
            challenge_params['USER_ID_FOR_SRP'] = device_group_key + device_key
            cr = srp_helper.process_challenge(challenge_params, {'USERNAME': user_name})
            cr['USERNAME'] = user_name
            cr['DEVICE_KEY'] = device_key
            response_verifier = self.cognito_idp_client.respond_to_auth_challenge(ClientId=self.client_id, ChallengeName='DEVICE_PASSWORD_VERIFIER', ChallengeResponses=cr)
            auth_tokens = response_verifier['AuthenticationResult']
        except ClientError as err:
            logger.error("Couldn't start client sign in for %s. Here's why: %s: %s", user_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return auth_tokens
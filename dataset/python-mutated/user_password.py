from django.utils.crypto import constant_time_compare
from rest_framework import serializers, status
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import control_silo_endpoint
from sentry.api.bases.user import UserEndpoint
from sentry.auth import password_validation
from sentry.security import capture_security_activity

class UserPasswordSerializer(serializers.Serializer):
    password = serializers.CharField(required=True, trim_whitespace=False)
    passwordNew = serializers.CharField(required=True, trim_whitespace=False)
    passwordVerify = serializers.CharField(required=True, trim_whitespace=False)

    def validate_password(self, value):
        if False:
            i = 10
            return i + 15
        user = self.context['user']
        if not user.check_password(value):
            raise serializers.ValidationError('The password you entered is not correct.')
        return value

    def validate_passwordNew(self, value):
        if False:
            print('Hello World!')
        user = self.context['user']
        password_validation.validate_password(value, user=user)
        if user.is_managed:
            raise serializers.ValidationError('This account is managed and the password cannot be changed via Sentry.')
        return value

    def validate(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        attrs = super().validate(attrs)
        if not constant_time_compare(attrs.get('passwordNew'), attrs.get('passwordVerify')):
            raise serializers.ValidationError('The passwords you entered did not match.')
        return attrs

@control_silo_endpoint
class UserPasswordEndpoint(UserEndpoint):
    publish_status = {'PUT': ApiPublishStatus.UNKNOWN}

    def put(self, request: Request, user) -> Response:
        if False:
            while True:
                i = 10
        serializer = UserPasswordSerializer(data=request.data, context={'user': user})
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        result = serializer.validated_data
        user.set_password(result['passwordNew'])
        user.refresh_session_nonce(request._request)
        user.clear_lost_passwords()
        user.save()
        capture_security_activity(account=user, type='password-changed', actor=request.user, ip_address=request.META['REMOTE_ADDR'], send_email=True)
        return Response(status=status.HTTP_204_NO_CONTENT)
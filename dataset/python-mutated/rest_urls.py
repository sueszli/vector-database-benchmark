from django.db import models as model_fields
from django.conf import settings
from django.conf.urls import url, include
from django.contrib.auth import models as django_models
from polymorphic.models import PolymorphicModel
from cabot.cabotapp import models, alert
from rest_framework import routers, serializers, viewsets, mixins
import logging
logger = logging.getLogger(__name__)
router = routers.DefaultRouter()

def create_viewset(arg_model, arg_fields, arg_read_only_fields=(), readonly=False):
    if False:
        while True:
            i = 10
    arg_read_only_fields = ('id',) + arg_read_only_fields
    for field in arg_read_only_fields:
        if field not in arg_fields:
            arg_fields = arg_fields + (field,)

    class Serializer(serializers.ModelSerializer):

        class Meta:
            model = arg_model
            fields = arg_fields
            read_only_fields = arg_read_only_fields
    viewset_class = None
    if readonly:
        viewset_class = viewsets.ReadOnlyModelViewSet
    else:
        viewset_class = viewsets.ModelViewSet

    class ViewSet(viewset_class):
        queryset = arg_model.objects
        serializer_class = Serializer
        ordering = ['id']
        filter_fields = arg_fields
    return ViewSet
check_group_mixin_fields = ('name', 'users_to_notify', 'alerts_enabled', 'status_checks', 'alerts', 'hackpad_id')
router.register('services', create_viewset(arg_model=models.Service, arg_fields=check_group_mixin_fields + ('url', 'instances', 'overall_status')))
router.register('instances', create_viewset(arg_model=models.Instance, arg_fields=check_group_mixin_fields + ('address', 'overall_status')))
status_check_fields = ('name', 'active', 'importance', 'frequency', 'debounce', 'calculated_status')
router.register('status_checks', create_viewset(arg_model=models.StatusCheck, arg_fields=status_check_fields, readonly=True))
router.register('icmp_checks', create_viewset(arg_model=models.ICMPStatusCheck, arg_fields=status_check_fields))
router.register('graphite_checks', create_viewset(arg_model=models.GraphiteStatusCheck, arg_fields=status_check_fields + ('metric', 'check_type', 'value', 'expected_num_hosts', 'allowed_num_failures')))
router.register('http_checks', create_viewset(arg_model=models.HttpStatusCheck, arg_fields=status_check_fields + ('endpoint', 'username', 'password', 'text_match', 'status_code', 'timeout', 'verify_ssl_certificate')))
router.register('jenkins_checks', create_viewset(arg_model=models.JenkinsStatusCheck, arg_fields=status_check_fields + ('max_queued_build_time', 'jenkins_config')))
if settings.EXPOSE_USER_API:
    router.register('users', create_viewset(arg_model=django_models.User, arg_fields=('password', 'is_active', 'groups', 'username', 'first_name', 'last_name', 'email')))
    router.register('user_profiles', create_viewset(arg_model=models.UserProfile, arg_fields=('user', 'fallback_alert_user')))
router.register('shifts', create_viewset(arg_model=models.Shift, arg_fields=('start', 'end', 'user', 'uid', 'deleted')))
router.register('alertplugins', create_viewset(arg_model=alert.AlertPlugin, arg_fields=('title',), readonly=True))
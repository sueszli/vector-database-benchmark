from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from sentry_relay.exceptions import RelayError
from sentry_relay.processing import compare_version as compare_version_relay
from sentry_relay.processing import parse_release
from sentry.backup.scopes import RelocationScope
from sentry.db.models import BoundedPositiveIntegerField, FlexibleForeignKey, Model, region_silo_only_model, sane_repr
from sentry.models.release import DB_VERSION_LENGTH, Release, follows_semver_versioning_scheme
from sentry.utils import metrics

@region_silo_only_model
class GroupResolution(Model):
    """
    Describes when a group was marked as resolved.
    """
    __relocation_scope__ = RelocationScope.Excluded

    class Type:
        in_release = 0
        in_next_release = 1

    class Status:
        pending = 0
        resolved = 1
    group = FlexibleForeignKey('sentry.Group', unique=True)
    release = FlexibleForeignKey('sentry.Release')
    current_release_version = models.CharField(max_length=DB_VERSION_LENGTH, null=True, blank=True)
    type = BoundedPositiveIntegerField(choices=((Type.in_next_release, 'in_next_release'), (Type.in_release, 'in_release')), null=True)
    actor_id = BoundedPositiveIntegerField(null=True)
    datetime = models.DateTimeField(default=timezone.now, db_index=True)
    status = BoundedPositiveIntegerField(default=Status.pending, choices=((Status.pending, _('Pending')), (Status.resolved, _('Resolved'))))

    class Meta:
        db_table = 'sentry_groupresolution'
        app_label = 'sentry'
    __repr__ = sane_repr('group_id', 'release_id')

    @classmethod
    def has_resolution(cls, group, release):
        if False:
            print('Hello World!')
        '\n        Determine if a resolution exists for the given group and release.\n\n        This is used to suggest if a regression has occurred.\n        '

        def compare_release_dates_for_in_next_release(res_release, res_release_datetime, release):
            if False:
                print('Hello World!')
            '\n            Helper function that compares release versions based on date for\n            `GroupResolution.Type.in_next_release`\n            '
            return res_release == release.id or res_release_datetime > release.date_added
        try:
            (res_type, res_release, res_release_version, res_release_datetime, current_release_version) = cls.objects.filter(group=group).select_related('release').values_list('type', 'release__id', 'release__version', 'release__date_added', 'current_release_version')[0]
        except IndexError:
            return False
        if not release:
            return True
        follows_semver = follows_semver_versioning_scheme(project_id=group.project.id, org_id=group.organization.id, release_version=release.version)
        if current_release_version:
            if follows_semver:
                try:
                    current_release_raw = parse_release(current_release_version).get('version_raw')
                    release_raw = parse_release(release.version).get('version_raw')
                    return compare_version_relay(current_release_raw, release_raw) >= 0
                except RelayError:
                    ...
            else:
                try:
                    current_release_obj = Release.objects.get(organization_id=group.organization.id, version=current_release_version)
                    return compare_release_dates_for_in_next_release(res_release=current_release_obj.id, res_release_datetime=current_release_obj.date_added, release=release)
                except Release.DoesNotExist:
                    ...
        if res_type in (None, cls.Type.in_next_release):
            metrics.incr('groupresolution.has_resolution.in_next_release', sample_rate=1.0)
            return compare_release_dates_for_in_next_release(res_release=res_release, res_release_datetime=res_release_datetime, release=release)
        elif res_type == cls.Type.in_release:
            if res_release == release.id:
                return False
            if follows_semver:
                try:
                    res_release_raw = parse_release(res_release_version).get('version_raw')
                    release_raw = parse_release(release.version).get('version_raw')
                    return compare_version_relay(res_release_raw, release_raw) == 1
                except RelayError:
                    ...
            return res_release_datetime >= release.date_added
        else:
            raise NotImplementedError
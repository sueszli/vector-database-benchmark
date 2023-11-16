from sentry.api.serializers import Serializer, register
from sentry.models.projectplatform import ProjectPlatform

@register(ProjectPlatform)
class ProjectPlatformSerializer(Serializer):
    """
    Tracks usage of a platform for a given project.

    Note: This model is used solely for analytics.
    """

    def serialize(self, obj, attrs, user):
        if False:
            i = 10
            return i + 15
        return {'platform': obj.platform, 'dateCreated': obj.date_added}
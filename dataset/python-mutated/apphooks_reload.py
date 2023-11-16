import uuid
from django.db import models

class UrlconfRevision(models.Model):
    revision = models.CharField(max_length=255)

    class Meta:
        app_label = 'cms'

    def save(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Simply forces this model to be a singleton.\n        '
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def get_or_create_revision(cls, revision=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convenience method for getting or creating revision.\n        '
        if revision is None:
            revision = str(uuid.uuid4())
        (obj, created) = cls.objects.get_or_create(pk=1, defaults={'revision': revision})
        return (obj.revision, created)

    @classmethod
    def update_revision(cls, revision):
        if False:
            i = 10
            return i + 15
        '\n        Convenience method for updating the revision.\n        '
        (obj, created) = cls.objects.get_or_create(pk=1, defaults={'revision': revision})
        if not created:
            obj.revision = revision
            obj.save()
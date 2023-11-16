from chatterbot.conversation import StatementMixin
from chatterbot import constants
from django.db import models
from django.utils import timezone
from django.conf import settings
DJANGO_APP_NAME = constants.DEFAULT_DJANGO_APP_NAME
STATEMENT_MODEL = 'Statement'
TAG_MODEL = 'Tag'
if hasattr(settings, 'CHATTERBOT'):
    '\n    Allow related models to be overridden in the project settings.\n    Default to the original settings if one is not defined.\n    '
    DJANGO_APP_NAME = settings.CHATTERBOT.get('django_app_name', DJANGO_APP_NAME)
    STATEMENT_MODEL = settings.CHATTERBOT.get('statement_model', STATEMENT_MODEL)

class AbstractBaseTag(models.Model):
    """
    The abstract base tag allows other models to be created
    using the attributes that exist on the default models.
    """
    name = models.SlugField(max_length=constants.TAG_NAME_MAX_LENGTH, unique=True)

    class Meta:
        abstract = True

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.name

class AbstractBaseStatement(models.Model, StatementMixin):
    """
    The abstract base statement allows other models to be created
    using the attributes that exist on the default models.
    """
    text = models.CharField(max_length=constants.STATEMENT_TEXT_MAX_LENGTH)
    search_text = models.CharField(max_length=constants.STATEMENT_TEXT_MAX_LENGTH, blank=True)
    conversation = models.CharField(max_length=constants.CONVERSATION_LABEL_MAX_LENGTH)
    created_at = models.DateTimeField(default=timezone.now, help_text='The date and time that the statement was created at.')
    in_response_to = models.CharField(max_length=constants.STATEMENT_TEXT_MAX_LENGTH, null=True)
    search_in_response_to = models.CharField(max_length=constants.STATEMENT_TEXT_MAX_LENGTH, blank=True)
    persona = models.CharField(max_length=constants.PERSONA_MAX_LENGTH)
    tags = models.ManyToManyField(TAG_MODEL, related_name='statements')
    confidence = 0

    class Meta:
        abstract = True

    def __str__(self):
        if False:
            print('Hello World!')
        if len(self.text.strip()) > 60:
            return '{}...'.format(self.text[:57])
        elif len(self.text.strip()) > 0:
            return self.text
        return '<empty>'

    def get_tags(self):
        if False:
            print('Hello World!')
        '\n        Return the list of tags for this statement.\n        (Overrides the method from StatementMixin)\n        '
        return list(self.tags.values_list('name', flat=True))

    def add_tags(self, *tags):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a list of strings to the statement as tags.\n        (Overrides the method from StatementMixin)\n        '
        for _tag in tags:
            self.tags.get_or_create(name=_tag)
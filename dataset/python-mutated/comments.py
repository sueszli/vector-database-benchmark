from django.forms import BooleanField, ValidationError
from django.utils.timezone import now
from django.utils.translation import gettext as _
from modelcluster.forms import BaseChildFormSet
from .models import WagtailAdminModelForm

class CommentReplyForm(WagtailAdminModelForm):

    class Meta:
        fields = ('text',)

    def clean(self):
        if False:
            for i in range(10):
                print('nop')
        cleaned_data = super().clean()
        user = self.for_user
        if not self.instance.pk:
            self.instance.user = user
        elif self.instance.user != user:
            if any((field for field in self.changed_data)):
                self.add_error(None, ValidationError(_("You cannot edit another user's comment.")))
        return cleaned_data

class CommentForm(WagtailAdminModelForm):
    """
    This is designed to be subclassed and have the user overridden to enable user-based validation within the edit handler system
    """
    resolved = BooleanField(required=False)

    class Meta:
        formsets = {'replies': {'form': CommentReplyForm, 'inherit_kwargs': ['for_user']}}

    def clean(self):
        if False:
            return 10
        cleaned_data = super().clean()
        user = self.for_user
        if not self.instance.pk:
            self.instance.user = user
        elif self.instance.user != user:
            if any((field for field in self.changed_data if field not in ['resolved', 'position', 'contentpath'])) or cleaned_data['contentpath'].split('.')[0] != self.instance.contentpath.split('.')[0]:
                self.add_error(None, ValidationError(_("You cannot edit another user's comment.")))
        return cleaned_data

    def save(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.cleaned_data.get('resolved', False):
            if not getattr(self.instance, 'resolved_at'):
                self.instance.resolved_at = now()
                self.instance.resolved_by = self.for_user
        else:
            self.instance.resolved_by = None
            self.instance.resolved_at = None
        return super().save(*args, **kwargs)

class CommentFormSet(BaseChildFormSet):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        valid_comment_ids = [comment.id for comment in self.queryset if comment.has_valid_contentpath(self.instance)]
        self.queryset = self.queryset.filter(id__in=valid_comment_ids)
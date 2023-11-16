from django.core.exceptions import PermissionDenied
from django.db import transaction
from wagtail.coreutils import find_available_slug
from wagtail.models.copying import _copy

class ParentNotTranslatedError(Exception):
    """
    Raised when a call to Page.copy_for_translation is made but the
    parent page is not translated and copy_parents is False.
    """
    pass

class CopyForTranslationPermissionError(PermissionDenied):
    """
    Raised when the object translation copy cannot be performed due to insufficient permissions.
    """
    pass

class CopyPageForTranslationPermissionError(CopyForTranslationPermissionError):
    pass

class CopyPageForTranslationAction:
    """
    Creates a copy of this page in the specified locale.

    The new page will be created in draft as a child of this page's translated
    parent.

    For example, if you are translating a blog post from English into French,
    this method will look for the French version of the blog index and create
    the French translation of the blog post under that.

    If this page's parent is not translated into the locale, then a ``ParentNotTranslatedError``
    is raised. You can circumvent this error by passing ``copy_parents=True`` which
    copies any parents that are not translated yet.

    The ``exclude_fields`` parameter can be used to set any fields to a blank value
    in the copy.

    Note that this method calls the ``.copy()`` method internally so any fields that
    are excluded in ``.exclude_fields_in_copy`` will be excluded from the translation.
    """

    def __init__(self, page, locale, copy_parents=False, alias=False, exclude_fields=None, user=None, include_subtree=False):
        if False:
            i = 10
            return i + 15
        self.page = page
        self.locale = locale
        self.copy_parents = copy_parents
        self.alias = alias
        self.exclude_fields = exclude_fields
        self.user = user
        self.include_subtree = include_subtree

    def check(self, skip_permission_checks=False):
        if False:
            print('Hello World!')
        if self.user and (not skip_permission_checks) and (not self.user.has_perms(['simple_translation.submit_translation'])):
            raise CopyPageForTranslationPermissionError('You do not have permission to submit a translation for this page.')

    def walk(self, current_page):
        if False:
            i = 10
            return i + 15
        for child_page in current_page.get_children():
            self._copy_for_translation(child_page if child_page.live else child_page.get_latest_revision_as_object(), self.locale, self.copy_parents, self.alias, self.exclude_fields)
            self.walk(child_page)

    @transaction.atomic
    def _copy_for_translation(self, page, locale, copy_parents, alias, exclude_fields):
        if False:
            print('Hello World!')
        parent = page.get_parent().specific
        slug = page.slug
        if not parent.is_root():
            try:
                translated_parent = parent.get_translation(locale)
            except parent.__class__.DoesNotExist:
                if not copy_parents:
                    raise ParentNotTranslatedError('Parent page is not translated.')
                translated_parent = parent.copy_for_translation(locale, copy_parents=True, alias=True)
        else:
            translated_parent = parent
            slug += '-' + locale.language_code
        slug = find_available_slug(translated_parent, slug)
        if alias:
            return page.create_alias(parent=translated_parent, update_slug=slug, update_locale=locale, reset_translation_key=False)
        else:

            def process_child_object(original_page, page_copy, child_relation, child_object):
                if False:
                    print('Hello World!')
                from wagtail.models import TranslatableMixin
                if isinstance(child_object, TranslatableMixin):
                    child_object.locale = locale
            return page.copy(to=translated_parent, update_attrs={'locale': locale, 'slug': slug}, copy_revisions=False, keep_live=False, reset_translation_key=False, process_child_object=process_child_object, exclude_fields=exclude_fields, log_action='wagtail.copy_for_translation')

    def execute(self, skip_permission_checks=False):
        if False:
            print('Hello World!')
        self.check(skip_permission_checks=skip_permission_checks)
        translated_page = self._copy_for_translation(self.page if self.page.live else self.page.get_latest_revision_as_object(), self.locale, self.copy_parents, self.alias, self.exclude_fields)
        if self.include_subtree:
            self.walk(self.page)
        return translated_page

class CopyForTranslationAction:
    """
    Creates a copy of this object in the specified locale.

    The ``exclude_fields`` parameter can be used to set any fields to a blank value
    in the copy.
    """

    def __init__(self, object, locale, exclude_fields=None, user=None):
        if False:
            while True:
                i = 10
        self.object = object
        self.locale = locale
        self.exclude_fields = exclude_fields
        self.user = user

    def check(self, skip_permission_checks=False):
        if False:
            for i in range(10):
                print('nop')
        if self.user and (not skip_permission_checks) and (not self.user.has_perms(['simple_translation.submit_translation'])):
            raise CopyForTranslationPermissionError('You do not have permission to submit a translation for this object.')

    @transaction.atomic
    def _copy_for_translation(self, object, locale, exclude_fields=None):
        if False:
            for i in range(10):
                print('nop')
        from wagtail.models import DraftStateMixin, TranslatableMixin
        if isinstance(object, DraftStateMixin):
            object = object.get_latest_revision_as_object()
        exclude_fields = getattr(object, 'default_exclude_fields_in_copy', []) + getattr(object, 'exclude_fields_in_copy', []) + (exclude_fields or [])
        (translated, child_object_map) = _copy(object, exclude_fields=exclude_fields)
        translated.locale = locale
        for ((_child_relation, _old_pk), child_object) in child_object_map.items():
            if isinstance(child_object, TranslatableMixin):
                child_object.locale = locale
        return translated

    def execute(self, skip_permission_checks=False):
        if False:
            i = 10
            return i + 15
        self.check(skip_permission_checks=skip_permission_checks)
        translated_object = self._copy_for_translation(self.object, self.locale, self.exclude_fields)
        return translated_object
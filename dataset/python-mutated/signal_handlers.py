import logging
from contextlib import contextmanager
from asgiref.local import Local
from django.core.cache import cache
from django.db import transaction
from django.db.models.signals import post_delete, post_migrate, post_save, pre_delete, pre_migrate
from modelcluster.fields import ParentalKey
from wagtail.models import Locale, Page, ReferenceIndex, Site
logger = logging.getLogger('wagtail')

def post_save_site_signal_handler(instance, update_fields=None, **kwargs):
    if False:
        return 10
    Site.clear_site_root_paths_cache()

def post_delete_site_signal_handler(instance, **kwargs):
    if False:
        i = 10
        return i + 15
    Site.clear_site_root_paths_cache()

def pre_delete_page_unpublish(sender, instance, **kwargs):
    if False:
        return 10
    if instance.live:
        instance.unpublish(commit=False, log_action=None)

def post_delete_page_log_deletion(sender, instance, **kwargs):
    if False:
        while True:
            i = 10
    logger.info('Page deleted: "%s" id=%d', instance.title, instance.id)

def reset_locales_display_names_cache(sender, instance, **kwargs):
    if False:
        while True:
            i = 10
    cache.delete('wagtail_locales_display_name')
reference_index_auto_update_disabled = Local()

@contextmanager
def disable_reference_index_auto_update():
    if False:
        i = 10
        return i + 15
    '\n    A context manager that can be used to temporarily disable the reference index auto-update signal handlers.\n\n    For example:\n\n    with disable_reference_index_auto_update():\n        my_instance.save()  # Reference index will not be updated by this save\n    '
    try:
        reference_index_auto_update_disabled.value = True
        yield
    finally:
        del reference_index_auto_update_disabled.value

def update_reference_index_on_save(instance, **kwargs):
    if False:
        return 10
    if kwargs.get('raw', False):
        return
    if getattr(reference_index_auto_update_disabled, 'value', False):
        return
    while True:
        parental_keys = list(filter(lambda field: isinstance(field, ParentalKey), instance._meta.get_fields()))
        if not parental_keys:
            break
        instance = getattr(instance, parental_keys[0].name)
        if instance is None:
            return
    if ReferenceIndex.is_indexed(instance._meta.model):
        with transaction.atomic():
            ReferenceIndex.create_or_update_for_object(instance)

def remove_reference_index_on_delete(instance, **kwargs):
    if False:
        return 10
    if getattr(reference_index_auto_update_disabled, 'value', False):
        return
    with transaction.atomic():
        ReferenceIndex.remove_for_object(instance)

def connect_reference_index_signal_handlers_for_model(model):
    if False:
        while True:
            i = 10
    post_save.connect(update_reference_index_on_save, sender=model)
    post_delete.connect(remove_reference_index_on_delete, sender=model)

def connect_reference_index_signal_handlers(**kwargs):
    if False:
        print('Hello World!')
    for model in ReferenceIndex.tracked_models:
        connect_reference_index_signal_handlers_for_model(model)

def disconnect_reference_index_signal_handlers_for_model(model):
    if False:
        for i in range(10):
            print('nop')
    post_save.disconnect(update_reference_index_on_save, sender=model)
    post_delete.disconnect(remove_reference_index_on_delete, sender=model)

def disconnect_reference_index_signal_handlers(**kwargs):
    if False:
        while True:
            i = 10
    for model in ReferenceIndex.tracked_models:
        disconnect_reference_index_signal_handlers_for_model(model)

def register_signal_handlers():
    if False:
        for i in range(10):
            print('nop')
    post_save.connect(post_save_site_signal_handler, sender=Site)
    post_delete.connect(post_delete_site_signal_handler, sender=Site)
    pre_delete.connect(pre_delete_page_unpublish, sender=Page)
    post_delete.connect(post_delete_page_log_deletion, sender=Page)
    post_save.connect(reset_locales_display_names_cache, sender=Locale)
    post_delete.connect(reset_locales_display_names_cache, sender=Locale)
    pre_migrate.connect(disconnect_reference_index_signal_handlers)
    post_migrate.connect(connect_reference_index_signal_handlers)
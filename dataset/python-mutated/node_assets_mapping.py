from django.db.models.signals import post_save, post_delete, m2m_changed
from django.dispatch import receiver
from django.utils.functional import lazy
from assets.models import Node, Asset
from common.decorators import merge_delay_run
from common.signals import django_ready
from common.utils import get_logger
from common.utils.connection import RedisPubSub
from orgs.models import Organization
logger = get_logger(__name__)
node_assets_mapping_pub_sub = lazy(lambda : RedisPubSub('fm.node_asset_mapping'), RedisPubSub)()

@merge_delay_run(ttl=5)
def expire_node_assets_mapping(org_ids=()):
    if False:
        print('Hello World!')
    logger.debug('Recv asset nodes changed signal, expire memery node asset mapping')
    root_org_id = Organization.ROOT_ID
    Node.expire_node_all_asset_ids_cache_mapping(root_org_id)
    for org_id in set(org_ids):
        org_id = str(org_id)
        Node.expire_node_all_asset_ids_cache_mapping(org_id)
        node_assets_mapping_pub_sub.publish(org_id)

@receiver(post_save, sender=Node)
def on_node_post_create(sender, instance, created, update_fields, **kwargs):
    if False:
        return 10
    if created:
        need_expire = True
    elif update_fields and 'key' in update_fields:
        need_expire = True
    else:
        need_expire = False
    if need_expire:
        expire_node_assets_mapping(org_ids=(instance.org_id,))

@receiver(post_delete, sender=Node)
def on_node_post_delete(sender, instance, **kwargs):
    if False:
        return 10
    expire_node_assets_mapping(org_ids=(instance.org_id,))

@receiver(m2m_changed, sender=Asset.nodes.through)
def on_node_asset_change(sender, instance, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    expire_node_assets_mapping(org_ids=(instance.org_id,))

@receiver(django_ready)
def subscribe_node_assets_mapping_expire(sender, **kwargs):
    if False:
        while True:
            i = 10
    logger.debug('Start subscribe for expire node assets id mapping from memory')

    def handle_node_relation_change(org_id):
        if False:
            return 10
        root_org_id = Organization.ROOT_ID
        Node.expire_node_all_asset_ids_memory_mapping(org_id)
        Node.expire_node_all_asset_ids_memory_mapping(root_org_id)
    node_assets_mapping_pub_sub.subscribe(handle_node_relation_change)
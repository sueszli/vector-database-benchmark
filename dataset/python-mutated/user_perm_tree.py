import time
from collections import defaultdict
from django.conf import settings
from django.core.cache import cache
from assets.models import Asset
from assets.utils import NodeAssetsUtil
from common.db.models import output_as_string
from common.decorators import on_transaction_commit
from common.utils import get_logger
from common.utils.common import lazyproperty, timeit
from orgs.models import Organization
from orgs.utils import current_org, tmp_to_org, tmp_to_root_org
from perms.locks import UserGrantedTreeRebuildLock
from perms.models import AssetPermission, UserAssetGrantedTreeNodeRelation, PermNode
from users.models import User
from .permission import AssetPermissionUtil
logger = get_logger(__name__)
__all__ = ['UserPermTreeRefreshUtil', 'UserPermTreeExpireUtil']

class _UserPermTreeCacheMixin:
    """ 缓存数据 users: {org_id, org_id }, 记录用户授权树已经构建完成的组织集合 """
    cache_key_template = 'perms.user.node_tree.built_orgs.user_id:{user_id}'

    def get_cache_key(self, user_id):
        if False:
            return 10
        return self.cache_key_template.format(user_id=user_id)

    @lazyproperty
    def client(self):
        if False:
            for i in range(10):
                print('nop')
        return cache.client.get_client(write=True)

class UserPermTreeRefreshUtil(_UserPermTreeCacheMixin):
    """ 用户授权树刷新工具, 针对某一个用户授权树的刷新 """

    def __init__(self, user):
        if False:
            i = 10
            return i + 15
        self.user = user
        self.orgs = self.user.orgs.distinct()
        self.org_ids = [str(o.id) for o in self.orgs]

    @lazyproperty
    def cache_key_user(self):
        if False:
            return 10
        return self.get_cache_key(self.user.id)

    @timeit
    def refresh_if_need(self, force=False):
        if False:
            print('Hello World!')
        self._clean_user_perm_tree_for_legacy_org()
        to_refresh_orgs = self.orgs if force else self._get_user_need_refresh_orgs()
        if not to_refresh_orgs:
            logger.info('Not have to refresh orgs')
            return
        with UserGrantedTreeRebuildLock(self.user.id):
            for org in to_refresh_orgs:
                self._rebuild_user_perm_tree_for_org(org)
        self._mark_user_orgs_refresh_finished(to_refresh_orgs)

    def _rebuild_user_perm_tree_for_org(self, org):
        if False:
            print('Hello World!')
        with tmp_to_org(org):
            start = time.time()
            UserPermTreeBuildUtil(self.user).rebuild_user_perm_tree()
            end = time.time()
            logger.info('Refresh user [{user}] org [{org}] perm tree, user {use_time:.2f}s'.format(user=self.user, org=org, use_time=end - start))

    def _clean_user_perm_tree_for_legacy_org(self):
        if False:
            print('Hello World!')
        with tmp_to_root_org():
            ' Clean user legacy org node relations '
            user_relations = UserAssetGrantedTreeNodeRelation.objects.filter(user=self.user)
            user_legacy_org_relations = user_relations.exclude(org_id__in=self.org_ids)
            user_legacy_org_relations.delete()

    def _get_user_need_refresh_orgs(self):
        if False:
            print('Hello World!')
        cached_org_ids = self.client.smembers(self.cache_key_user)
        cached_org_ids = {oid.decode() for oid in cached_org_ids}
        to_refresh_org_ids = set(self.org_ids) - cached_org_ids
        to_refresh_orgs = Organization.objects.filter(id__in=to_refresh_org_ids)
        logger.info(f'Need to refresh orgs: {to_refresh_orgs}')
        return to_refresh_orgs

    def _mark_user_orgs_refresh_finished(self, orgs):
        if False:
            for i in range(10):
                print('nop')
        org_ids = [str(org.id) for org in orgs]
        self.client.sadd(self.cache_key_user, *org_ids)

class UserPermTreeExpireUtil(_UserPermTreeCacheMixin):
    """ 用户授权树过期工具 """

    @lazyproperty
    def cache_key_all_user(self):
        if False:
            return 10
        return self.get_cache_key('*')

    def expire_perm_tree_for_nodes_assets(self, node_ids, asset_ids):
        if False:
            i = 10
            return i + 15
        node_perm_ids = AssetPermissionUtil().get_permissions_for_nodes(node_ids, flat=True)
        asset_perm_ids = AssetPermissionUtil().get_permissions_for_assets(asset_ids, flat=True)
        perm_ids = set(node_perm_ids) | set(asset_perm_ids)
        self.expire_perm_tree_for_perms(perm_ids)

    @tmp_to_root_org()
    def expire_perm_tree_for_perms(self, perm_ids):
        if False:
            return 10
        org_perm_ids = AssetPermission.objects.filter(id__in=perm_ids).values_list('org_id', 'id')
        org_perms_mapper = defaultdict(set)
        for (org_id, perm_id) in org_perm_ids:
            org_perms_mapper[org_id].add(perm_id)
        for (org_id, perms_id) in org_perms_mapper.items():
            user_ids = AssetPermission.get_all_users_for_perms(perm_ids, flat=True)
            self.expire_perm_tree_for_users_orgs(user_ids, [org_id])

    def expire_perm_tree_for_user_group(self, user_group):
        if False:
            for i in range(10):
                print('nop')
        group_ids = [user_group.id]
        org_ids = [user_group.org_id]
        self.expire_perm_tree_for_user_groups_orgs(group_ids, org_ids)

    def expire_perm_tree_for_user_groups_orgs(self, group_ids, org_ids):
        if False:
            for i in range(10):
                print('nop')
        user_ids = User.groups.through.objects.filter(usergroup_id__in=group_ids).values_list('user_id', flat=True).distinct()
        self.expire_perm_tree_for_users_orgs(user_ids, org_ids)

    @on_transaction_commit
    def expire_perm_tree_for_users_orgs(self, user_ids, org_ids):
        if False:
            while True:
                i = 10
        org_ids = [str(oid) for oid in org_ids]
        with self.client.pipeline() as p:
            for uid in user_ids:
                cache_key = self.get_cache_key(uid)
                p.srem(cache_key, *org_ids)
            p.execute()
        logger.info('Expire perm tree for users: [{}], orgs: [{}]'.format(user_ids, org_ids))

    def expire_perm_tree_for_all_user(self):
        if False:
            i = 10
            return i + 15
        keys = self.client.keys(self.cache_key_all_user)
        with self.client.pipeline() as p:
            for k in keys:
                p.delete(k)
            p.execute()
        logger.info('Expire all user perm tree')

class UserPermTreeBuildUtil(object):
    node_only_fields = ('id', 'key', 'parent_key', 'org_id')

    def __init__(self, user):
        if False:
            while True:
                i = 10
        self.user = user
        self.user_perm_ids = AssetPermissionUtil().get_permissions_for_user(self.user, flat=True)
        self._perm_nodes_key_node_mapper = {}

    def rebuild_user_perm_tree(self):
        if False:
            return 10
        self.clean_user_perm_tree()
        if not self.user_perm_ids:
            logger.info('User({}) not have permissions'.format(self.user))
            return
        self.compute_perm_nodes()
        self.compute_perm_nodes_asset_amount()
        self.create_mapping_nodes()

    def clean_user_perm_tree(self):
        if False:
            i = 10
            return i + 15
        UserAssetGrantedTreeNodeRelation.objects.filter(user=self.user).delete()

    def compute_perm_nodes(self):
        if False:
            while True:
                i = 10
        self._compute_perm_nodes_for_direct()
        self._compute_perm_nodes_for_direct_asset_if_need()
        self._compute_perm_nodes_for_ancestor()

    def compute_perm_nodes_asset_amount(self):
        if False:
            i = 10
            return i + 15
        ' 这里计算的是一个组织的授权树 '
        computed = self._only_compute_root_node_assets_amount_if_need()
        if computed:
            return
        nodekey_assetid_mapper = defaultdict(set)
        org_id = current_org.id
        for key in self.perm_node_keys_for_granted:
            asset_ids = PermNode.get_all_asset_ids_by_node_key(org_id, key)
            nodekey_assetid_mapper[key].update(asset_ids)
        for (asset_id, node_id) in self.direct_asset_id_node_id_pairs:
            node_key = self.perm_nodes_id_key_mapper.get(node_id)
            if not node_key:
                continue
            nodekey_assetid_mapper[node_key].add(asset_id)
        util = NodeAssetsUtil(self.perm_nodes, nodekey_assetid_mapper)
        util.generate()
        for node in self.perm_nodes:
            assets_amount = util.get_assets_amount(node.key)
            node.assets_amount = assets_amount

    def create_mapping_nodes(self):
        if False:
            print('Hello World!')
        to_create = []
        for node in self.perm_nodes:
            relation = UserAssetGrantedTreeNodeRelation(user=self.user, node=node, node_key=node.key, node_parent_key=node.parent_key, node_from=node.node_from, node_assets_amount=node.assets_amount, org_id=node.org_id)
            to_create.append(relation)
        UserAssetGrantedTreeNodeRelation.objects.bulk_create(to_create)

    def _compute_perm_nodes_for_direct(self):
        if False:
            for i in range(10):
                print('nop')
        ' 直接授权的节点（叶子节点）'
        for node in self.direct_nodes:
            if self.has_any_ancestor_direct_permed(node):
                continue
            node.node_from = node.NodeFrom.granted
            self._perm_nodes_key_node_mapper[node.key] = node

    def _compute_perm_nodes_for_direct_asset_if_need(self):
        if False:
            return 10
        ' 直接授权的资产所在的节点（叶子节点）'
        if settings.PERM_SINGLE_ASSET_TO_UNGROUP_NODE:
            return
        for node in self.direct_asset_nodes:
            if self.has_any_ancestor_direct_permed(node):
                continue
            if node.key in self._perm_nodes_key_node_mapper:
                continue
            node.node_from = node.NodeFrom.asset
            self._perm_nodes_key_node_mapper[node.key] = node

    def _compute_perm_nodes_for_ancestor(self):
        if False:
            for i in range(10):
                print('nop')
        ' 直接授权节点 和 直接授权资产所在节点 的所有祖先节点 (构造完整树) '
        ancestor_keys = set()
        for node in self._perm_nodes_key_node_mapper.values():
            ancestor_keys.update(node.get_ancestor_keys())
        ancestor_keys -= set(self._perm_nodes_key_node_mapper.keys())
        ancestors = PermNode.objects.filter(key__in=ancestor_keys).only(*self.node_only_fields)
        for node in ancestors:
            node.node_from = node.NodeFrom.child
            self._perm_nodes_key_node_mapper[node.key] = node

    @lazyproperty
    def perm_node_keys_for_granted(self):
        if False:
            return 10
        keys = [key for (key, node) in self._perm_nodes_key_node_mapper.items() if node.node_from == node.NodeFrom.granted]
        return keys

    @lazyproperty
    def perm_nodes_id_key_mapper(self):
        if False:
            for i in range(10):
                print('nop')
        mapper = {node.id.hex: node.key for (key, node) in self._perm_nodes_key_node_mapper.items()}
        return mapper

    def _only_compute_root_node_assets_amount_if_need(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.perm_nodes) != 1:
            return False
        root_node = self.perm_nodes[0]
        if not root_node.is_org_root():
            return False
        if root_node.node_from != root_node.NodeFrom.granted:
            return False
        root_node.granted_assets_amount = len(root_node.get_all_asset_ids())
        return True

    @lazyproperty
    def perm_nodes(self):
        if False:
            print('Hello World!')
        ' 授权树的所有节点 '
        return list(self._perm_nodes_key_node_mapper.values())

    def has_any_ancestor_direct_permed(self, node):
        if False:
            while True:
                i = 10
        ' 任何一个祖先节点被直接授权 '
        return bool(set(node.get_ancestor_keys()) & set(self.direct_node_keys))

    @lazyproperty
    def direct_node_keys(self):
        if False:
            i = 10
            return i + 15
        ' 直接授权的节点 keys '
        return {n.key for n in self.direct_nodes}

    @lazyproperty
    def direct_nodes(self):
        if False:
            i = 10
            return i + 15
        ' 直接授权的节点 '
        node_ids = AssetPermission.nodes.through.objects.filter(assetpermission_id__in=self.user_perm_ids).values_list('node_id', flat=True).distinct()
        nodes = PermNode.objects.filter(id__in=node_ids).only(*self.node_only_fields)
        return nodes

    @lazyproperty
    def direct_asset_nodes(self):
        if False:
            i = 10
            return i + 15
        ' 获取直接授权的资产所在的节点 '
        node_ids = [node_id for (asset_id, node_id) in self.direct_asset_id_node_id_pairs]
        nodes = PermNode.objects.filter(id__in=node_ids).distinct().only(*self.node_only_fields)
        return nodes

    @lazyproperty
    def direct_asset_id_node_id_pairs(self):
        if False:
            while True:
                i = 10
        ' 直接授权的资产 id 和 节点 id  '
        asset_node_pairs = Asset.nodes.through.objects.filter(asset_id__in=self.direct_asset_ids).annotate(str_asset_id=output_as_string('asset_id'), str_node_id=output_as_string('node_id')).values_list('str_asset_id', 'str_node_id')
        asset_node_pairs = list(asset_node_pairs)
        return asset_node_pairs

    @lazyproperty
    def direct_asset_ids(self):
        if False:
            for i in range(10):
                print('nop')
        ' 直接授权的资产 ids '
        asset_ids = AssetPermission.assets.through.objects.filter(assetpermission_id__in=self.user_perm_ids).values_list('asset_id', flat=True).distinct()
        return asset_ids
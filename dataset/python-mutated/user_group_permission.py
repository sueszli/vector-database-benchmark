from itertools import chain
from django.db.models import Q
from rest_framework.generics import ListAPIView
from rest_framework.response import Response
from assets.api.mixin import SerializeToTreeNodeMixin
from assets.models import Asset, Node
from perms import serializers
from perms.models import AssetPermission
__all__ = ['UserGroupGrantedAssetsApi', 'UserGroupGrantedNodesApi', 'UserGroupGrantedNodeAssetsApi', 'UserGroupGrantedNodeChildrenAsTreeApi']

class UserGroupGrantedAssetsApi(ListAPIView):
    serializer_class = serializers.AssetPermedSerializer
    only_fields = serializers.AssetPermedSerializer.Meta.only_fields
    filterset_fields = ['name', 'address', 'id', 'comment']
    search_fields = ['name', 'address', 'comment']
    rbac_perms = {'list': 'perms.view_usergroupassets'}

    def get_queryset(self):
        if False:
            while True:
                i = 10
        user_group_id = self.kwargs.get('pk')
        if not user_group_id:
            return Asset.objects.none()
        asset_perm_ids = list(AssetPermission.objects.valid().filter(user_groups__id=user_group_id).distinct().values_list('id', flat=True))
        granted_node_keys = Node.objects.filter(granted_by_permissions__id__in=asset_perm_ids).distinct().values_list('key', flat=True)
        granted_q = Q()
        for _key in granted_node_keys:
            granted_q |= Q(nodes__key__startswith=f'{_key}:')
            granted_q |= Q(nodes__key=_key)
        granted_q |= Q(granted_by_permissions__id__in=asset_perm_ids)
        assets = Asset.objects.filter(granted_q).only(*self.only_fields).distinct()
        return assets

class UserGroupGrantedNodeAssetsApi(ListAPIView):
    serializer_class = serializers.AssetPermedSerializer
    only_fields = serializers.AssetPermedSerializer.Meta.only_fields
    filterset_fields = ['name', 'address', 'id', 'comment']
    search_fields = ['name', 'address', 'comment']
    rbac_perms = {'list': 'perms.view_usergroupassets'}

    def get_queryset(self):
        if False:
            i = 10
            return i + 15
        if getattr(self, 'swagger_fake_view', False):
            return Asset.objects.none()
        user_group_id = self.kwargs.get('pk', '')
        node_id = self.kwargs.get('node_id')
        node = Node.objects.get(id=node_id)
        granted = AssetPermission.objects.filter(user_groups__id=user_group_id, nodes__id=node_id).valid().exists()
        if granted:
            assets = Asset.objects.filter(Q(nodes__key__startswith=f'{node.key}:') | Q(nodes__key=node.key))
            return assets
        else:
            asset_perm_ids = list(AssetPermission.objects.valid().filter(user_groups__id=user_group_id).distinct().values_list('id', flat=True))
            granted_node_keys = Node.objects.filter(granted_by_permissions__id__in=asset_perm_ids, key__startswith=f'{node.key}:').distinct().values_list('key', flat=True)
            granted_node_q = Q()
            for _key in granted_node_keys:
                granted_node_q |= Q(nodes__key__startswith=f'{_key}:')
                granted_node_q |= Q(nodes__key=_key)
            granted_asset_q = Q(granted_by_permissions__id__in=asset_perm_ids) & (Q(nodes__key__startswith=f'{node.key}:') | Q(nodes__key=node.key))
            assets = Asset.objects.filter(granted_node_q | granted_asset_q).distinct()
            return assets

class UserGroupGrantedNodesApi(ListAPIView):
    serializer_class = serializers.NodePermedSerializer
    rbac_perms = {'list': 'perms.view_usergroupassets'}

    def get_queryset(self):
        if False:
            i = 10
            return i + 15
        user_group_id = self.kwargs.get('pk')
        if not user_group_id:
            return Node.objects.none()
        nodes = Node.objects.filter(Q(granted_by_permissions__user_groups__id=user_group_id) | Q(assets__granted_by_permissions__user_groups__id=user_group_id))
        return nodes

class UserGroupGrantedNodeChildrenAsTreeApi(SerializeToTreeNodeMixin, ListAPIView):
    rbac_perms = {'list': 'perms.view_usergroupassets', 'GET': 'perms.view_usergroupassets'}

    def get_children_nodes(self, parent_key):
        if False:
            print('Hello World!')
        return Node.objects.filter(parent_key=parent_key)

    def add_children_key(self, node_key, key, key_set):
        if False:
            while True:
                i = 10
        if key.startswith(f'{node_key}:'):
            try:
                end = key.index(':', len(node_key) + 1)
                key_set.add(key[:end])
            except ValueError:
                key_set.add(key)

    def get_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        group_id = self.kwargs.get('pk')
        node_key = self.request.query_params.get('key', None)
        asset_perm_ids = list(AssetPermission.objects.valid().filter(user_groups__id=group_id).distinct().values_list('id', flat=True))
        granted_keys = Node.objects.filter(granted_by_permissions__id__in=asset_perm_ids).values_list('key', flat=True)
        asset_granted_keys = Node.objects.filter(assets__granted_by_permissions__id__in=asset_perm_ids).values_list('key', flat=True)
        if node_key is None:
            root_keys = set()
            for _key in chain(granted_keys, asset_granted_keys):
                root_keys.add(_key.split(':', 1)[0])
            return Node.objects.filter(key__in=root_keys)
        else:
            children_keys = set()
            for _key in granted_keys:
                if node_key == _key:
                    return self.get_children_nodes(node_key)
                if node_key.startswith(f'{_key}:'):
                    return self.get_children_nodes(node_key)
                self.add_children_key(node_key, _key, children_keys)
            for _key in asset_granted_keys:
                self.add_children_key(node_key, _key, children_keys)
            return Node.objects.filter(key__in=children_keys)

    def list(self, request, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        nodes = self.get_nodes()
        nodes = self.serialize_nodes(nodes)
        return Response(data=nodes)
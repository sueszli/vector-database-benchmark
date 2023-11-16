from dataclasses import dataclass
from typing import Optional
import graphene
from django.core.exceptions import ValidationError
from ....core.tracing import traced_atomic_transaction
from ....menu import models
from ....menu.error_codes import MenuErrorCode
from ....permission.enums import MenuPermissions
from ....webhook.event_types import WebhookEventAsyncType
from ...channel import ChannelContext
from ...core import ResolveInfo
from ...core.doc_category import DOC_CATEGORY_MENU
from ...core.mutations import BaseMutation
from ...core.types import MenuError, NonNullList
from ...core.utils import WebhookEventInfo
from ...core.utils.reordering import perform_reordering
from ...plugins.dataloaders import get_plugin_manager_promise
from ..dataloaders import MenuItemsByParentMenuLoader
from ..types import Menu, MenuItem, MenuItemMoveInput

@dataclass(frozen=True)
class _MenuMoveOperation:
    menu_item: models.MenuItem
    parent_changed: bool
    new_parent: Optional[models.MenuItem]
    sort_order: int

class MenuItemMove(BaseMutation):
    menu = graphene.Field(Menu, description='Assigned menu to move within.')

    class Arguments:
        menu = graphene.ID(required=True, description='ID of the menu.')
        moves = NonNullList(MenuItemMoveInput, required=True, description='The menu position data.')

    class Meta:
        description = 'Moves items of menus.'
        doc_category = DOC_CATEGORY_MENU
        permissions = (MenuPermissions.MANAGE_MENUS,)
        error_type_class = MenuError
        error_type_field = 'menu_errors'
        webhook_events_info = [WebhookEventInfo(type=WebhookEventAsyncType.MENU_ITEM_UPDATED, description='Optionally triggered when sort order or parent changed for menu item.')]

    @classmethod
    def success_response(cls, instance):
        if False:
            print('Hello World!')
        instance = ChannelContext(node=instance, channel_slug=None)
        return super().success_response(instance)

    @staticmethod
    def clean_move(move: MenuItemMoveInput):
        if False:
            i = 10
            return i + 15
        'Validate if the given move could be possibly possible.'
        if move.parent_id:
            if move.item_id == move.parent_id:
                raise ValidationError({'parent_id': ValidationError('Cannot assign a node to itself.', code=MenuErrorCode.CANNOT_ASSIGN_NODE.value)})

    @staticmethod
    def clean_operation(operation: _MenuMoveOperation):
        if False:
            print('Hello World!')
        'Validate if the given move will be actually possible.'
        if operation.new_parent is not None:
            if operation.menu_item.is_ancestor_of(operation.new_parent):
                raise ValidationError({'parent_id': ValidationError('Cannot assign a node as child of one of its descendants.', code=MenuErrorCode.CANNOT_ASSIGN_NODE.value)})

    @classmethod
    def get_operation(cls, info: ResolveInfo, menu_item_to_current_parent, menu: models.Menu, move: MenuItemMoveInput) -> _MenuMoveOperation:
        if False:
            print('Hello World!')
        menu_item = cls.get_node_or_error(info, move.item_id, field='item', only_type=MenuItem, qs=menu.items)
        (new_parent, parent_changed) = (None, False)
        old_parent_id = menu_item_to_current_parent[menu_item.pk] if menu_item.pk in menu_item_to_current_parent else menu_item.parent_id
        if move.parent_id is not None:
            parent_pk = cls.get_global_id_or_error(move.parent_id, only_type=MenuItem, field='parent_id')
            if int(parent_pk) != old_parent_id:
                new_parent = cls.get_node_or_error(info, move.parent_id, field='parent_id', only_type=MenuItem, qs=menu.items)
                parent_changed = True
        elif move.parent_id is None and old_parent_id is not None:
            parent_changed = True
        return _MenuMoveOperation(menu_item=menu_item, new_parent=new_parent, parent_changed=parent_changed, sort_order=move.sort_order)

    @classmethod
    def clean_moves(cls, info: ResolveInfo, menu: models.Menu, move_operations: list[MenuItemMoveInput]) -> list[_MenuMoveOperation]:
        if False:
            i = 10
            return i + 15
        operations = []
        item_to_current_parent: dict[int, Optional[models.MenuItem]] = {}
        for move in move_operations:
            cls.clean_move(move)
            operation = cls.get_operation(info, item_to_current_parent, menu, move)
            if operation.parent_changed:
                cls.clean_operation(operation)
                item_to_current_parent[operation.menu_item.id] = operation.new_parent
            operations.append(operation)
        return operations

    @staticmethod
    def perform_change_parent_operation(operation: _MenuMoveOperation):
        if False:
            print('Hello World!')
        menu_item = operation.menu_item
        if not operation.parent_changed:
            return
        menu_item.refresh_from_db()
        menu_item._mptt_meta.update_mptt_cached_fields(menu_item)
        menu_item.parent = operation.new_parent
        menu_item.sort_order = None
        menu_item.save()

    @classmethod
    def perform_mutation(cls, _root, info: ResolveInfo, /, **data):
        if False:
            while True:
                i = 10
        menu: str = data['menu']
        moves: list[MenuItemMoveInput] = data['moves']
        qs = models.Menu.objects.prefetch_related('items')
        menu = cls.get_node_or_error(info, menu, only_type=Menu, field='menu', qs=qs)
        operations = cls.clean_moves(info, menu, moves)
        manager = get_plugin_manager_promise(info.context).get()
        with traced_atomic_transaction():
            for operation in operations:
                cls.perform_change_parent_operation(operation)
                menu_item = operation.menu_item
                if operation.sort_order:
                    perform_reordering(menu_item.get_ordering_queryset(), {menu_item.pk: operation.sort_order})
                if operation.sort_order or operation.parent_changed:
                    cls.call_event(manager.menu_item_updated, menu_item)
        menu = qs.get(pk=menu.pk)
        MenuItemsByParentMenuLoader(info.context).clear(menu.id)
        return MenuItemMove(menu=ChannelContext(node=menu, channel_slug=None))
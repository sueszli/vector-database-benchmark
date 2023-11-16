from collections import OrderedDict
from dataclasses import dataclass
from django.db import transaction
from django.db.models import F, QuerySet
from django.utils.functional import cached_property
__all__ = ['perform_reordering']

@dataclass(frozen=True)
class FinalSortOrder:
    """Describe a final sort order value for a given PK.

    This is needed to tell django which objects and values to associate and update.
    """
    pk: int
    sort_order: int

class Reordering:

    def __init__(self, qs: QuerySet, operations: dict[int, int], field: str):
        if False:
            print('Hello World!')
        self.qs = qs
        self.operations = operations
        self.field = field
        self.old_sort_map: dict[int, str] = {}
        self.ordered_pks: list[int] = []

    @cached_property
    def ordered_node_map(self):
        if False:
            for i in range(10):
                print('nop')
        ordering_map = OrderedDict(self.qs.select_for_update().values_list('pk', 'sort_order').order_by(F('sort_order').asc(nulls_last=True), 'id'))
        self.old_sort_map = ordering_map.copy()
        self.ordered_pks = list(ordering_map.keys())
        previous_sort_order = 0
        for (pk, sort_order) in ordering_map.items():
            if sort_order is not None:
                previous_sort_order = sort_order
                continue
            previous_sort_order += 1
            ordering_map[pk] = previous_sort_order
        return ordering_map

    def calculate_new_sort_order(self, pk, move) -> tuple[int, int, int]:
        if False:
            print('Hello World!')
        'Return the proper sort order for the current operation.\n\n        Allows to properly move the node in a given direction with by amount.\n\n        This ensures the new sort order is not biased from gaps between the sort orders.\n        '
        node_pos = self.ordered_pks.index(pk)
        target_pos = node_pos + move
        target_pos = max(0, target_pos)
        target_pos = min(len(self.ordered_pks) - 1, target_pos)
        target_pk = self.ordered_pks[target_pos]
        target_position = self.ordered_node_map[target_pk]
        return (node_pos, target_pos, target_position)

    def process_move_operation(self, pk, move):
        if False:
            return 10
        old_sort_order = self.ordered_node_map[pk]
        if move == 0:
            return
        if move is None:
            move = +1
        (node_pos, target_pos, new_sort_order) = self.calculate_new_sort_order(pk, move)
        if move > 0:
            shift = -1
            range_ = (old_sort_order + 1, new_sort_order)
        else:
            shift = +1
            range_ = (new_sort_order, old_sort_order - 1)
        self.add_to_sort_value_if_in_range(shift, *range_)
        self.ordered_node_map[pk] = new_sort_order
        self.ordered_pks.remove(pk)
        self.ordered_pks.insert(target_pos, pk)

    def add_to_sort_value_if_in_range(self, value_to_add, start, end):
        if False:
            return 10
        for (pk, sort_order) in self.ordered_node_map.items():
            if not start <= sort_order <= end:
                continue
            self.ordered_node_map[pk] += value_to_add

    def commit(self):
        if False:
            return 10
        if not self.old_sort_map:
            return
        batch = [FinalSortOrder(pk, sort_order) for (pk, sort_order) in self.ordered_node_map.items() if sort_order != self.old_sort_map[pk]]
        if not batch:
            return
        self.qs.model.objects.bulk_update(batch, ['sort_order'])

    def run(self):
        if False:
            return 10
        for (pk, move) in self.operations.items():
            if pk not in self.ordered_node_map:
                continue
            self.process_move_operation(pk, move)
        self.commit()

def perform_reordering(qs: QuerySet, operations: dict[int, int], field: str='moves'):
    if False:
        i = 10
        return i + 15
    'Perform reordering over given operations on a queryset.\n\n    This utility takes a set of operations containing a node\n    and a relative sort order. It then converts the relative sorting\n    to an absolute sorting.\n\n    This will then commit the changes onto the nodes.\n\n    :param qs: The query set on which we want to retrieve and reorder the node.\n    :param operations: The operations to make: {pk_to_move: +/- 123}.\n    :param field: The field from which nodes are resolved.\n\n    :raises RuntimeError: If the bulk operation is not run inside an atomic transaction.\n    '
    if not transaction.get_connection().in_atomic_block:
        raise RuntimeError('Needs to be run inside an atomic transaction')
    Reordering(qs, operations, field).run()
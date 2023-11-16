from django.db import transaction
from posthog.demo.matrix import MatrixManager
from posthog.demo.products import HedgeboxMatrix

def demo_reset_master_team() -> None:
    if False:
        return 10
    matrix = HedgeboxMatrix()
    manager = MatrixManager(matrix)
    with transaction.atomic():
        manager.reset_master()
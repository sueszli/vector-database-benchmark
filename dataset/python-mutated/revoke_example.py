from time import sleep
from tasks import identity_task, mul, wait_for_revoke, xsum
from visitors import MonitoringIdStampingVisitor
from celery.canvas import Signature, chain, chord, group
from celery.result import AsyncResult

def create_canvas(n: int) -> Signature:
    if False:
        for i in range(10):
            print('nop')
    'Creates a canvas to calculate: n * sum(1..n) * 10\n    For example, if n = 3, the result is 3 * (1 + 2 + 3) * 10 = 180\n    '
    canvas = chain(group((identity_task.s(i) for i in range(1, n + 1))) | xsum.s(), chord(group((mul.s(10) for _ in range(1, n + 1))), xsum.s()))
    return canvas

def revoke_by_headers(result: AsyncResult, terminate: bool) -> None:
    if False:
        return 10
    'Revokes the last task in the workflow by its stamped header\n\n    Arguments:\n        result (AsyncResult): Can be either a frozen or a running result\n        terminate (bool): If True, the revoked task will be terminated\n    '
    result.revoke_by_stamped_headers({'mystamp': 'I am a stamp!'}, terminate=terminate)

def prepare_workflow() -> Signature:
    if False:
        i = 10
        return i + 15
    'Creates a canvas that waits "n * sum(1..n) * 10" in seconds,\n    with n = 3.\n\n    The canvas itself is stamped with a unique monitoring id stamp per task.\n    The waiting task is stamped with different consistent stamp, which is used\n    to revoke the task by its stamped header.\n    '
    canvas = create_canvas(n=3)
    canvas = canvas | wait_for_revoke.s()
    canvas.stamp(MonitoringIdStampingVisitor())
    return canvas

def run_then_revoke():
    if False:
        while True:
            i = 10
    'Runs the workflow and lets the waiting task run for a while.\n    Then, the waiting task is revoked by its stamped header.\n\n    The expected outcome is that the canvas will be calculated to the end,\n    but the waiting task will be revoked and terminated *during its run*.\n\n    See worker logs for more details.\n    '
    canvas = prepare_workflow()
    result = canvas.delay()
    print('Wait 5 seconds, then revoke the last task by its stamped header: "mystamp": "I am a stamp!"')
    sleep(5)
    print('Revoking the last task...')
    revoke_by_headers(result, terminate=True)

def revoke_then_run():
    if False:
        return 10
    'Revokes the waiting task by its stamped header before it runs.\n    Then, run the workflow, which will not run the waiting task that was revoked.\n\n    The expected outcome is that the canvas will be calculated to the end,\n    but the waiting task will not run at all.\n\n    See worker logs for more details.\n    '
    canvas = prepare_workflow()
    result = canvas.freeze()
    revoke_by_headers(result, terminate=False)
    result = canvas.delay()
from dagster import HookContext, failure_hook
import traceback

@failure_hook
def my_failure_hook(context: HookContext):
    if False:
        i = 10
        return i + 15
    op_exception: BaseException = context.op_exception
    traceback.print_tb(op_exception.__traceback__)
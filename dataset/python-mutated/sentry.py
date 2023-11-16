import functools
import os
import sentry_sdk
from dagster import AssetExecutionContext, OpExecutionContext, SensorEvaluationContext, get_dagster_logger
sentry_logger = get_dagster_logger('sentry')

def setup_dagster_sentry():
    if False:
        i = 10
        return i + 15
    '\n    Setup the sentry SDK for Dagster if SENTRY_DSN is defined for the environment.\n\n    Additionally TRACES_SAMPLE_RATE can be set 0-1 otherwise will default to 0.\n\n    Manually sets up a bunch of the default integrations and disables logging of dagster\n    to quiet things down.\n    '
    from sentry_sdk.integrations.argv import ArgvIntegration
    from sentry_sdk.integrations.atexit import AtexitIntegration
    from sentry_sdk.integrations.dedupe import DedupeIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration, ignore_logger
    from sentry_sdk.integrations.modules import ModulesIntegration
    from sentry_sdk.integrations.stdlib import StdlibIntegration
    ignore_logger('dagster')
    SENTRY_DSN = os.environ.get('SENTRY_DSN')
    SENTRY_ENVIRONMENT = os.environ.get('SENTRY_ENVIRONMENT')
    TRACES_SAMPLE_RATE = float(os.environ.get('SENTRY_TRACES_SAMPLE_RATE', 0))
    sentry_logger.info('Setting up Sentry with')
    sentry_logger.info(f'SENTRY_DSN: {SENTRY_DSN}')
    sentry_logger.info(f'SENTRY_ENVIRONMENT: {SENTRY_ENVIRONMENT}')
    sentry_logger.info(f'SENTRY_TRACES_SAMPLE_RATE: {TRACES_SAMPLE_RATE}')
    if SENTRY_DSN:
        sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=TRACES_SAMPLE_RATE, environment=SENTRY_ENVIRONMENT, default_integrations=False, integrations=[AtexitIntegration(), DedupeIntegration(), StdlibIntegration(), ModulesIntegration(), ArgvIntegration(), LoggingIntegration()])

def _is_context(context):
    if False:
        return 10
    '\n    Check if the given object is a valid context object.\n    '
    return isinstance(context, OpExecutionContext) or isinstance(context, SensorEvaluationContext) or isinstance(context, AssetExecutionContext)

def _get_context_from_args_kwargs(args, kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Given args and kwargs from a function call, return the context object if it exists.\n    '
    if len(args) > 0 and _is_context(args[0]):
        return args[0]
    if 'context' in kwargs and _is_context(kwargs['context']):
        return kwargs['context']
    raise Exception(f'No context provided to Sentry Transaction. When using @instrument, ensure that the asset/op has a context as the first argument.')

def _with_sentry_op_asset_transaction(context: OpExecutionContext):
    if False:
        for i in range(10):
            print('nop')
    '\n    Start or continue a Sentry transaction for the Dagster Op/Asset\n    '
    op_name = context.op_def.name
    job_name = context.job_name
    sentry_logger.debug(f'Initializing Sentry Transaction for Dagster Op/Asset {job_name} - {op_name}')
    transaction = sentry_sdk.Hub.current.scope.transaction
    sentry_logger.debug(f'Current Sentry Transaction: {transaction}')
    if transaction:
        return transaction.start_child(op=op_name)
    else:
        return sentry_sdk.start_transaction(op=op_name, name=job_name)

def capture_asset_op_context(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Capture Dagster OP context for Sentry Error handling\n    '

    @functools.wraps(func)
    def wrapped_fn(*args, **kwargs):
        if False:
            print('Hello World!')
        context = _get_context_from_args_kwargs(args, kwargs)
        with sentry_sdk.configure_scope() as scope:
            scope.set_transaction_name(context.job_name)
            scope.set_tag('job_name', context.job_name)
            scope.set_tag('op_name', context.op_def.name)
            scope.set_tag('run_id', context.run_id)
            scope.set_tag('retry_number', context.retry_number)
            return func(*args, **kwargs)
    return wrapped_fn

def capture_sensor_context(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Capture Dagster Sensor context for Sentry Error handling\n    '

    @functools.wraps(func)
    def wrapped_fn(*args, **kwargs):
        if False:
            return 10
        context = _get_context_from_args_kwargs(args, kwargs)
        with sentry_sdk.configure_scope() as scope:
            scope.set_transaction_name(context._sensor_name)
            scope.set_tag('sensor_name', context._sensor_name)
            scope.set_tag('run_id', context.cursor)
            return func(*args, **kwargs)
    return wrapped_fn

def capture_exceptions(func):
    if False:
        i = 10
        return i + 15
    "\n    Note: This is nessesary as Dagster captures exceptions and logs them before Sentry can.\n\n    Captures exceptions thrown by Dagster Ops and forwards them to Sentry\n    before re-throwing them for Dagster.\n\n    Expects ops to receive Dagster context as the first argument,\n    but it will continue if it doesn't (it just won't get as much context).\n\n    It will log a unique ID that can be then entered into Sentry to find\n    the exception.\n    "

    @functools.wraps(func)
    def wrapped_fn(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            event_id = sentry_sdk.capture_exception(e)
            sentry_logger.info(f'Sentry captured an exception. Event ID: {event_id}')
            raise e
    return wrapped_fn

def start_sentry_transaction(func):
    if False:
        print('Hello World!')
    '\n    Start a Sentry transaction for the Dagster Op/Asset\n    '

    def wrapped_fn(*args, **kwargs):
        if False:
            return 10
        context = _get_context_from_args_kwargs(args, kwargs)
        with _with_sentry_op_asset_transaction(context):
            return func(*args, **kwargs)
    return wrapped_fn

def instrument_asset_op(func):
    if False:
        print('Hello World!')
    "\n    Instrument a Dagster Op/Asset with Sentry.\n\n    This should be used as a decorator after Dagster's `@op`, or `@asset`\n    and the function to be handled.\n\n    This will start a Sentry transaction for the Op/Asset and capture\n    any exceptions thrown by the Op/Asset and forward them to Sentry\n    before re-throwing them for Dagster.\n\n    This will also send traces to Sentry to help with debugging and performance monitoring.\n    "

    @functools.wraps(func)
    @start_sentry_transaction
    @capture_asset_op_context
    @capture_exceptions
    def wrapped_fn(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return func(*args, **kwargs)
    return wrapped_fn

def instrument_sensor(func):
    if False:
        while True:
            i = 10
    "\n    Instrument a Dagster Sensor with Sentry.\n\n    This should be used as a decorator after Dagster's `@sensor`\n    and the function to be handled.\n\n    This will start a Sentry transaction for the Sensor and capture\n    any exceptions thrown by the Sensor and forward them to Sentry\n    before re-throwing them for Dagster.\n\n    "

    @functools.wraps(func)
    @capture_sensor_context
    @capture_exceptions
    def wrapped_fn(*args, **kwargs):
        if False:
            return 10
        return func(*args, **kwargs)
    return wrapped_fn
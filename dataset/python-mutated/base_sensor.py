import uuid
from metaflow.decorators import FlowDecorator
from ..exception import AirflowException
from ..airflow_utils import AirflowTask, id_creator, TASK_ID_HASH_LEN

class AirflowSensorDecorator(FlowDecorator):
    """
    Base class for all Airflow sensor decorators.
    """
    allow_multiple = True
    defaults = dict(timeout=3600, poke_interval=60, mode='reschedule', exponential_backoff=True, pool=None, soft_fail=False, name=None, description=None)
    operator_type = None

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._airflow_task_name = None
        self._id = str(uuid.uuid4())

    def serialize_operator_args(self):
        if False:
            print('Hello World!')
        '\n        Subclasses will parse the decorator arguments to\n        Airflow task serializable arguments.\n        '
        task_args = dict(**self.attributes)
        del task_args['name']
        if task_args['description'] is not None:
            task_args['doc'] = task_args['description']
        del task_args['description']
        task_args['do_xcom_push'] = True
        return task_args

    def create_task(self):
        if False:
            i = 10
            return i + 15
        task_args = self.serialize_operator_args()
        return AirflowTask(self._airflow_task_name, operator_type=self.operator_type).set_operator_args(**{k: v for (k, v) in task_args.items() if v is not None})

    def validate(self):
        if False:
            return 10
        '\n        Validate if the arguments for the sensor are correct.\n        '
        if self.attributes['name'] is None:
            deco_index = [d._id for d in self._flow_decorators if issubclass(d.__class__, AirflowSensorDecorator)].index(self._id)
            self._airflow_task_name = '%s-%s' % (self.operator_type, id_creator([self.operator_type, str(deco_index)], TASK_ID_HASH_LEN))
        else:
            self._airflow_task_name = self.attributes['name']

    def flow_init(self, flow, graph, environment, flow_datastore, metadata, logger, echo, options):
        if False:
            return 10
        self.validate()
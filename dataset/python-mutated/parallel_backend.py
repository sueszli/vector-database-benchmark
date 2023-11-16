from typing import Any, Dict, List, Optional, Type, Union

class ParallelBackend:
    """The parallel backend base class for different parallel implementations.
    None of the methods of this class should be called by users.
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._exp_class: Optional[Type] = None
        self._instance_pack: Any = None

    def attach(self, instance: Any):
        if False:
            while True:
                i = 10
        'Attach the current setup function to this backend.\n\n        instance: Any\n            The ``_PyCaretExperiment`` instance\n        '
        self._instance_pack = instance._pack_for_remote()
        self._exp_class = type(instance)

    def remote_setup(self) -> Any:
        if False:
            return 10
        'Call setup on a worker.'
        instance = self._exp_class()
        params = dict(self._instance_pack['_setup_params'])
        params['verbose'] = False
        params['html'] = False
        params['session_id'] = self._instance_pack.get('seed', None)
        instance._remote = True
        instance.setup(**params)
        instance._unpack_at_remote(self._instance_pack)
        return instance

    def compare_models(self, instance: Any, params: Dict[str, Any]) -> Union[Any, List[Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Distributed ``compare_models`` wrapper.\n\n        instance: Any\n            The ``_PyCaretExperiment`` instance\n\n        params: Dict[str, Any]\n            The parameters used to call the ``compare_models`` function\n        '
        raise NotImplementedError
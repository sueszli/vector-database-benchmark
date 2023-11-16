from dataclasses import dataclass, field
import pathlib
import pprint
from typing import Any, Callable, Dict, KeysView, List, Mapping, Optional, Set, Type, Union
from ray.util.annotations import PublicAPI
from ray.rllib.utils.annotations import override, ExperimentalAPI
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.core.rl_module.rl_module import RLModule, RLMODULE_METADATA_FILE_NAME, RLMODULE_STATE_DIR_NAME, SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.serialization import serialize_type, deserialize_type
from ray.rllib.utils.typing import T
ModuleID = str

@PublicAPI(stability='alpha')
class MultiAgentRLModule(RLModule):
    """Base class for multi-agent RLModules.

    This class holds a mapping from module_ids to the underlying RLModules. It provides
    a convenient way of accessing each individual module, as well as accessing all of
    them with only one API call. Whether or not a given module is trainable is
    determined by the caller of this class (not the instance of this class itself).

    The extension of this class can include any arbitrary neural networks as part of
    the multi-agent module. For example, a multi-agent module can include a shared
    encoder network that is used by all the individual RLModules. It is up to the user
    to decide how to implement this class.

    The default implementation assumes the data communicated as input and output of
    the APIs in this class are `MultiAgentBatch` types. The `MultiAgentRLModule` simply
    loops through each `module_id`, and runs the forward pass of the corresponding
    `RLModule` object with the associated `SampleBatch` within the `MultiAgentBatch`.
    It also assumes that the underlying RLModules do not share any parameters or
    communication with one another. The behavior of modules with such advanced
    communication would be undefined by default. To share parameters or communication
    between the underlying RLModules, you should implement your own
    `MultiAgentRLModule` subclass.
    """

    def __init__(self, config: Optional['MultiAgentRLModuleConfig']=None) -> None:
        if False:
            i = 10
            return i + 15
        'Initializes a MultiagentRLModule instance.\n\n        Args:\n            config: The MultiAgentRLModuleConfig to use.\n        '
        super().__init__(config or MultiAgentRLModuleConfig())

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        'Sets up the underlying RLModules.'
        self._rl_modules = {}
        self.__check_module_configs(self.config.modules)
        for (module_id, module_spec) in self.config.modules.items():
            self._rl_modules[module_id] = module_spec.build()

    @classmethod
    def __check_module_configs(cls, module_configs: Dict[ModuleID, Any]):
        if False:
            for i in range(10):
                print('nop')
        'Checks the module configs for validity.\n\n        The module_configs be a mapping from module_ids to SingleAgentRLModuleSpec\n        objects.\n\n        Args:\n            module_configs: The module configs to check.\n\n        Raises:\n            ValueError: If the module configs are invalid.\n        '
        for (module_id, module_spec) in module_configs.items():
            if not isinstance(module_spec, SingleAgentRLModuleSpec):
                raise ValueError(f'Module {module_id} is not a SingleAgentRLModuleSpec object.')

    def keys(self) -> KeysView[ModuleID]:
        if False:
            return 10
        'Returns a keys view over the module IDs in this MultiAgentRLModule.'
        return self._rl_modules.keys()

    @override(RLModule)
    def as_multi_agent(self) -> 'MultiAgentRLModule':
        if False:
            for i in range(10):
                print('nop')
        'Returns a multi-agent wrapper around this module.\n\n        This method is overridden to avoid double wrapping.\n\n        Returns:\n            The instance itself.\n        '
        return self

    def add_module(self, module_id: ModuleID, module: RLModule, *, override: bool=False) -> None:
        if False:
            return 10
        'Adds a module at run time to the multi-agent module.\n\n        Args:\n            module_id: The module ID to add. If the module ID already exists and\n                override is False, an error is raised. If override is True, the module\n                is replaced.\n            module: The module to add.\n            override: Whether to override the module if it already exists.\n\n        Raises:\n            ValueError: If the module ID already exists and override is False.\n            Warnings are raised if the module id is not valid according to the logic of\n            validate_policy_id().\n        '
        validate_policy_id(module_id)
        if module_id in self._rl_modules and (not override):
            raise ValueError(f'Module ID {module_id} already exists. If your intention is to override, set override=True.')
        self._rl_modules[module_id] = module

    def remove_module(self, module_id: ModuleID, *, raise_err_if_not_found: bool=True) -> None:
        if False:
            while True:
                i = 10
        'Removes a module at run time from the multi-agent module.\n\n        Args:\n            module_id: The module ID to remove.\n            raise_err_if_not_found: Whether to raise an error if the module ID is not\n                found.\n        Raises:\n            ValueError: If the module ID does not exist and raise_err_if_not_found is\n                True.\n        '
        if raise_err_if_not_found:
            self._check_module_exists(module_id)
        del self._rl_modules[module_id]

    def foreach_module(self, func: Callable[[ModuleID, RLModule, Optional[Any]], T], **kwargs) -> List[T]:
        if False:
            i = 10
            return i + 15
        'Calls the given function with each (module_id, module).\n\n        Args:\n            func: The function to call with each (module_id, module) tuple.\n\n        Returns:\n            The lsit of return values of all calls to\n            `func([module_id, module, **kwargs])`.\n        '
        return [func(module_id, module, **kwargs) for (module_id, module) in self._rl_modules.items()]

    def __getitem__(self, module_id: ModuleID) -> RLModule:
        if False:
            while True:
                i = 10
        'Returns the module with the given module ID.\n\n        Args:\n            module_id: The module ID to get.\n\n        Returns:\n            The module with the given module ID.\n        '
        self._check_module_exists(module_id)
        return self._rl_modules[module_id]

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        if False:
            return 10
        return []

    @override(RLModule)
    def output_specs_inference(self) -> SpecType:
        if False:
            i = 10
            return i + 15
        return []

    @override(RLModule)
    def output_specs_exploration(self) -> SpecType:
        if False:
            return 10
        return []

    @override(RLModule)
    def _default_input_specs(self) -> SpecType:
        if False:
            return 10
        'Multi-agent RLModule should not check the input specs.\n\n        The underlying single-agent RLModules will check the input specs.\n        '
        return []

    @override(RLModule)
    def _forward_train(self, batch: MultiAgentBatch, **kwargs) -> Union[Mapping[str, Any], Dict[ModuleID, Mapping[str, Any]]]:
        if False:
            for i in range(10):
                print('nop')
        'Runs the forward_train pass.\n\n        TODO(avnishn, kourosh): Review type hints for forward methods.\n\n        Args:\n            batch: The batch of multi-agent data (i.e. mapping from module ids to\n                SampleBaches).\n\n        Returns:\n            The output of the forward_train pass the specified modules.\n        '
        return self._run_forward_pass('forward_train', batch, **kwargs)

    @override(RLModule)
    def _forward_inference(self, batch: MultiAgentBatch, **kwargs) -> Union[Mapping[str, Any], Dict[ModuleID, Mapping[str, Any]]]:
        if False:
            return 10
        'Runs the forward_inference pass.\n\n        TODO(avnishn, kourosh): Review type hints for forward methods.\n\n        Args:\n            batch: The batch of multi-agent data (i.e. mapping from module ids to\n                SampleBaches).\n\n        Returns:\n            The output of the forward_inference pass the specified modules.\n        '
        return self._run_forward_pass('forward_inference', batch, **kwargs)

    @override(RLModule)
    def _forward_exploration(self, batch: MultiAgentBatch, **kwargs) -> Union[Mapping[str, Any], Dict[ModuleID, Mapping[str, Any]]]:
        if False:
            while True:
                i = 10
        'Runs the forward_exploration pass.\n\n        TODO(avnishn, kourosh): Review type hints for forward methods.\n\n        Args:\n            batch: The batch of multi-agent data (i.e. mapping from module ids to\n                SampleBaches).\n\n        Returns:\n            The output of the forward_exploration pass the specified modules.\n        '
        return self._run_forward_pass('forward_exploration', batch, **kwargs)

    @override(RLModule)
    def get_state(self, module_ids: Optional[Set[ModuleID]]=None) -> Mapping[ModuleID, Any]:
        if False:
            return 10
        'Returns the state of the multi-agent module.\n\n        This method returns the state of each module specified by module_ids. If\n        module_ids is None, the state of all modules is returned.\n\n        Args:\n            module_ids: The module IDs to get the state of. If None, the state of all\n                modules is returned.\n        Returns:\n            A nested state dict with the first layer being the module ID and the second\n            is the state of the module. The returned dict values are framework-specific\n            tensors.\n        '
        if module_ids is None:
            module_ids = self._rl_modules.keys()
        return {module_id: self._rl_modules[module_id].get_state() for module_id in module_ids}

    @override(RLModule)
    def set_state(self, state_dict: Mapping[ModuleID, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sets the state of the multi-agent module.\n\n        It is assumed that the state_dict is a mapping from module IDs to their\n        corressponding state. This method sets the state of each module by calling\n        their set_state method. If you want to set the state of some of the RLModules\n        within this MultiAgentRLModule your state_dict can only include the state of\n        those RLModules. Override this method to customize the state_dict for custom\n        more advanced multi-agent use cases.\n\n        Args:\n            state_dict: The state dict to set.\n        '
        for (module_id, state) in state_dict.items():
            self._rl_modules[module_id].set_state(state)

    @override(RLModule)
    def save_state(self, path: Union[str, pathlib.Path]) -> None:
        if False:
            i = 10
            return i + 15
        'Saves the weights of this MultiAgentRLModule to dir.\n\n        Args:\n            path: The path to the directory to save the checkpoint to.\n\n        '
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for (module_id, module) in self._rl_modules.items():
            module.save_to_checkpoint(str(path / module_id))

    @override(RLModule)
    def load_state(self, path: Union[str, pathlib.Path], modules_to_load: Optional[Set[ModuleID]]=None) -> None:
        if False:
            return 10
        'Loads the weights of an MultiAgentRLModule from dir.\n\n        NOTE:\n            If you want to load a module that is not already\n            in this MultiAgentRLModule, you should add it to this MultiAgentRLModule\n            before loading the checkpoint.\n\n        Args:\n            path: The path to the directory to load the state from.\n            modules_to_load: The modules whose state is to be loaded from the path. If\n                this is None, all modules that are checkpointed will be loaded into this\n                marl module.\n\n\n        '
        path = pathlib.Path(path)
        if not modules_to_load:
            modules_to_load = set(self._rl_modules.keys())
        path.mkdir(parents=True, exist_ok=True)
        for submodule_id in modules_to_load:
            if submodule_id not in self._rl_modules:
                raise ValueError(f'Module {submodule_id} from `modules_to_load`: {modules_to_load} not found in this MultiAgentRLModule.')
            submodule = self._rl_modules[submodule_id]
            submodule_weights_dir = path / submodule_id / RLMODULE_STATE_DIR_NAME
            if not submodule_weights_dir.exists():
                raise ValueError(f"Submodule {submodule_id}'s module state directory: {submodule_weights_dir} not found in checkpoint dir {path}.")
            submodule.load_state(submodule_weights_dir)

    @override(RLModule)
    def save_to_checkpoint(self, checkpoint_dir_path: Union[str, pathlib.Path]) -> None:
        if False:
            while True:
                i = 10
        path = pathlib.Path(checkpoint_dir_path)
        path.mkdir(parents=True, exist_ok=True)
        self.save_state(path)
        self._save_module_metadata(path, MultiAgentRLModuleSpec)

    @classmethod
    @override(RLModule)
    def from_checkpoint(cls, checkpoint_dir_path: Union[str, pathlib.Path]) -> None:
        if False:
            i = 10
            return i + 15
        path = pathlib.Path(checkpoint_dir_path)
        metadata_path = path / RLMODULE_METADATA_FILE_NAME
        marl_module = cls._from_metadata_file(metadata_path)
        marl_module.load_state(path)
        return marl_module

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'MARL({pprint.pformat(self._rl_modules)})'

    def _run_forward_pass(self, forward_fn_name: str, batch: NestedDict[Any], **kwargs) -> Dict[ModuleID, Mapping[ModuleID, Any]]:
        if False:
            while True:
                i = 10
        'This is a helper method that runs the forward pass for the given module.\n\n        It uses forward_fn_name to get the forward pass method from the RLModule\n        (e.g. forward_train vs. forward_exploration) and runs it on the given batch.\n\n        Args:\n            forward_fn_name: The name of the forward pass method to run.\n            batch: The batch of multi-agent data (i.e. mapping from module ids to\n                SampleBaches).\n            **kwargs: Additional keyword arguments to pass to the forward function.\n\n        Returns:\n            The output of the forward pass the specified modules. The output is a\n            mapping from module ID to the output of the forward pass.\n        '
        module_ids = list(batch.shallow_keys())
        for module_id in module_ids:
            self._check_module_exists(module_id)
        outputs = {}
        for module_id in module_ids:
            rl_module = self._rl_modules[module_id]
            forward_fn = getattr(rl_module, forward_fn_name)
            outputs[module_id] = forward_fn(batch[module_id], **kwargs)
        return outputs

    def _check_module_exists(self, module_id: ModuleID) -> None:
        if False:
            return 10
        if module_id not in self._rl_modules:
            raise KeyError(f'Module with module_id {module_id} not found. Available modules: {set(self.keys())}')

@PublicAPI(stability='alpha')
@dataclass
class MultiAgentRLModuleSpec:
    """A utility spec class to make it constructing MARL modules easier.


    Users can extend this class to modify the behavior of base class. For example to
    share neural networks across the modules, the build method can be overriden to
    create the shared module first and then pass it to custom module classes that would
    then use it as a shared module.

    Args:
        marl_module_class: The class of the multi-agent RLModule to construct. By
            default it is set to MultiAgentRLModule class. This class simply loops
            throught each module and calls their foward methods.
        module_specs: The module specs for each individual module. It can be either a
            SingleAgentRLModuleSpec used for all module_ids or a dictionary mapping
            from module IDs to SingleAgentRLModuleSpecs for each individual module.
        load_state_path: The path to the module state to load from. NOTE: This must be
            an absolute path. NOTE: If the load_state_path of this spec is set, and
            the load_state_path of one of the SingleAgentRLModuleSpecs' is also set,
            the weights of that RL Module will be loaded from the path specified in
            the SingleAgentRLModuleSpec. This is useful if you want to load the weights
            of a MARL module and also manually load the weights of some of the RL
            modules within that MARL module from other checkpoints.
        modules_to_load: A set of module ids to load from the checkpoint. This is
            only used if load_state_path is set. If this is None, all modules are
            loaded.
    """
    marl_module_class: Type[MultiAgentRLModule] = MultiAgentRLModule
    module_specs: Union[SingleAgentRLModuleSpec, Dict[ModuleID, SingleAgentRLModuleSpec]] = None
    load_state_path: Optional[str] = None
    modules_to_load: Optional[Set[ModuleID]] = None

    def __post_init__(self):
        if False:
            print('Hello World!')
        if self.module_specs is None:
            raise ValueError('Module_specs cannot be None. It should be either a SingleAgentRLModuleSpec or a dictionary mapping from module IDs to SingleAgentRLModuleSpecs for each individual module.')

    def get_marl_config(self) -> 'MultiAgentRLModuleConfig':
        if False:
            i = 10
            return i + 15
        'Returns the MultiAgentRLModuleConfig for this spec.'
        return MultiAgentRLModuleConfig(modules=self.module_specs)

    @OverrideToImplementCustomLogic
    def build(self, module_id: Optional[ModuleID]=None) -> Union[SingleAgentRLModuleSpec, 'MultiAgentRLModule']:
        if False:
            print('Hello World!')
        'Builds either the multi-agent module or the single-agent module.\n\n        If module_id is None, it builds the multi-agent module. Otherwise, it builds\n        the single-agent module with the given module_id.\n\n        Note: If when build is called the module_specs is not a dictionary, it will\n        raise an error, since it should have been updated by the caller to inform us\n        about the module_ids.\n\n        Args:\n            module_id: The module_id of the single-agent module to build. If None, it\n                builds the multi-agent module.\n\n        Returns:\n            The built module. If module_id is None, it returns the multi-agent module.\n        '
        self._check_before_build()
        if module_id:
            return self.module_specs[module_id].build()
        module_config = self.get_marl_config()
        module = self.marl_module_class(module_config)
        return module

    def add_modules(self, module_specs: Dict[ModuleID, SingleAgentRLModuleSpec], overwrite: bool=True) -> None:
        if False:
            while True:
                i = 10
        'Add new module specs to the spec or updates existing ones.\n\n        Args:\n            module_specs: The mapping for the module_id to the single-agent module\n                specs to be added to this multi-agent module spec.\n            overwrite: Whether to overwrite the existing module specs if they already\n                exist. If False, they will be updated only.\n        '
        if self.module_specs is None:
            self.module_specs = {}
        for (module_id, module_spec) in module_specs.items():
            if overwrite or module_id not in self.module_specs:
                self.module_specs[module_id] = module_spec
            else:
                self.module_specs[module_id].update(module_spec)

    @classmethod
    def from_module(self, module: MultiAgentRLModule) -> 'MultiAgentRLModuleSpec':
        if False:
            return 10
        'Creates a MultiAgentRLModuleSpec from a MultiAgentRLModule.\n\n        Args:\n            module: The MultiAgentRLModule to create the spec from.\n\n        Returns:\n            The MultiAgentRLModuleSpec.\n        '
        module_specs = {module_id: SingleAgentRLModuleSpec.from_module(rl_module.unwrapped()) for (module_id, rl_module) in module._rl_modules.items()}
        marl_module_class = module.__class__
        return MultiAgentRLModuleSpec(marl_module_class=marl_module_class, module_specs=module_specs)

    def _check_before_build(self):
        if False:
            return 10
        if not isinstance(self.module_specs, dict):
            raise ValueError(f'When build() is called on {self.__class__}, the module_specs should be a dictionary mapping from module IDs to SingleAgentRLModuleSpecs for each individual module.')

    def to_dict(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Converts the MultiAgentRLModuleSpec to a dictionary.'
        return {'marl_module_class': serialize_type(self.marl_module_class), 'module_specs': {module_id: module_spec.to_dict() for (module_id, module_spec) in self.module_specs.items()}}

    @classmethod
    def from_dict(cls, d) -> 'MultiAgentRLModuleSpec':
        if False:
            i = 10
            return i + 15
        'Creates a MultiAgentRLModuleSpec from a dictionary.'
        return MultiAgentRLModuleSpec(marl_module_class=deserialize_type(d['marl_module_class']), module_specs={module_id: SingleAgentRLModuleSpec.from_dict(module_spec) for (module_id, module_spec) in d['module_specs'].items()})

    def update(self, other: 'MultiAgentRLModuleSpec', overwrite=False) -> None:
        if False:
            i = 10
            return i + 15
        "Updates this spec with the other spec.\n\n        Traverses this MultiAgentRLModuleSpec's module_specs and updates them with\n        the module specs from the other MultiAgentRLModuleSpec.\n\n        Args:\n            other: The other spec to update this spec with.\n            overwrite: Whether to overwrite the existing module specs if they already\n                exist. If False, they will be updated only.\n        "
        assert type(other) is MultiAgentRLModuleSpec
        if isinstance(other.module_specs, dict):
            self.add_modules(other.module_specs, overwrite=overwrite)
        elif not self.module_specs:
            self.module_specs = other.module_specs
        else:
            self.module_specs.update(other.module_specs)

@ExperimentalAPI
@dataclass
class MultiAgentRLModuleConfig:
    modules: Mapping[ModuleID, SingleAgentRLModuleSpec] = field(default_factory=dict)

    def to_dict(self):
        if False:
            print('Hello World!')
        return {'modules': {module_id: module_spec.to_dict() for (module_id, module_spec) in self.modules.items()}}

    @classmethod
    def from_dict(cls, d) -> 'MultiAgentRLModuleConfig':
        if False:
            while True:
                i = 10
        return cls(modules={module_id: SingleAgentRLModuleSpec.from_dict(module_spec) for (module_id, module_spec) in d['modules'].items()})
from typing import Any, Dict, Set, Tuple, Callable, List
import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat
from abc import ABC, abstractmethod
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import QConfig, default_qconfig, _assert_valid_qconfig
from torch.ao.quantization.observer import ObserverBase, default_dynamic_quant_observer, default_per_channel_weight_observer, default_observer, default_weight_observer
from torch.ao.quantization.fx._equalize import default_equalization_qconfig, EqualizationQConfig
from torch.ao.quantization.observer import _is_activation_post_process
DETECTOR_TARGET_NODE_KEY = 'target_node'
DETECTOR_OBS_TO_INSERT_KEY = 'observer_to_insert'
DETECTOR_IS_POST_OBS_KEY = 'is_post_observer'
DETECTOR_OBS_ARGS_KEY = 'observer_args'

class DetectorQConfigInfo:
    """
    This class contains the QConfig information for a single module.
    The list of variables / values this contains can grow depending on the
    extensibility of the qconfig mapping feature set but this currently includes:
    - if activation observer is dynamic
    - if weight observer is per channel


    Args:
        module_fqn (str): The fully qualified name (fqn) of the module that this
            information contains info relevant to qconfig for
    """

    def __init__(self, module_fqn: str):
        if False:
            print('Hello World!')
        super().__init__()
        self.module_fqn = module_fqn
        self.is_activation_dynamic = False
        self.is_weight_per_channel = False
        self.is_equalization_recommended = False

    def generate_quantization_qconfig(self, module: torch.nn.Module) -> QConfig:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            module (torch.nn.Module) The module we are generating\n            the qconfig for\n\n        Returns the generated quantization QConfig according to what a valid configuration is\n        '
        module_qconfig = default_qconfig
        recommendations_list = []
        recommendations_list.append((self.is_activation_dynamic, self.is_weight_per_channel))
        recommendations_list.append((self.is_activation_dynamic, False))
        recommendations_list.append((False, self.is_weight_per_channel))
        for rec in recommendations_list:
            activation = default_dynamic_quant_observer if rec[0] else default_observer
            weight = default_per_channel_weight_observer if rec[1] else default_weight_observer
            test_config = QConfig(activation, weight)
            try:
                _assert_valid_qconfig(test_config, module)
                module_qconfig = test_config
                break
            except AssertionError:
                continue
        return module_qconfig

    def generate_equalization_qconfig(self) -> EqualizationQConfig:
        if False:
            while True:
                i = 10
        '\n        This returns the equalization configuration for a module.\n\n        For now, it just returns the default, but as more equalization options become\n        possible, this method can get more fleshed out with more nuanced granularity.\n\n\n        Returns the generated equalization QConfig according to what a valid configuration is\n        '
        return default_equalization_qconfig

class DetectorBase(ABC):
    """ Base Detector Module
    Any detector class should derive from this class.

    Concrete detectors should follow the same general API, which includes:
    - A method to calculate and return observer insertion points
        - Should return both the fqns and the Observer class to insert
    - A method to return a report based on the detector
        - Should return a str-based report and dict info in Tuple[str,Dict] format
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.detector_config_info = None

    @abstractmethod
    def determine_observer_insert_points(self, model) -> Dict:
        if False:
            i = 10
            return i + 15
        '\n        Args\n            model (nn.Module or subclass): model to find observer insertion points\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict.\n            This dict maps string keys to detector specific information\n        '
        pass

    @abstractmethod
    def get_detector_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ' Returns the name of the current detector '
        pass

    @abstractmethod
    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        if False:
            i = 10
            return i + 15
        ' Returns the DetectorQConfigInfo for each module_fqn relevant\n        Args\n            model (nn.Module or subclass): model to find observer insertion points\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:\n            A DetectorQConfigInfo with the information to generate a QConfig for a specific module\n        '
        pass

    def _get_targeting_node(self, prepared_fx_model: GraphModule, target_fqn: str) -> torch.fx.node.Node:
        if False:
            print('Hello World!')
        "\n        Takes in a GraphModule and the target_fqn and finds the node whose target is this fqn.\n\n        If it's not found, it means it is most likely inside a fused layer\n            We just go one layer up in terms of the fqn we are searching for until we find parent node\n            If we get to empty string, then we know that it doesn't exist\n\n        The reason for the recursion is that if the model that we are looking for got fused,\n        we will have module fqn as e.g. x.linear.0 but the graph will only have a node for the fused module,\n        which would have fqn as x.linear so they will not match.\n        To handle this, if we don't match, we then take off the last bit of the fqn e.g. x.linear.0 -> x.linear,\n        or more generally foo.bar.baz -> foo.bar and search again, this will allow us to locate the correct module\n        even in cases with fusion\n\n        Args:\n            prepared_fx_model (GraphModule):  The prepared Fx GraphModule\n            target_fqn (str): The fqn of the layer we are trying to target\n\n        Returns the node object we are trying to add observers around\n        "
        for node in prepared_fx_model.graph.nodes:
            if node.target == target_fqn:
                return node
        parent_fqn_sep_index = target_fqn.rfind('.')
        if parent_fqn_sep_index == -1:
            raise ValueError("passed in target_fqn not found in graph's targets.")
        else:
            return self._get_targeting_node(prepared_fx_model, target_fqn[:parent_fqn_sep_index])

    @abstractmethod
    def generate_detector_report(self, model) -> Tuple[str, Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        '\n        Args\n            model (nn.Module or subclass): model to find observer insertion points\n\n        Returns a Tuple of two elements:\n            Str: string report of the suggested improvements\n            Dict: contains useful data collected by the observer pertinent to this report\n        '
        pass

class PerChannelDetector(DetectorBase):
    """ This class is used to detect if any Linear or Conv layers in a model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        per_channel quantization can lead to major benefits in the form of accuracy.
        Therefore, if the backend used by the user supports it, it is recommended to use

        Args:
            backend (str, optional): the backend the user wishes to use in production
                Default value is current torch.backends.quantized.engine
    """
    BACKEND_KEY = 'backend'
    PER_CHAN_SUPPORTED_KEY = 'per_channel_quantization_supported'
    PER_CHAN_USED_KEY = 'per_channel_quantization_used'
    DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES: Dict[str, Set[Any]] = {'fbgemm': {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}, 'qnnpack': {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}, 'onednn': {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}, 'x86': {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}}

    def __init__(self, backend: str=torch.backends.quantized.engine):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.backend_chosen = backend
        self.supported_modules = set()
        if self.backend_chosen in self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES:
            self.supported_modules = self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES[self.backend_chosen]
        else:
            raise ValueError(f'Not configured to work with {self.backend_chosen}. Try a different default backend')

    def get_detector_name(self) -> str:
        if False:
            print('Hello World!')
        ' returns the string name of this detector'
        return 'per_channel_detector'

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        if False:
            return 10
        ' Returns the DetectorQConfigInfo for each module_fqn relevant\n        Args\n            model (nn.Module or subclass): model to find observer insertion points\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:\n            A DetectorQConfigInfo with the information to generate a QConfig for a specific module\n        '
        per_channel_info = self._detect_per_channel_helper(model)
        module_fqn_to_detector_qconfig_info = {}
        for module_fqn in per_channel_info:
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)
            per_chan_supported: bool = per_channel_info[module_fqn][self.PER_CHAN_SUPPORTED_KEY]
            detector_qconfig_info.is_weight_per_channel = per_chan_supported
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info
        return module_fqn_to_detector_qconfig_info

    def determine_observer_insert_points(self, model: nn.Module) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        There is no observers inserted for the PerChannelDetector.\n\n        Returns an empty dictionary since no observers are added or needed\n        '
        return {}

    def _detect_per_channel_helper(self, model: nn.Module):
        if False:
            return 10
        '\n        determines if per_channel quantization is supported in modules and submodules.\n\n        Returns a dictionary in the higher level _detect_per_channel function.\n        Each entry maps the fully-qualified-name to information on whether per_channel quantization.\n\n        Args:\n            model: The current module that is being checked to see if it is per_channel quantizable\n\n        Returns dictionary mapping fqns to if per_channel quantization is possible\n        '
        per_channel_info: Dict = {}
        for (fqn, module) in model.named_modules():
            is_in_include_list = sum([isinstance(module, x) for x in self.supported_modules]) > 0
            per_channel_supported = False
            if is_in_include_list:
                per_channel_supported = True
                q_config_file = module.qconfig
                assert isinstance(q_config_file, QConfig)
                q_or_s_obj = module.qconfig.weight.p.func()
                assert isinstance(q_or_s_obj, (FakeQuantize, ObserverBase))
                per_channel_used = False
                if hasattr(q_or_s_obj, 'ch_axis'):
                    if isinstance(q_or_s_obj, FakeQuantize):
                        if hasattr(q_or_s_obj, 'is_per_channel') and q_or_s_obj.is_per_channel:
                            per_channel_used = True
                    elif isinstance(q_or_s_obj, ObserverBase):
                        per_channel_used = True
                    else:
                        raise ValueError('Should be either observer or fake quant')
                per_channel_info[fqn] = {self.PER_CHAN_SUPPORTED_KEY: per_channel_supported, self.PER_CHAN_USED_KEY: per_channel_used, self.BACKEND_KEY: self.backend_chosen}
        return per_channel_info

    def generate_detector_report(self, model: nn.Module) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10
        'Checks if any Linear or Conv layers in the model utilize per_channel quantization.\n        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.\n\n        Looks at q_config format and backend to determine if per_channel can be utilized.\n        Uses the DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES structure to determine support\n\n        Args:\n            model: The prepared and calibrated model we want to check if using per_channel\n\n        Returns a tuple with two elements:\n            String report of potential actions to improve model (if per_channel quantization is available in backend)\n            Dictionary mapping per_channel quantizable elements to:\n                whether per_channel quantization is supported by the backend\n                if it is being utilized in the current model\n        '
        per_channel_info = self._detect_per_channel_helper(model)
        further_optims_str = f'Further Optimizations for backend {self.backend_chosen}: \n'
        optimizations_possible = False
        for fqn in per_channel_info:
            fqn_dict = per_channel_info[fqn]
            if fqn_dict[self.PER_CHAN_SUPPORTED_KEY] and (not fqn_dict[self.PER_CHAN_USED_KEY]):
                optimizations_possible = True
                further_optims_str += f'Module {fqn} can be configured to use per_channel quantization.\n'
        if optimizations_possible:
            further_optims_str += 'To use per_channel quantization, make sure the qconfig has a per_channel weight observer.'
        else:
            further_optims_str += 'No further per_channel optimizations possible.'
        return (further_optims_str, per_channel_info)

class DynamicStaticDetector(DetectorBase):
    """
    Determines whether dynamic or static quantization is more appropriate for a given module.

    Takes advantage of the ModelReportObserver that records range information.
    Stationary distribution of data are strictly above tolerance level for the comparison statistic:

        S = average_batch_activation_range/epoch_activation_range

    Nonstationary distributions are below or at the tolerance level for this metric.

    If the distribution of data right after the module is non-stationary, recommend dynamic quantization
        Otherwise recommend static quantization

    Args:
        tolerance (float, optional): The threshold where S metric is stationary above and non-stationary otherwise. Default: 0.5
    """
    DEFAULT_PRE_OBSERVER_NAME = 'model_report_pre_observer'
    DEFAULT_POST_OBSERVER_NAME = 'model_report_post_observer'
    STATIONARY_STR = 'stationary'
    NON_STATIONARY_STR = 'non-stationary'
    INPUT_ACTIVATION_PREFIX = 'input_activation_'
    OUTPUT_ACTIVATION_PREFIX = 'output_activation_'
    TOLERANCE_KEY = 'dynamic_static_tolerance'
    DEFAULT_DYNAMIC_REC_KEY = 'dynamic_recommended'
    PRE_OBS_COMP_STAT_KEY = INPUT_ACTIVATION_PREFIX + 'dynamic_static_comp_stat'
    POST_OBS_COMP_STAT_KEY = OUTPUT_ACTIVATION_PREFIX + 'dynamic_static_comp_stat'
    PRE_OBS_DATA_DIST_KEY = INPUT_ACTIVATION_PREFIX + 'dynamic_static_data_classification'
    POST_OBS_DATA_DIST_KEY = OUTPUT_ACTIVATION_PREFIX + 'dynamic_static_data_classification'
    IS_CURRENTLY_SUPPORTED_KEY = 'is_dynamic_supported'
    DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED = {nn.Linear}
    DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED = {nn.Conv1d, nn.Conv2d, nn.Conv3d}

    def __init__(self, tolerance=0.5):
        if False:
            print('Hello World!')
        super().__init__()
        self.tolerance = tolerance
        self.useful_observer_fqns: Set[str] = set()

    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Determines where observers need to be inserted for the Dynamic vs Static detector.\n        For this detector, we want to place observers on either side of linear layers in the model.\n\n        Currently inserts observers for:\n            linear layers\n\n        Args:\n            prepared_fx_model (GraphModule):  The prepared Fx GraphModule\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:\n            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)\n            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)\n            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer\n            key "observer_args" -> The arguments that are meant to be passed into the observer\n        '
        obs_ctr = ModelReportObserver
        obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}
        for (fqn, module) in prepared_fx_model.named_modules():
            if self._is_supported(module, insert=True):
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)
                pre_obs_fqn = fqn + '.' + self.DEFAULT_PRE_OBSERVER_NAME
                obs_fqn_to_info[pre_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(), DETECTOR_IS_POST_OBS_KEY: False, DETECTOR_OBS_ARGS_KEY: targeted_node.args}
                post_obs_fqn = fqn + '.' + self.DEFAULT_POST_OBSERVER_NAME
                obs_fqn_to_info[post_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(), DETECTOR_IS_POST_OBS_KEY: True, DETECTOR_OBS_ARGS_KEY: (targeted_node,)}
        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ' returns the string name of this detector'
        return 'dynamic_vs_static_detector'

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        if False:
            while True:
                i = 10
        ' Returns the DetectorQConfigInfo for each module_fqn relevant\n        Args\n            model (nn.Module or subclass): model to find observer insertion points\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:\n            A DetectorQConfigInfo with the information to generate a QConfig for a specific module\n        '
        dynamic_static_info = self._generate_dict_info(model)
        module_fqn_to_detector_qconfig_info = {}
        for module_fqn in dynamic_static_info:
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)
            dynamic_static_recommended: bool = dynamic_static_info[module_fqn][self.DEFAULT_DYNAMIC_REC_KEY]
            detector_qconfig_info.is_activation_dynamic = dynamic_static_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info
        return module_fqn_to_detector_qconfig_info

    def _is_supported(self, module: nn.Module, insert: bool=False) -> bool:
        if False:
            i = 10
            return i + 15
        'Returns whether the given module is supported for observers\n\n        Args\n            module: The module to check and ensure is supported\n            insert: True if this is check for observer insertion, false if for report gen\n\n        Returns True if the module is supported by observer, False otherwise\n        '
        is_supported_type = sum([isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED]) > 0
        future_supported_type = sum([isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED]) > 0
        supported = is_supported_type or future_supported_type
        if insert:
            return supported
        else:
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME) and hasattr(module, self.DEFAULT_POST_OBSERVER_NAME)
            return supported and has_obs

    def _generate_dict_info(self, model: GraphModule) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper function for generate_detector_report that does the generation of the dictionary.\n        This process is done as specified in generate_detector_report documentation\n\n        Args:\n            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers\n\n        Returns a Dictionary mapping modules with ModelReportObservers around them to:\n                whether dynamic quantization is recommended\n                their S metric of input to module\n                whether input to module is stationary or non-stationary\n                their S metric of output of module\n                whether output of module is stationary or non-stationary\n                the tolerance level to decided whether input/output is stationary or non-stationary\n                whether it is currently supported or planned for the future\n        '
        module_dynamic_static_info = {}
        for (fqn, module) in model.named_modules():
            if self._is_supported(module):
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                post_obs = getattr(module, self.DEFAULT_POST_OBSERVER_NAME)
                pre_stat = pre_obs.get_batch_to_epoch_ratio()
                post_stat = post_obs.get_batch_to_epoch_ratio()
                dynamic_recommended = post_stat <= self.tolerance
                pre_obs_dist_classif = self.STATIONARY_STR if pre_stat > self.tolerance else self.NON_STATIONARY_STR
                post_obs_dist_classif = self.STATIONARY_STR if post_stat > self.tolerance else self.NON_STATIONARY_STR
                is_supported_type = sum([isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED]) > 0
                module_info = {self.TOLERANCE_KEY: self.tolerance, self.DEFAULT_DYNAMIC_REC_KEY: dynamic_recommended, self.PRE_OBS_COMP_STAT_KEY: pre_stat, self.PRE_OBS_DATA_DIST_KEY: pre_obs_dist_classif, self.POST_OBS_COMP_STAT_KEY: post_stat, self.POST_OBS_DATA_DIST_KEY: post_obs_dist_classif, self.IS_CURRENTLY_SUPPORTED_KEY: is_supported_type}
                module_dynamic_static_info[fqn] = module_info
        return module_dynamic_static_info

    def generate_detector_report(self, model: GraphModule) -> Tuple[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        '\n        Determines whether dynamic or static quantization is more appropriate for a given module.\n\n        Takes advantage of the ModelReportObserver that records range information.\n        Stationary distribution of data are strictly above tolerance level for the comparison statistic:\n\n            S = average_batch_activation_range/epoch_activation_range\n\n        Nonstationary distributions are below or at the tolerance level for this metric.\n\n        If the distribution of data right after the module is non-stationary, recommend dynamic quantization\n            Otherwise recommend static quantization\n\n        This will then generate suggestions for dynamic vs static quantization focused around Linear.\n\n        Args:\n            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers\n\n        Returns a tuple with two elements:\n            String report of of whether dynamic or static quantization is recommended for certain modules\n            Dictionary mapping modules with ModelReportObservers around them to:\n                whether dynamic quantization is recommended\n                their S metric of input to module\n                whether input to module is stationary or non-stationary\n                their S metric of output of module\n                whether output of module is stationary or non-stationary\n                the tolerance level to decided whether input/output is stationary or non-stationary\n                whether it is currently supported or planned for the future\n        '
        module_dynamic_static_info = self._generate_dict_info(model)
        dynamic_vs_static_string = 'Dynamic vs. Static Quantization suggestions: \n'
        modules_added: bool = False
        dynamic_benefit = ' You will get more accurate results if you use dynamic quantization'
        static_benefit = ' You can increase model efficiency if you use static quantization'
        future_support_str = '. This layer is not yet supported for dynamic quantization'
        for module_fqn in module_dynamic_static_info.keys():
            modules_added = True
            module_info = module_dynamic_static_info[module_fqn]
            suggestion_string_template = 'For module {} it is suggested to use {} quantization because {}.\n'
            quantization_type = ''
            quantization_reasoning = 'the distribution of data before {} is {} and the distribution after is {}.'
            benefit_str = ''
            recommend_per_tensor = '. We recommend to add a {} before this module if it is static.'
            rec_lay_to_add = 'dynamic quantize per tensor layer'
            dynamic_per_tensor_string = recommend_per_tensor.format(rec_lay_to_add)
            dynamic_per_tensor_reasoning_string = ' This is because the input to this module has a non-stationary distribution'
            if module_info[self.DEFAULT_DYNAMIC_REC_KEY]:
                quantization_type = 'dynamic'
                benefit_str = dynamic_benefit
                if not module_info[self.IS_CURRENTLY_SUPPORTED_KEY]:
                    benefit_str += future_support_str
            else:
                quantization_type = 'static'
                benefit_str = static_benefit
            quantization_reasoning = quantization_reasoning.format(module_fqn, module_info[self.PRE_OBS_DATA_DIST_KEY], module_info[self.POST_OBS_DATA_DIST_KEY]) + benefit_str
            if module_info[self.PRE_OBS_DATA_DIST_KEY] == self.NON_STATIONARY_STR and module_info[self.POST_OBS_DATA_DIST_KEY] == self.STATIONARY_STR:
                quantization_reasoning = quantization_reasoning + dynamic_per_tensor_string + dynamic_per_tensor_reasoning_string
            module_suggestion_string = suggestion_string_template.format(module_fqn, quantization_type, quantization_reasoning)
            dynamic_vs_static_string += module_suggestion_string
        if not modules_added:
            dynamic_vs_static_string += 'No applicable layers for suggestions. Only linear and conv are valid.\n'
        return (dynamic_vs_static_string, module_dynamic_static_info)

class InputWeightEqualizationDetector(DetectorBase):
    """
    Determines whether input-weight equalization can help improve quantization for certain modules.

    Specifically, this list of modules includes:
        linear
        conv

    Determines whether input-weight equalization is recommended based on the comp stat:
        s_c = sqrt(w_c/W)/sqrt(i_c/I)
        where:
            w_c is range of weight for channel c, W is range of weight over all channels
            i_c is range of input for channel c, I is range of input over all channels

        if s_c >= threshold or <= 1 / threshold, recommends input-weight equalization

    Args:
        ratio_threshold (float): The threshold for s_c to determine if input-weight equalization is suggested
            Should be between 0 and 1 (both non-inclusive)
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for s_c to determine if input-weight equalization is suggested
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine input weight equalization

    * :attr:`SUPPORTED_MODULES`: This specifies the modules that are supported for input-weight equalization

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """
    SUPPORTED_MODULES: Set[Callable] = {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d}
    DEFAULT_PRE_OBSERVER_NAME: str = 'model_report_pre_observer'
    WEIGHT_PREFIX = 'weight_'
    ACTIVATION_PREFIX = 'input_activation_'
    PER_CHANNEL_MAX_KEY = 'per_channel_max'
    PER_CHANNEL_MIN_KEY = 'per_channel_min'
    GLOBAL_MAX_KEY = 'global_max'
    GLOBAL_MIN_KEY = 'global_min'
    RECOMMENDED_KEY = 'input_weight_equalization_recommended'
    COMP_METRIC_KEY = 'input_weight_channel_comparison_metrics'
    THRESHOLD_KEY = 'input_weight_threshold'
    CHANNEL_KEY = 'input_weight_channel_axis'
    WEIGHT_STR = 'weight'
    INPUT_STR = 'input'
    DEFAULT_RECOMMEND_INPUT_WEIGHT_CHANNEL_RATIO = 0.4

    def __init__(self, ratio_threshold: float, ch_axis: int=1):
        if False:
            while True:
                i = 10
        if ratio_threshold <= 0 or ratio_threshold >= 1:
            raise ValueError('Make sure threshold is > 0 and < 1')
        self.ratio_threshold: float = ratio_threshold
        self.ch_axis: int = ch_axis

    def _is_supported(self, module: nn.Module, insert: bool=False) -> bool:
        if False:
            return 10
        'Returns whether the given module is supported for observers\n\n        Args\n            module: The module to check and ensure is supported\n            insert: True if this is check for observer insertion, false if for report gen\n\n        Returns True if the module is supported by observer, False otherwise\n        '
        is_supported_type = sum([type(module) is x for x in self.SUPPORTED_MODULES]) > 0
        if insert:
            return is_supported_type
        else:
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
            return is_supported_type and has_obs

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        if False:
            for i in range(10):
                print('nop')
        ' Returns the DetectorQConfigInfo for each module_fqn relevant\n        Args\n            model (nn.Module or subclass): model to find observer insertion points\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:\n            A DetectorQConfigInfo with the information to generate a QConfig for a specific module\n        '
        input_values: Dict[str, Dict] = self._extract_input_info(model)
        weight_values: Dict[str, Dict] = self._extract_weight_info(model)
        comp_stats: Dict[str, torch.Tensor] = self._generate_comparison_values(input_values, weight_values)
        input_weight_equalization_info: Dict[str, Dict] = self._generate_dict_info(input_values, weight_values, comp_stats)
        module_fqn_to_detector_qconfig_info = {}
        for module_fqn in input_weight_equalization_info:
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)
            input_weight_recommended: bool = input_weight_equalization_info[module_fqn][self.RECOMMENDED_KEY]
            detector_qconfig_info.is_equalization_recommended = input_weight_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info
        return module_fqn_to_detector_qconfig_info

    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        if False:
            return 10
        'Determines where observers need to be inserted for the Input Weight Equalization Detector.\n        For this detector, we want to place observers in front of supported layers.\n\n        Currently inserts observers for:\n            linear layers\n            conv layers\n\n        Args:\n            prepared_fx_model (GraphModule):  The prepared Fx GraphModule\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:\n            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)\n            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)\n            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer\n            key "observer_args" -> The arguments that are meant to be passed into the observer\n        '
        obs_ctr = ModelReportObserver
        obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}
        for (fqn, module) in prepared_fx_model.named_modules():
            if self._is_supported(module, insert=True):
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)
                pre_obs_fqn = fqn + '.' + self.DEFAULT_PRE_OBSERVER_NAME
                obs_fqn_to_info[pre_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis), DETECTOR_IS_POST_OBS_KEY: False, DETECTOR_OBS_ARGS_KEY: targeted_node.args}
        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the name of this detector'
        return 'input_weight_equalization_detector'

    def _extract_input_info(self, model: GraphModule) -> Dict[str, Dict]:
        if False:
            i = 10
            return i + 15
        '\n        Takes in a calibrated GraphModule and then finds the relevant observers.\n        It then extracts the input information for each observer returns it\n\n        Args\n            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers\n\n        Returns a dict mapping relevant module fqns (str) to a dict with keys:\n            "input_activation_per_channel_max" : maps to the per_channel max values\n            "input_activation_per_channel_min" : maps to the per_channel min values\n            "input_activation_global_max" : maps to the global max recorded\n            "input_activation_global_min" : maps to the global min recorded\n        '
        input_info: Dict[str, Dict] = {}
        for (fqn, module) in model.named_modules():
            if self._is_supported(module):
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                input_info[fqn] = {self.ACTIVATION_PREFIX + self.PER_CHANNEL_MAX_KEY: pre_obs.max_val, self.ACTIVATION_PREFIX + self.PER_CHANNEL_MIN_KEY: pre_obs.min_val, self.ACTIVATION_PREFIX + self.GLOBAL_MAX_KEY: max(pre_obs.max_val), self.ACTIVATION_PREFIX + self.GLOBAL_MIN_KEY: min(pre_obs.min_val)}
        return input_info

    def _extract_weight_info(self, model: GraphModule) -> Dict[str, Dict]:
        if False:
            print('Hello World!')
        '\n        Takes in a calibrated GraphModule and then finds the relevant observers.\n        It then extracts the weight information for each layer an observer is attached to.\n\n        Args\n            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers\n\n        Returns a dict mapping module fqns (str) to a dict with keys:\n            "per_channel_max" : maps to the per_channel max values\n            "per_channel_min" : maps to the per_channel min values\n            "global_max" : maps to the global max recorded\n            "global_min" : maps to the global min recorded\n        '
        weight_info: Dict[str, Dict] = {}
        for (fqn, module) in model.named_modules():
            if self._is_supported(module):
                device = module.weight.device
                min_val: torch.Tensor = torch.tensor([float('inf')], device=device)
                max_val: torch.Tensor = torch.tensor([float('-inf')], device=device)
                x_copy = module.weight
                x_dim = x_copy.size()
                new_axis_list = [i for i in range(len(x_dim))]
                new_axis_list[self.ch_axis] = 0
                new_axis_list[0] = self.ch_axis
                y = x_copy.permute(new_axis_list)
                y = y.to(min_val.dtype)
                y = torch.flatten(y, start_dim=1)
                if min_val.numel() == 0 or max_val.numel() == 0:
                    (min_val, max_val) = torch.aminmax(y, dim=1)
                else:
                    (min_val_cur, max_val_cur) = torch.aminmax(y, dim=1)
                    min_val = torch.min(min_val_cur, min_val)
                    max_val = torch.max(max_val_cur, max_val)
                weight_info[fqn] = {self.WEIGHT_PREFIX + self.PER_CHANNEL_MAX_KEY: max_val, self.WEIGHT_PREFIX + self.PER_CHANNEL_MIN_KEY: min_val, self.WEIGHT_PREFIX + self.GLOBAL_MAX_KEY: max(max_val), self.WEIGHT_PREFIX + self.GLOBAL_MIN_KEY: min(min_val)}
        return weight_info

    def _calculate_range_ratio(self, info_dict: Dict, info_str: str, module_fqn: str) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Takes in an info dict and calculates the s_c matrix.\n\n        Args:\n            info_dict (dict): A dictionary of either input or weight range info\n            info_str (str): A str describing whether currently looking at weight or input info\n                Either "weight" or "input"\n            module_fqn (str): The fqn of the module we are looking at\n\n        Returns a tensor of values, where each value is the s_c stat for a different channel\n        '
        prefix_str = self.ACTIVATION_PREFIX if info_str == self.INPUT_STR else self.WEIGHT_PREFIX
        per_channel_range = info_dict[prefix_str + self.PER_CHANNEL_MAX_KEY] - info_dict[prefix_str + self.PER_CHANNEL_MIN_KEY]
        global_range = info_dict[prefix_str + self.GLOBAL_MAX_KEY] - info_dict[prefix_str + self.GLOBAL_MIN_KEY]
        if global_range == 0:
            range_zero_explanation = "We recommend removing this channel as it doesn't provide any useful information."
            raise ValueError('The range of the {} data for module {} is 0, which means you have a constant value channel. {}'.format(info_str, module_fqn, range_zero_explanation))
        ratio = per_channel_range / global_range
        return ratio

    def _generate_comparison_values(self, input_info: Dict, weight_info: Dict) -> Dict[str, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Takes in the information on the min and max values of the inputs and weights and:\n            Calculates the comp stat for each channel: s_c = sqrt(w_c/W)/sqrt(i_c/I)\n\n        Args:\n            input_info (dict): A dict mapping each observer to input range information\n            weight_info (dict): A dict mapping each observer to weight range information\n\n        Returns a dict mapping relevant observer fqns (str) to a 1-D tensor.\n            Each value is a different s_c value for a different channel\n        '
        module_fqn_to_channel: Dict[str, torch.Tensor] = {}
        for module_fqn in input_info:
            if module_fqn not in weight_info:
                raise KeyError(f'Unable to find weight range stats for module {module_fqn}')
            weight_ratio = self._calculate_range_ratio(weight_info[module_fqn], self.WEIGHT_STR, module_fqn)
            input_ratio = self._calculate_range_ratio(input_info[module_fqn], self.INPUT_STR, module_fqn)
            weight_channels = len(weight_ratio)
            input_channels = len(input_ratio)
            if weight_channels != input_channels:
                assert input_channels % weight_channels == 0, 'input channels should be divisible by weight channels.'
                rep_factor: int = input_channels // weight_channels
                weight_ratio = weight_ratio.repeat(rep_factor)
            s = torch.sqrt(weight_ratio) / torch.sqrt(input_ratio)
            module_fqn_to_channel[module_fqn] = s
        return module_fqn_to_channel

    def _generate_dict_info(self, input_info: Dict, weight_info: Dict, comp_stats: Dict) -> Dict[str, Dict]:
        if False:
            return 10
        '\n        Helper function for generate_detector_report that does the generation of the dictionary.\n        This process is done as specified in generate_detector_report documentation\n\n        Args:\n            input_info (dict): A dict mapping each module to input range information\n            weight_info (dict): A dict mapping each module to weight range information\n            comp_stats (dict): A dict mapping each module to its corresponding comp stat\n\n        Returns a dictionary mapping each module with relevant ModelReportObservers around them to:\n            whether input weight equalization is recommended\n            their s_c metric compared to the threshold\n            the threshold used to make the recommendation\n            the channel used for recording data\n            the input channel range info\n            the weight channel range info\n        '
        input_weight_equalization_info: Dict[str, Dict] = {}
        for module_fqn in input_info:
            mod_input_info: Dict = input_info[module_fqn]
            mod_weight_info: Dict = weight_info[module_fqn]
            mod_comp_stat: Dict = comp_stats[module_fqn]
            channel_rec_vals: list = []
            for val in mod_comp_stat:
                float_rep: float = val.item()
                recommended: bool = float_rep >= self.ratio_threshold and float_rep <= 1 / self.ratio_threshold
                channel_rec_vals.append(recommended)
            input_weight_equalization_info[module_fqn] = {self.RECOMMENDED_KEY: channel_rec_vals, self.COMP_METRIC_KEY: mod_comp_stat, self.THRESHOLD_KEY: self.ratio_threshold, self.CHANNEL_KEY: self.ch_axis, **mod_input_info, **mod_weight_info}
        return input_weight_equalization_info

    def generate_detector_report(self, model: GraphModule) -> Tuple[str, Dict[str, Any]]:
        if False:
            return 10
        '\n        Determines whether input weight equalization is appropriate for a given module.\n\n        Takes advantage of the ModelReport Observer which records per channel information of input range\n        It then uses the passed in weight info inconjunction to compute the desired ratio\n        Finally, it gives suggestions based on this information for each module of interest\n\n        Args:\n            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers\n\n        Returns a tuple with two elements:\n            String report of of whether input weight equalization is recommended for certain modules\n            Dictionary mapping modules of interest to:\n                whether input weight equalization is recommended\n                their s_c metric compared to the threshold\n                the threshold used to make the recommendation\n                the channel used for recording data\n                the input channel range info\n                the weight channel range info\n        '
        input_values: Dict[str, Dict] = self._extract_input_info(model)
        weight_values: Dict[str, Dict] = self._extract_weight_info(model)
        comp_stats: Dict[str, torch.Tensor] = self._generate_comparison_values(input_values, weight_values)
        input_weight_equalization_info: Dict[str, Dict] = self._generate_dict_info(input_values, weight_values, comp_stats)
        input_weight_string = 'Input-Weight Equalization suggestions: \n'
        module_suggestion_str = 'For Module {} looked at with axis {}: \n'
        channel_suggestion_str = '\tWe suggest {} input weight equalization because {}\n'
        use_str = 'to use'
        no_use_str = 'to not use'
        input_weight_benefit_str = '{}/{} channels would benefit and we expect significant reduction in quantization error.'
        input_weight_non_benefit_reasoning = '{}/{} channels benefitting from input-weight equalization being applied.'
        input_weight_non_benefit_str = "we don't expect much improvement from input-weight equalization based on {}"
        added_module: bool = False
        for module_fqn in input_weight_equalization_info:
            added_module = True
            input_weight_string += module_suggestion_str.format(module_fqn, self.ch_axis)
            mod_info: Dict[str, Any] = input_weight_equalization_info[module_fqn]
            recommendation_per_channel: torch.Tensor = mod_info[self.RECOMMENDED_KEY]
            num_recs = sum(recommendation_per_channel)
            if num_recs / len(recommendation_per_channel) >= self.DEFAULT_RECOMMEND_INPUT_WEIGHT_CHANNEL_RATIO:
                input_benefit_formatted = input_weight_benefit_str.format(num_recs, len(recommendation_per_channel))
                channel_str = channel_suggestion_str.format(use_str, input_benefit_formatted)
                input_weight_string += channel_str
            else:
                non_benefit_reason_formatted = input_weight_non_benefit_reasoning.format(num_recs, len(recommendation_per_channel))
                non_benefit_str = input_weight_non_benefit_str.format(non_benefit_reason_formatted)
                channel_str = channel_suggestion_str.format(no_use_str, non_benefit_str)
                input_weight_string += channel_str
        if not added_module:
            input_weight_string += 'No applicable layers for suggestions. Only linear and conv valid.\n'
        return (input_weight_string, input_weight_equalization_info)

class OutlierDetector(DetectorBase):
    """
    Determines whether there are significant outliers in activation data around a certain layer.

    This is ideally used in conjunction with information on stationary vs. non-stationary distribution:
        If the data is stationary, and there are significant outliers, then we want to flag them
        We want to do this on a per channel basis for detecting outliers

    Determines whether activation data is flagged as outlier based on if data is stationary and:
        p_r = avg(100th percentile / "reference_percentile"th percentile)
        where:
            p_r is average percentile ratio across all batches in the epoch
            reference_percentile is a percentile values between 0 and 100 exclusive

        if p_r is above some threshold, then we consider the activations to have significant outliers

    Args:
        ratio_threshold (float, optional): The threshold for p_r to determine if there are outliers in activations
            Should be >= 1
            Default: 3.5
        reference_percentile (float, optional): The denominator to find the relative scale of the 100th percentile
            Should be between 0 and 1
            Default: 0.975
        fraction_batches_used_threshold (float, optional): Threshold of fraction of batches per channel to determine outlier
            If fraction is below this, we deem number of samples used to calculate outliers as insignificant and alert user
            regardless of whether we detected outliers or not in channel to take a closer look at channel results
            Should be between 0 and 1
            Default: 0.95
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for p_r to determine if there are outliers in activations
        The p_r value (average ratio of 100th percentile/reference_percentile) is compared to ratio_threshold
        If it is significantly greater, then we consider it an outlier
        This threshold was calculated based on the ratio of the percentiles in a normal distribution
        The calculations behind value choice: https://drive.google.com/file/d/1N2wdtXWI-kOH8S7HH4-PYB_NmqzZil4p/view?usp=sharing

    * :attr:`reference_percentile`: The denominator of the top fraction to find the relative scale of the 100th percentile
        Should be between 0 and 1
        The calculations behind value choice: https://drive.google.com/file/d/1N2wdtXWI-kOH8S7HH4-PYB_NmqzZil4p/view?usp=sharing

    * :attr:`fraction_batches_used_threshold`: The fraction of batches to determine outliers for each channel should be above this
        Some batches may not be used because of 0-based errors, so this is to ensure a good amount of the total batches are used
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine outliers

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """
    DEFAULT_PRE_OBSERVER_NAME: str = 'model_report_pre_observer'
    INPUT_ACTIVATION_PREFIX = 'input_activation_'
    OUTLIER_KEY = 'outliers_detected'
    NUM_BATCHES_KEY = 'outlier_detection_batches_used'
    IS_SUFFICIENT_BATCHES_KEY = 'outlier_detection_is_sufficient_batches'
    COMP_METRIC_KEY = 'outlier_detection_percentile_ratios'
    RATIO_THRES_KEY = 'outlier_detection_ratio_threshold'
    REF_PERCENTILE_KEY = 'outlier_detection_reference_percentile'
    CHANNEL_AXIS_KEY = 'outlier_detection_channel_axis'
    MAX_VALS_KEY = INPUT_ACTIVATION_PREFIX + 'per_channel_max'
    CONSTANT_COUNTS_KEY = 'constant_batch_counts'

    def __init__(self, ratio_threshold: float=3.5, reference_percentile: float=0.975, fraction_batches_used_threshold: float=0.95, ch_axis: int=1):
        if False:
            for i in range(10):
                print('nop')
        self.ratio_threshold = ratio_threshold
        assert reference_percentile >= 0 and reference_percentile <= 1
        assert fraction_batches_used_threshold >= 0 and fraction_batches_used_threshold <= 1
        self.reference_percentile = reference_percentile
        self.fraction_batches_used_threshold = fraction_batches_used_threshold
        self.ch_axis = ch_axis

    def get_detector_name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the name of this detector'
        return 'outlier_detector'

    def _supports_insertion(self, module: nn.Module) -> bool:
        if False:
            while True:
                i = 10
        "Returns whether the given module is supported for observers insertion\n\n        Any module that doesn't have children and isn't an observer itself is supported\n\n        Args\n            module: The module to check and ensure is supported\n\n        Returns True if the module is supported by observer, False otherwise\n        "
        num_children = len(list(module.children()))
        return num_children == 0 and (not _is_activation_post_process(module))

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        if False:
            while True:
                i = 10
        ' Returns the DetectorQConfigInfo for each module_fqn relevant\n        Args\n            model (nn.Module or subclass): model to find observer insertion points\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:\n            A DetectorQConfigInfo with the information to generate a QConfig for a specific module\n        '
        return {}

    def _supports_report_gen(self, module: nn.Module) -> bool:
        if False:
            return 10
        'Returns whether the given module is supported for report generation\n\n        Any module that has a model report pre-observer is supported\n\n        Args\n            module: The module to check and ensure is supported\n\n        Returns True if the module is supported by observer, False otherwise\n        '
        return hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)

    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        if False:
            return 10
        ' Determines where observers need to be inserted for the Outlier Detector.\n\n        For this detector, we want to place observers in front of supported layers.\n\n        Currently inserts observers for:\n            all layers that do not have children (leaf level layers)\n\n        Args:\n            prepared_fx_model (GraphModule):  The prepared Fx GraphModule\n\n        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:\n            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)\n            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)\n            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer\n            key "observer_args" -> The arguments that are meant to be passed into the observer\n        '
        obs_ctr = ModelReportObserver
        obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}
        for (fqn, module) in prepared_fx_model.named_modules():
            if self._supports_insertion(module):
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)
                pre_obs_fqn = fqn + '.' + self.DEFAULT_PRE_OBSERVER_NAME
                obs_fqn_to_info[pre_obs_fqn] = {DETECTOR_TARGET_NODE_KEY: targeted_node, DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis, comp_percentile=self.reference_percentile), DETECTOR_IS_POST_OBS_KEY: False, DETECTOR_OBS_ARGS_KEY: targeted_node.args}
        return obs_fqn_to_info

    def _calculate_outlier_info(self, percentile_ratios: torch.Tensor, counted_batches: torch.Tensor, total_batches: int) -> Dict[str, List[bool]]:
        if False:
            i = 10
            return i + 15
        '\n        Gives info on whether the percentile ratios calculated would be considered outliers\n        Also gives information on whether the collected data is statistically significant to make this claim\n\n        Args:\n            percentile_ratios (torch.Tensor): The average percentile_ratios per channel calculated by the observer\n            counted_batches (torch.Tensor): The number of batches used for average calculation per tensor\n            total_batches (int): The total number of batches that passed through observer in this epoch\n\n        Returns a dictionary mapping:\n            "outliers_detected" : list of bools per channel that are true if it is considered an outlier\n            "is_sufficient_batches": if o_r was >= fraction_batches_used_threshold:\n                where o_r = counted_batches / total_batches\n        '
        outlier_dict: Dict[str, List[bool]] = {self.OUTLIER_KEY: [], self.IS_SUFFICIENT_BATCHES_KEY: []}
        ratios_list: List = percentile_ratios.tolist()
        num_batches_list: List = counted_batches.tolist()
        significant_size = [batch_size / total_batches >= self.fraction_batches_used_threshold for batch_size in num_batches_list]
        outlier_dict[self.IS_SUFFICIENT_BATCHES_KEY] = significant_size
        outlier_detected = [ratio > self.ratio_threshold for ratio in ratios_list]
        outlier_dict[self.OUTLIER_KEY] = outlier_detected
        return outlier_dict

    def _generate_info_dict(self, model: GraphModule) -> Dict[str, Dict]:
        if False:
            while True:
                i = 10
        '\n        Helper function for generate_detector_report that does the generation of the dictionary.\n        This process is done as specified in generate_detector_report documentation\n\n        Args:\n            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers\n\n        Returns a dict mapping relevant module fqns to:\n            whether there were outliers found in activation before\n            the number of batches used for each channel\n            whether fraction of applicable batches used is above fraction_batches_used_threshold\n            their p_r metric compared to the threshold\n            the threshold used to make the recommendation\n            the reference_percentile used to make the recommendation\n            the channel axis used to determine individual channels\n            the constant batch counts per channel\n            the per channel max values\n        '
        info_dict: Dict[str, Dict] = {}
        for (fqn, module) in model.named_modules():
            if self._supports_report_gen(module):
                pre_obs: ModelReportObserver = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
                num_batches: torch.Tensor = pre_obs.percentile_batches_tracked
                average_ratios: torch.Tensor = pre_obs.average_percentile_ratio
                channel_batch_cnts: torch.Tensor = pre_obs.constant_channels
                total_batches: int = pre_obs.num_batches_tracked
                max_vals: torch.Tensor = pre_obs.max_val
                for (index, ratio_val) in enumerate(average_ratios):
                    if ratio_val.item() < 0:
                        average_ratios[index] = -ratio_val
                    if ratio_val.item() < 1:
                        average_ratios[index] = 1 / ratio_val
                outlier_calcs = self._calculate_outlier_info(average_ratios, num_batches, total_batches)
                info_dict[fqn] = {self.CHANNEL_AXIS_KEY: self.ch_axis, self.REF_PERCENTILE_KEY: self.reference_percentile, self.RATIO_THRES_KEY: self.ratio_threshold, self.COMP_METRIC_KEY: average_ratios, self.NUM_BATCHES_KEY: num_batches, self.OUTLIER_KEY: outlier_calcs[self.OUTLIER_KEY], self.IS_SUFFICIENT_BATCHES_KEY: outlier_calcs[self.IS_SUFFICIENT_BATCHES_KEY], self.CONSTANT_COUNTS_KEY: channel_batch_cnts, self.MAX_VALS_KEY: max_vals}
        return info_dict

    def generate_detector_report(self, model: GraphModule) -> Tuple[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Determines whether input weight equalization is appropriate for a given module.\n\n        Takes advantage of the ModelReport Observer which records the relevant percentile information\n\n        Args:\n            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers\n\n        Returns a tuple with two elements:\n            String report of of whether there are outliers in the activations around certain modules\n            Dictionary mapping modules of interest to:\n                whether there were outliers found in activation before\n                the number of batches used for each channel\n                whether fraction of applicable batches used is above fraction_batches_used_threshold\n                their p_r metric compared to the threshold\n                the threshold used to make the recommendation\n                the reference_percentile used to make the recommendation\n                the channel axis used to determine individual channels\n                the constant batch counts per channel\n                the per channel max values\n        '
        info_dict = self._generate_info_dict(model)
        outlier_string = 'Outlier detection report: \n'
        added_module: bool = False
        module_suggestion_str = 'For Module {} looked at with axis {}: \n'
        channel_suggestion_str = '\tFor channel {}, we found outliers in the preceding activation data with {}.\n'
        channel_max_value_str = 'a max value across all batches of {}'
        note_string = 'Note: outlier detection is only reliable for {}. We recommend {} to ensure the most accurate results.'
        note_distribution = 'stationary distributions'
        note_rec = 'running the static vs. dynamic detector to ensure activation data before modules above is stationary'
        constant_str = '\tFor channel {}, we found {} constant value batches. {}\n'
        constant_suggestion = 'We recommend taking a look at the dict and data to see how frequent this occurred and why.'
        for module_fqn in info_dict:
            mod_info: Dict[str, Any] = info_dict[module_fqn]
            added_model_desc = False
            for (index, outlier_detected) in enumerate(mod_info[self.OUTLIER_KEY]):
                if outlier_detected:
                    if not added_model_desc:
                        outlier_string += module_suggestion_str.format(module_fqn, self.ch_axis)
                        added_model_desc = True
                    added_module = True
                    max_value_found_str = channel_max_value_str.format(mod_info[self.MAX_VALS_KEY][index])
                    channel_str = channel_suggestion_str.format(index, max_value_found_str)
                    outlier_string += channel_str
                if mod_info[self.CONSTANT_COUNTS_KEY][index] != 0:
                    if not added_model_desc:
                        outlier_string += module_suggestion_str.format(module_fqn, self.ch_axis)
                        added_model_desc = True
                    constant_values_for_channel = mod_info[self.CONSTANT_COUNTS_KEY][index]
                    formatted_str = constant_str.format(index, constant_values_for_channel, constant_suggestion)
                    outlier_string += formatted_str
                    added_module = True
        if added_module:
            note_composed = note_string.format(note_distribution, note_rec)
            outlier_string += note_composed
        else:
            outlier_string += 'There were no outliers found in the activations.\n'
        return (outlier_string, info_dict)
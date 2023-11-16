"""
This module contains tooling to compare weights and activations
across models. Example usage::

    import copy
    import torch
    import torch.ao.quantization.quantize_fx as quantize_fx
    import torch.ao.ns._numeric_suite_fx as ns

    m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1)).eval()
    mp = quantize_fx.prepare_fx(m, {'': torch.ao.quantization.default_qconfig})
    # We convert a copy because we need the original prepared model
    # to be available for comparisons, and `quantize_fx.convert_fx` is inplace.
    mq = quantize_fx.convert_fx(copy.deepcopy(mp))

    #
    # Comparing weights
    #

    # extract weight pairs
    weight_comparison = ns.extract_weights('a', mp, 'b', mq)

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        weight_comparison, 'a', 'b', torch.ao.ns.fx.utils.compute_sqnr,
        'sqnr')

    # weight_comparison contains the weights from `mp` and `mq` stored
    # in pairs, and can be used for further analysis.


    #
    # Comparing activations, with error propagation
    #

    # add loggers
    mp_ns, mq_ns = ns.add_loggers(
        'a', copy.deepcopy(mp),
        'b', copy.deepcopy(mq),
        ns.OutputLogger)

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_ns(datum)
    mq_ns(datum)

    # extract intermediate activations
    act_comparison = ns.extract_logger_info(
        mp_ns, mq_ns, ns.OutputLogger, 'b')

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        act_comparison, 'a', 'b', torch.ao.ns.fx.utils.compute_sqnr,
        'sqnr')

    # act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

    #
    # Comparing activations, without error propagation
    #

    # create shadow model
    mp_shadows_mq = ns.add_shadow_loggers(
        'a', copy.deepcopy(mp),
        'b', copy.deepcopy(mq),
        ns.OutputLogger)

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_shadows_mq(datum)

    # extract intermediate activations
    shadow_act_comparison = ns.extract_shadow_logger_info(
        mp_shadows_mq, ns.OutputLogger, 'b')

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        shadow_act_comparison, 'a', 'b', torch.ao.ns.fx.utils.compute_sqnr,
        'sqnr')

    # shadow_act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

"""
import collections
import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.ns.fx.mappings import get_base_name_to_sets_of_related_ops
from torch.ao.ns.fx.graph_matcher import get_matching_subgraph_pairs, get_type_a_related_to_b
from .fx.weight_utils import extract_weight_from_node
from .fx.graph_passes import add_loggers_to_model, create_a_shadows_b
from .fx.utils import rekey_logger_info_on_node_name_of_model, maybe_add_missing_fqns, get_target_type_str
from .fx.ns_types import NSSingleResultValuesType, NSResultsType, NSNodeTargetType
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_root_node_getter
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.fx.match_utils import _find_matches
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from torch.ao.quantization.fx.qconfig_mapping_utils import _generate_node_name_to_qconfig
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization import QConfigMapping
from torch.ao.ns.fx.n_shadows_utils import OutputProp, _get_dedup_subgraphs, SHADOW_WRAPPER_NODE_NAME_PREFIX, group_results_by_subgraph, create_results_comparison, print_n_shadows_summary, create_n_transformed_and_logged_copies_of_subgraph, create_add_loggers_graph, extract_weight_comparison
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from typing import Dict, Tuple, Callable, List, Optional, Set, Any, Type
RNNReturnType = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

class OutputLogger(nn.Module):
    """
    Base class for capturing intermediate values.
    """
    stats: List[torch.Tensor]
    stats_rnn: List[RNNReturnType]
    _is_impure = True

    def __init__(self, ref_node_name: str, prev_node_name: str, model_name: str, ref_name: str, prev_node_target_type: str, ref_node_target_type: str, results_type: str, index_within_arg: int, index_of_arg: int, fqn: Optional[str], qconfig_str: Optional[str]=''):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.stats: List[torch.Tensor] = []
        self.stats_rnn: List[RNNReturnType] = []
        self.ref_node_name = ref_node_name
        self.prev_node_name = prev_node_name
        self.model_name = model_name
        self.ref_name = ref_name
        self.prev_node_target_type = prev_node_target_type
        self.ref_node_target_type = ref_node_target_type
        self.results_type = results_type
        self.index_within_arg = index_within_arg
        self.index_of_arg = index_of_arg
        self.fqn = fqn
        self.enabled = True
        self.qconfig_str = qconfig_str
        self.save_activations = True

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        '
        if not self.enabled:
            return x
        if not self.save_activations:
            return x
        if isinstance(x, torch.Tensor):
            self.stats.append(x.detach())
        elif isinstance(x, tuple) and len(x) == 2 and (len(x[1]) == 2):
            new_res = (x[0].detach(), (x[1][0].detach(), x[1][1].detach()))
            self.stats_rnn.append(new_res)
        return x

    def __repr__(self):
        if False:
            print('Hello World!')
        clean_dict = {k: v for (k, v) in self.__dict__.items() if k != 'training' and (not k.startswith('_'))}
        return f'OutputLogger({clean_dict})'

class OutputComparisonLogger(OutputLogger):
    """
    Same as OutputLogger, but also requires the original activation
    in order to calculate the comparison at calibration time
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.comparison_fn = torch.ao.ns.fx.utils.compute_sqnr
        self.comparison_fn_name = 'sqnr'
        self.comparisons = []

    def forward(self, x, x_ref):
        if False:
            for i in range(10):
                print('nop')
        '\n        '
        if not self.enabled:
            return x
        assert isinstance(x, torch.Tensor), 'non-tensor inputs not yet supported'
        if self.save_activations:
            self.stats.append(x.detach())
        self.comparisons.append(self.comparison_fn(x, x_ref))
        return x

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        clean_dict = {k: v for (k, v) in self.__dict__.items() if k != 'training' and (not k.startswith('_'))}
        return f'OutputComparisonLogger({clean_dict})'

class NSTracer(quantize_fx.QuantizationTracer):
    """
    Just like a regular FX quantization tracer, but treats observers and fake_quantize
    modules as leaf modules.
    """

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        '
        if isinstance(m, torch.ao.quantization.ObserverBase):
            return True
        elif isinstance(m, torch.ao.quantization.FakeQuantizeBase):
            return True
        return super().is_leaf_module(m, module_qualified_name)

def _extract_weights_one_model(model_name: str, model: GraphModule, nodes_and_names_to_instrument: List[Tuple[Node, str]], results: NSResultsType, op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable, Callable]]]=None) -> None:
    if False:
        i = 10
        return i + 15
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._extract_weights_one_model')
    for (node, ref_name) in nodes_and_names_to_instrument:
        res_type = NSSingleResultValuesType.WEIGHT.value
        extracted_weight = extract_weight_from_node(node, model, op_to_type_to_weight_extraction_fn)
        if extracted_weight:
            if ref_name not in results:
                results[ref_name] = {res_type: {}}
            results[ref_name][res_type][model_name] = [extracted_weight]

def _extract_weights_impl(model_name_a: str, gm_a: GraphModule, model_name_b: str, gm_b: GraphModule, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None, op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable, Callable]]]=None) -> NSResultsType:
    if False:
        for i in range(10):
            print('nop')
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._extract_weights_impl')
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map)
    nodes_and_names_to_instrument_a: List[Tuple[Node, str]] = []
    nodes_and_names_to_instrument_b: List[Tuple[Node, str]] = []
    for (match_name, match) in matched_subgraph_pairs.items():
        (subgraph_a, subgraph_b) = match
        nodes_and_names_to_instrument_a.append((subgraph_a.base_op_node, match_name))
        nodes_and_names_to_instrument_b.append((subgraph_b.base_op_node, match_name))
    results: NSResultsType = {}
    _extract_weights_one_model(model_name_a, gm_a, nodes_and_names_to_instrument_a, results, op_to_type_to_weight_extraction_fn)
    _extract_weights_one_model(model_name_b, gm_b, nodes_and_names_to_instrument_b, results, op_to_type_to_weight_extraction_fn)
    maybe_add_missing_fqns(results)
    results = rekey_logger_info_on_node_name_of_model(results, model_name_b)
    return results

def extract_weights(model_name_a: str, model_a: nn.Module, model_name_b: str, model_b: nn.Module, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None, op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable, Callable]]]=None) -> NSResultsType:
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract weights from model A and model B, and return a comparison.\n\n    Args:\n        model_name_a: string name of model A to use in results\n        model_a: model A\n        model_name_b: string name of model B to use in results\n        model_b: model B\n        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change\n        unmatchable_types_map: optional override of unmatchable types, subject to change\n        op_to_type_to_weight_extraction_fn: optional override of function which extracts weight\n            from a type, subject to change\n\n    Return:\n        NSResultsType, containing the weight comparisons\n    '
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx.extract_weights')
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = get_type_a_related_to_b(base_name_to_sets_of_related_ops)
    skipped_module_names: List[str] = []
    skipped_module_classes: List[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(model_a, 'node_name_to_scope')
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(model_b, 'node_name_to_scope')
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _extract_weights_impl(model_name_a, gm_a, model_name_b, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map, op_to_type_to_weight_extraction_fn)

def _add_loggers_one_model(model_name: str, model: GraphModule, nodes_and_names_to_instrument_inputs: List[Tuple[Node, str, str]], nodes_and_names_to_instrument_outputs: List[Tuple[Node, str, str]], logger_cls: Callable) -> nn.Module:
    if False:
        return 10
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._add_loggers_one_model')
    node_to_instrument_inputs_to_ref_name: Dict[Node, Tuple[str, str]] = {}
    node_to_instrument_outputs_to_ref_name: Dict[Node, Tuple[str, str]] = {}
    for (node, ref_name, ref_node_type) in nodes_and_names_to_instrument_inputs:
        node_to_instrument_inputs_to_ref_name[node] = (ref_name, ref_node_type)
    for (node, ref_name, ref_node_type) in nodes_and_names_to_instrument_outputs:
        node_to_instrument_outputs_to_ref_name[node] = (ref_name, ref_node_type)
    model = add_loggers_to_model(model, node_to_instrument_inputs_to_ref_name, node_to_instrument_outputs_to_ref_name, logger_cls, model_name)
    return model

def _add_loggers_impl(name_a: str, gm_a: GraphModule, name_b: str, gm_b: GraphModule, logger_cls: Callable, should_log_inputs: bool, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None) -> Tuple[nn.Module, nn.Module]:
    if False:
        return 10
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._add_loggers_impl')
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map)
    nodes_and_names_to_instrument_inputs_a = []
    nodes_and_names_to_instrument_inputs_b = []
    nodes_and_names_to_instrument_outputs_a = []
    nodes_and_names_to_instrument_outputs_b = []
    for (match_name, (subgraph_a, subgraph_b)) in matched_subgraph_pairs.items():
        ref_node_type_a = get_target_type_str(subgraph_a.base_op_node, gm_a)
        ref_node_type_b = get_target_type_str(subgraph_b.base_op_node, gm_b)
        if should_log_inputs:
            nodes_and_names_to_instrument_inputs_a.append((subgraph_a.start_node, match_name, ref_node_type_a))
            nodes_and_names_to_instrument_inputs_b.append((subgraph_b.start_node, match_name, ref_node_type_b))
        nodes_and_names_to_instrument_outputs_a.append((subgraph_a.end_node, match_name, ref_node_type_a))
        nodes_and_names_to_instrument_outputs_b.append((subgraph_b.end_node, match_name, ref_node_type_b))
    new_model_a = _add_loggers_one_model(name_a, gm_a, nodes_and_names_to_instrument_inputs_a, nodes_and_names_to_instrument_outputs_a, logger_cls)
    new_model_b = _add_loggers_one_model(name_b, gm_b, nodes_and_names_to_instrument_inputs_b, nodes_and_names_to_instrument_outputs_b, logger_cls)
    return (new_model_a, new_model_b)

def add_loggers(name_a: str, model_a: nn.Module, name_b: str, model_b: nn.Module, logger_cls: Callable, should_log_inputs: bool=False, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None) -> Tuple[nn.Module, nn.Module]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Instrument model A and model B with loggers.\n\n    Args:\n        name_a: string name of model A to use in results\n        model_a: model A\n        name_b: string name of model B to use in results\n        model_b: model B\n        logger_cls: class of Logger to use\n        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change\n        unmatchable_types_map: optional override of unmatchable types, subject to change\n\n    Return:\n        Returns a tuple of (model_a_with_loggers, model_b_with_loggers).  Modifies both models inplace.\n    '
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx.add_loggers')
    skipped_module_names: List[str] = []
    skipped_module_classes: List[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(model_a, 'node_name_to_scope')
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(model_b, 'node_name_to_scope')
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _add_loggers_impl(name_a, gm_a, name_b, gm_b, logger_cls, should_log_inputs=should_log_inputs, base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops, unmatchable_types_map=unmatchable_types_map)

def _extract_logger_info_one_model(model: nn.Module, results: NSResultsType, logger_cls: Callable) -> None:
    if False:
        print('Hello World!')
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._extract_logger_info_one_model')
    for (gm_name, mod) in model.named_modules():
        is_logger = isinstance(mod, logger_cls) or (isinstance(mod, torch.jit.RecursiveScriptModule) and mod.original_name == 'OutputLogger')
        if is_logger:
            key = mod.ref_name
            if key not in results:
                results[key] = {}
            assert mod.model_name not in results[key], f'{mod.model_name} is already present in results'
            if mod.results_type not in results[key]:
                results[key][mod.results_type] = {}
            if mod.model_name not in results[key][mod.results_type]:
                results[key][mod.results_type][mod.model_name] = []
            stats_to_use = mod.stats
            if len(mod.stats_rnn) > 0:
                stats_to_use = mod.stats_rnn
            data = {'type': mod.results_type, 'values': stats_to_use, 'ref_node_name': mod.ref_node_name, 'ref_node_target_type': mod.ref_node_target_type, 'prev_node_name': mod.prev_node_name, 'prev_node_target_type': mod.prev_node_target_type, 'index_within_arg': mod.index_within_arg, 'index_of_arg': mod.index_of_arg, 'fqn': mod.fqn, 'qconfig_str': mod.qconfig_str}
            if hasattr(mod, 'comparisons'):
                data['comparisons'] = mod.comparisons
                data['comparison_fn_name'] = mod.comparison_fn_name
            else:
                data['comparisons'] = []
                data['comparison_fn_name'] = ''
            results[key][mod.results_type][mod.model_name].append(data)
            results[key][mod.results_type][mod.model_name].sort(key=lambda res: f"{res['index_of_arg']}:{res['index_within_arg']}")

def extract_logger_info(model_a: nn.Module, model_b: nn.Module, logger_cls: Callable, model_name_to_use_for_layer_names: str) -> NSResultsType:
    if False:
        return 10
    '\n    Traverse all loggers in `model_a` and `model_b`, and extract the logged\n    information.\n\n    Args:\n        model_a: model A\n        model_b: model B\n        logger_cls: class of Logger to use\n        model_name_to_use_for_layer_names: string name of model to use for\n          layer names in the output\n\n    Return:\n        NSResultsType, containing the logged comparisons\n    '
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx.extract_logger_info')
    results: NSResultsType = {}
    for model in (model_a, model_b):
        _extract_logger_info_one_model(model, results, logger_cls)
    maybe_add_missing_fqns(results)
    results = rekey_logger_info_on_node_name_of_model(results, model_name_to_use_for_layer_names)
    return results

def _add_shadow_loggers_impl(name_a: str, gm_a: GraphModule, name_b: str, gm_b: GraphModule, logger_cls: Callable, should_log_inputs: bool, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, node_type_to_io_type_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None) -> nn.Module:
    if False:
        i = 10
        return i + 15
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._add_shadow_loggers_impl')
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map)
    gm_a_shadows_b = create_a_shadows_b(name_a, gm_a, name_b, gm_b, matched_subgraph_pairs, logger_cls, should_log_inputs=should_log_inputs, node_type_to_io_type_map=node_type_to_io_type_map)
    return gm_a_shadows_b

def add_shadow_loggers(name_a: str, model_a: nn.Module, name_b: str, model_b: nn.Module, logger_cls: Callable, should_log_inputs: bool=False, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, node_type_to_io_type_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None) -> nn.Module:
    if False:
        for i in range(10):
            print('nop')
    '\n    Instrument model A and model B with shadow loggers.\n\n    Args:\n        name_a: string name of model A to use in results\n        model_a: model A\n        name_b: string name of model B to use in results\n        model_b: model B\n        logger_cls: class of Logger to use\n        should_log_inputs: whether to log inputs\n        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change\n        unmatchable_types_map: optional override of unmatchable types, subject to change\n    '
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx.add_shadow_loggers')
    skipped_module_names: List[str] = []
    skipped_module_classes: List[Callable] = []
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(model_a, 'node_name_to_scope')
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(model_b, 'node_name_to_scope')
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    return _add_shadow_loggers_impl(name_a, gm_a, name_b, gm_b, logger_cls, should_log_inputs=should_log_inputs, base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops, node_type_to_io_type_map=node_type_to_io_type_map, unmatchable_types_map=unmatchable_types_map)

def extract_shadow_logger_info(model_a_shadows_b: nn.Module, logger_cls: Callable, model_name_to_use_for_layer_names: str) -> NSResultsType:
    if False:
        print('Hello World!')
    '\n    Traverse all loggers in a shadow model, and extract the logged\n    information.\n\n    Args:\n        model_a_shadows_b: shadow model\n        logger_cls: class of Logger to use\n        model_name_to_use_for_layer_names: string name of model to use for\n          layer names in the output\n\n    Return:\n        NSResultsType, containing the logged comparisons\n    '
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx.extract_shadow_logger_info')
    results: NSResultsType = collections.defaultdict(dict)
    _extract_logger_info_one_model(model_a_shadows_b, results, logger_cls)
    maybe_add_missing_fqns(results)
    results = rekey_logger_info_on_node_name_of_model(results, model_name_to_use_for_layer_names)
    return dict(results)

def extend_logger_results_with_comparison(results: NSResultsType, model_name_1: str, model_name_2: str, comparison_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], comparison_name: str) -> None:
    if False:
        return 10
    "\n    Compares the logged values from `model_name_2` against the corresponding\n    values in `model_name_1`, using `comparison_fn`. Records the result\n    in `model_name_2`'s results under `comparison_name`. Modifies `results` inplace.\n\n    Args:\n        results: the result data structure from `extract_logger_info` or\n          `extract_shadow_logger_info`.\n        model_name_1: string name of model 1\n        model_name_2: string name of model 2\n        comparison_fn: function to compare two Tensors\n        comparison_name: string name of model to use for\n          layer names in the output\n    "
    for results_type_to_results in results.values():
        for model_name_to_results in results_type_to_results.values():
            assert model_name_1 in model_name_to_results, f'{model_name_1} not found in results'
            assert model_name_2 in model_name_to_results, f'{model_name_2} not found in results'
            results_1 = model_name_to_results[model_name_1]
            results_2 = model_name_to_results[model_name_2]
            for result_2 in results_2:
                index_within_arg_2 = result_2['index_within_arg']
                index_of_arg_2 = result_2['index_of_arg']
                result_1 = None
                for cur_result_1 in results_1:
                    index_within_arg_1 = cur_result_1['index_within_arg']
                    index_of_arg_1 = cur_result_1['index_of_arg']
                    if index_within_arg_1 == index_within_arg_2 and index_of_arg_1 == index_of_arg_2:
                        result_1 = cur_result_1
                        break
                assert result_1 is not None
                values_1 = result_1['values']
                values_2 = result_2['values']
                result_2[comparison_name] = []
                for (value_1, value_2) in zip(values_1, values_2):
                    comparison_result = comparison_fn(value_1, value_2)
                    result_2[comparison_name].append(comparison_result)

def prepare_n_shadows_model(model: torch.nn.Module, example_inputs: Any, qconfig_multi_mapping: QConfigMultiMapping, backend_config: BackendConfig, custom_prepare_fn: Optional[Callable]=None, custom_prepare_kwargs: Optional[Dict[str, Any]]=None, custom_tracer: Any=None) -> GraphModule:
    if False:
        print('Hello World!')
    '\n    Given a model with a graph with M ops such as\n\n\n      args_kwargs_m -> op_m -> output_m\n\n\n    And a set of N qconfigs for each op, creates a new model, with\n    each of the subgraph of `op_m` transformed into\n\n    .. code::\n\n           |---------> op_m_n -> log_m_n\n           |                     /\n      args_kwargs_m ---------> op_m -> log_m_0\n\n    Where op_m_n is op_m wrapped in a submodule and transformed with\n    qconfig_n, and its inner graph looks like\n\n    .. code::\n\n      args_m -------- op_m_prepared_with_qconfig_n -> out_m_n\n                  /\n      kwargs_m ---\n\n    This is useful for testing different quantization of multiple layers in\n    a single pass through the model.\n\n    High level TODOs for future PRs:\n    * figure out a better way to name the output structure\n    * return a results data structure instead of printing it out\n    * add examples to docblocks\n    '
    if custom_tracer is None:
        tracer = quantize_fx.QuantizationTracer([], [])
    else:
        tracer = custom_tracer
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    mt._node_name_to_scope = tracer.node_name_to_scope
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: List[str] = []
    standalone_module_classes: List[Type] = []
    custom_module_classes: List[Type] = []
    matches = _find_matches(mt.graph, modules, patterns, root_node_getter_mapping, standalone_module_names, standalone_module_classes, custom_module_classes)
    subgraphs_dedup: Dict[str, List[Node]] = _get_dedup_subgraphs(matches)
    list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]] = []
    for qconfig_mapping in qconfig_multi_mapping.qconfig_mappings_list:
        node_name_to_qconfig = _generate_node_name_to_qconfig(mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope)
        list_of_node_name_to_qconfig.append(node_name_to_qconfig)
    for (subgraph_idx, (match_name, nodes_in_this_subgraph)) in enumerate(subgraphs_dedup.items()):
        create_n_transformed_and_logged_copies_of_subgraph(mt, subgraph_idx, match_name, nodes_in_this_subgraph, qconfig_multi_mapping.qconfig_mappings_list, list_of_node_name_to_qconfig, custom_prepare_fn, custom_prepare_kwargs)
    return mt

def _prepare_n_shadows_add_loggers_model(model: torch.nn.Module, example_inputs: Any, qconfig_mapping: QConfigMapping, backend_config: BackendConfig) -> torch.nn.Module:
    if False:
        i = 10
        return i + 15
    '\n    Note: this API is not recommended for wide usage, it is only\n    provided for customers who need to migrate from the `add_loggers`\n    API.\n\n    This creates a model which provides logging for the following\n    problem: if we quantize `model` with `qconfig_mapping` and feed\n    the same input through both models, log the comparisons of\n    corresponding intermediate layers.\n\n    The problem is solved with a single model.  Specifically, we\n    partition `model` into N subgraphs, create a copy of each relevant\n    subgraph, wrap it in a module, apply the quantization API to that\n    module, and hook up loggers to measure the comparisons.\n\n    Example starting graph:\n\n      x0 -> op0 -> x1 -> op1 -> x2\n\n    Example config: quantize op0 to int8, do nothing to op1.\n    The following graph will be created:\n\n    .. code::\n\n      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log\n       \\                        \\                           \\       # noqa: W605\n         ---> op0_1 -> x1_1 ----> clog -> op1_0 -> x2_1 ----> clog\n\n    Where op0_0 is op0, op0_1 is op0 wrapped in a submodule and quantized\n    to int8, op1_0 is op1 (appearing in the graph twice), log is a logger,\n    and clog is a comparison logger.\n    '
    tracer = quantize_fx.QuantizationTracer([], [])
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    mt._node_name_to_scope = tracer.node_name_to_scope
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: List[str] = []
    standalone_module_classes: List[Type] = []
    custom_module_classes: List[Type] = []
    matches = _find_matches(mt.graph, modules, patterns, root_node_getter_mapping, standalone_module_names, standalone_module_classes, custom_module_classes)
    subgraphs_dedup: Dict[str, List[Node]] = _get_dedup_subgraphs(matches)
    node_name_to_qconfig = _generate_node_name_to_qconfig(mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope)
    create_add_loggers_graph(mt, subgraphs_dedup, qconfig_mapping, node_name_to_qconfig)
    return mt

def _n_shadows_compare_weights(model: torch.nn.Module, example_inputs: Any, qconfig_mapping: QConfigMapping, backend_config: BackendConfig) -> NSResultsType:
    if False:
        i = 10
        return i + 15
    '\n    Note: this API is not recommended for wide usage, it is only\n    provided for customers who need to migrate from the `add_loggers`\n    API.\n    '
    qconfig_multi_mapping = QConfigMultiMapping.from_list_qconfig_mapping([qconfig_mapping])
    mp = prepare_n_shadows_model(model, example_inputs, qconfig_multi_mapping, backend_config)
    mp(*example_inputs)
    mq = convert_n_shadows_model(mp)
    weight_comparison = extract_weight_comparison(mq)
    return weight_comparison

def loggers_set_enabled(model: torch.nn.Module, enabled: bool) -> None:
    if False:
        while True:
            i = 10
    "\n    Sets the `enabled` setting on a `model`'s loggers\n    "
    for (name, child) in model.named_modules():
        if isinstance(child, OutputLogger):
            child.enabled = enabled

def loggers_set_save_activations(model: torch.nn.Module, save_activations: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Sets the `save_activations` setting on a `model`'s loggers\n    "
    for (name, child) in model.named_modules():
        if isinstance(child, OutputLogger):
            child.save_activations = save_activations

def convert_n_shadows_model(model: GraphModule, custom_convert_fn: Optional[Callable]=None, custom_convert_kwargs: Optional[Dict[str, Any]]=None) -> GraphModule:
    if False:
        return 10
    '\n    Given a model from `prepare_n_shadows_model`, runs `convert_fx`\n    on each shadow submodule.\n    '
    for node in model.graph.nodes:
        if node.name.startswith(SHADOW_WRAPPER_NODE_NAME_PREFIX):
            orig_mod = getattr(model, node.name)
            if custom_convert_fn is None:
                converted_mod = torch.ao.quantization.quantize_fx.convert_fx(orig_mod)
            else:
                if custom_convert_kwargs is None:
                    custom_convert_kwargs = {}
                converted_mod = custom_convert_fn(orig_mod, **custom_convert_kwargs)
            setattr(model, node.name, converted_mod)
    return model

def extract_results_n_shadows_model(model: torch.nn.Module) -> NSResultsType:
    if False:
        for i in range(10):
            print('nop')
    '\n    Extracts logger results from `model`.\n    '
    results: NSResultsType = {}
    _extract_logger_info_one_model(model, results, OutputLogger)
    return results

def print_comparisons_n_shadows_model(results: NSResultsType) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Prints a summary of extracted `results`.\n    '
    results_grouped = group_results_by_subgraph(results)
    results_comparison = create_results_comparison(results_grouped)
    print_n_shadows_summary(results_comparison)
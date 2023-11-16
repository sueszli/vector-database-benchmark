"""Pass manager for optimization level 2, providing medium optimization.

Level 2 pass manager: medium optimization by noise adaptive qubit mapping and
gate cancellation using commutativity rules.
"""
from __future__ import annotations
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager

def level_2_pass_manager(pass_manager_config: PassManagerConfig) -> StagedPassManager:
    if False:
        for i in range(10):
            print('nop')
    'Level 2 pass manager: medium optimization by initial layout selection and\n    gate cancellation using commutativity rules.\n\n    This pass manager applies the user-given initial layout. If none is given, a search\n    for a perfect layout (i.e. one that satisfies all 2-qubit interactions) is conducted.\n    If no such layout is found, qubits are laid out on the most densely connected subset\n    which also exhibits the best gate fidelities.\n\n    The pass manager then transforms the circuit to match the coupling constraints.\n    It is then unrolled to the basis, and any flipped cx directions are fixed.\n    Finally, optimizations in the form of commutative gate cancellation and redundant\n    reset removal are performed.\n\n    Args:\n        pass_manager_config: configuration of the pass manager.\n\n    Returns:\n        a level 2 pass manager.\n\n    Raises:\n        TranspilerError: if the passmanager config is invalid.\n    '
    plugin_manager = PassManagerStagePluginManager()
    basis_gates = pass_manager_config.basis_gates
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    init_method = pass_manager_config.init_method or 'default'
    layout_method = pass_manager_config.layout_method or 'default'
    routing_method = pass_manager_config.routing_method or 'sabre'
    translation_method = pass_manager_config.translation_method or 'translator'
    optimization_method = pass_manager_config.optimization_method or 'default'
    scheduling_method = pass_manager_config.scheduling_method or 'default'
    target = pass_manager_config.target
    routing_pm = plugin_manager.get_passmanager_stage('routing', routing_method, pass_manager_config, optimization_level=2)
    if coupling_map or initial_layout:
        layout = plugin_manager.get_passmanager_stage('layout', layout_method, pass_manager_config, optimization_level=2)
        routing = routing_pm
    else:
        layout = None
        routing = None
    translation = plugin_manager.get_passmanager_stage('translation', translation_method, pass_manager_config, optimization_level=2)
    if coupling_map and (not coupling_map.is_symmetric) or (target is not None and target.get_non_global_operation_names(strict_direction=True)):
        pre_optimization = common.generate_pre_op_passmanager(target, coupling_map, remove_reset_in_zero=False)
    else:
        pre_optimization = common.generate_pre_op_passmanager(remove_reset_in_zero=False)
    optimization = plugin_manager.get_passmanager_stage('optimization', optimization_method, pass_manager_config, optimization_level=2)
    sched = plugin_manager.get_passmanager_stage('scheduling', scheduling_method, pass_manager_config, optimization_level=2)
    pre_init = common.generate_control_flow_options_check(layout_method=layout_method, routing_method=routing_method, translation_method=translation_method, optimization_method=optimization_method, scheduling_method=scheduling_method, basis_gates=basis_gates, target=target)
    init = plugin_manager.get_passmanager_stage('init', init_method, pass_manager_config, optimization_level=2)
    return StagedPassManager(pre_init=pre_init, init=init, layout=layout, routing=routing, translation=translation, pre_optimization=pre_optimization, optimization=optimization, scheduling=sched)
import copy
import warnings
from functools import cached_property
from pathlib import Path, PurePath
from typing import Optional, Tuple
from vyper import ast as vy_ast
from vyper.codegen import module
from vyper.codegen.core import anchor_opt_level
from vyper.codegen.global_context import GlobalContext
from vyper.codegen.ir_node import IRnode
from vyper.compiler.input_bundle import FilesystemInputBundle, InputBundle
from vyper.compiler.settings import OptimizationLevel, Settings
from vyper.exceptions import StructureException
from vyper.ir import compile_ir, optimizer
from vyper.semantics import set_data_positions, validate_semantics
from vyper.semantics.types.function import ContractFunctionT
from vyper.typing import StorageLayout
DEFAULT_CONTRACT_NAME = PurePath('VyperContract.vy')

class CompilerData:
    """
    Object for fetching and storing compiler data for a Vyper contract.

    This object acts as a wrapper over the pure compiler functions, triggering
    compilation phases as needed and providing the data for use when generating
    the final compiler outputs.

    Attributes
    ----------
    vyper_module : vy_ast.Module
        Top-level Vyper AST node
    vyper_module_folded : vy_ast.Module
        Folded Vyper AST
    global_ctx : GlobalContext
        Sorted, contextualized representation of the Vyper AST
    ir_nodes : IRnode
        IR used to generate deployment bytecode
    ir_runtime : IRnode
        IR used to generate runtime bytecode
    assembly : list
        Assembly instructions for deployment bytecode
    assembly_runtime : list
        Assembly instructions for runtime bytecode
    bytecode : bytes
        Deployment bytecode
    bytecode_runtime : bytes
        Runtime bytecode
    """

    def __init__(self, source_code: str, input_bundle: InputBundle=None, contract_path: Path | PurePath=DEFAULT_CONTRACT_NAME, source_id: int=0, settings: Settings=None, storage_layout: StorageLayout=None, show_gas_estimates: bool=False, no_bytecode_metadata: bool=False) -> None:
        if False:
            return 10
        '\n        Initialization method.\n\n        Arguments\n        ---------\n        source_code: str\n            Vyper source code.\n        contract_path: Path, optional\n            The name of the contract being compiled.\n        source_id: int, optional\n            ID number used to identify this contract in the source map.\n        settings: Settings\n            Set optimization mode.\n        show_gas_estimates: bool, optional\n            Show gas estimates for abi and ir output modes\n        no_bytecode_metadata: bool, optional\n            Do not add metadata to bytecode. Defaults to False\n        '
        self.contract_path = contract_path
        self.source_code = source_code
        self.source_id = source_id
        self.storage_layout_override = storage_layout
        self.show_gas_estimates = show_gas_estimates
        self.no_bytecode_metadata = no_bytecode_metadata
        self.settings = settings or Settings()
        self.input_bundle = input_bundle or FilesystemInputBundle([Path('.')])
        _ = self._generate_ast

    @cached_property
    def _generate_ast(self):
        if False:
            print('Hello World!')
        contract_name = str(self.contract_path)
        (settings, ast) = generate_ast(self.source_code, self.source_id, contract_name)
        if settings.evm_version is not None:
            if self.settings.evm_version is not None and self.settings.evm_version != settings.evm_version:
                raise StructureException(f'compiler settings indicate evm version {self.settings.evm_version}, but source pragma indicates {settings.evm_version}.')
            self.settings.evm_version = settings.evm_version
        if settings.optimize is not None:
            if self.settings.optimize is not None and self.settings.optimize != settings.optimize:
                raise StructureException(f'compiler options indicate optimization mode {self.settings.optimize}, but source pragma indicates {settings.optimize}.')
            self.settings.optimize = settings.optimize
        if self.settings.optimize is None:
            self.settings.optimize = OptimizationLevel.default()
        return ast

    @cached_property
    def vyper_module(self):
        if False:
            return 10
        return self._generate_ast

    @cached_property
    def vyper_module_unfolded(self) -> vy_ast.Module:
        if False:
            return 10
        return generate_unfolded_ast(self.contract_path, self.vyper_module, self.input_bundle)

    @cached_property
    def _folded_module(self):
        if False:
            return 10
        return generate_folded_ast(self.contract_path, self.vyper_module, self.input_bundle, self.storage_layout_override)

    @property
    def vyper_module_folded(self) -> vy_ast.Module:
        if False:
            i = 10
            return i + 15
        (module, storage_layout) = self._folded_module
        return module

    @property
    def storage_layout(self) -> StorageLayout:
        if False:
            return 10
        (module, storage_layout) = self._folded_module
        return storage_layout

    @property
    def global_ctx(self) -> GlobalContext:
        if False:
            while True:
                i = 10
        return GlobalContext(self.vyper_module_folded)

    @cached_property
    def _ir_output(self):
        if False:
            for i in range(10):
                print('nop')
        return generate_ir_nodes(self.global_ctx, self.settings.optimize)

    @property
    def ir_nodes(self) -> IRnode:
        if False:
            print('Hello World!')
        (ir, ir_runtime) = self._ir_output
        return ir

    @property
    def ir_runtime(self) -> IRnode:
        if False:
            return 10
        (ir, ir_runtime) = self._ir_output
        return ir_runtime

    @property
    def function_signatures(self) -> dict[str, ContractFunctionT]:
        if False:
            while True:
                i = 10
        _ = self._ir_output
        fs = self.vyper_module_folded.get_children(vy_ast.FunctionDef)
        return {f.name: f._metadata['type'] for f in fs}

    @cached_property
    def assembly(self) -> list:
        if False:
            return 10
        return generate_assembly(self.ir_nodes, self.settings.optimize)

    @cached_property
    def assembly_runtime(self) -> list:
        if False:
            i = 10
            return i + 15
        return generate_assembly(self.ir_runtime, self.settings.optimize)

    @cached_property
    def bytecode(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        insert_compiler_metadata = not self.no_bytecode_metadata
        return generate_bytecode(self.assembly, insert_compiler_metadata=insert_compiler_metadata)

    @cached_property
    def bytecode_runtime(self) -> bytes:
        if False:
            return 10
        return generate_bytecode(self.assembly_runtime, insert_compiler_metadata=False)

    @cached_property
    def blueprint_bytecode(self) -> bytes:
        if False:
            while True:
                i = 10
        blueprint_preamble = b'\xfeq\x00'
        blueprint_bytecode = blueprint_preamble + self.bytecode
        len_bytes = len(blueprint_bytecode).to_bytes(2, 'big')
        deploy_bytecode = b'a' + len_bytes + b'=\x81`\n=9\xf3'
        return deploy_bytecode + blueprint_bytecode

def generate_ast(source_code: str, source_id: int, contract_name: str) -> tuple[Settings, vy_ast.Module]:
    if False:
        i = 10
        return i + 15
    '\n    Generate a Vyper AST from source code.\n\n    Arguments\n    ---------\n    source_code : str\n        Vyper source code.\n    source_id : int\n        ID number used to identify this contract in the source map.\n    contract_name: str\n        Name of the contract.\n\n    Returns\n    -------\n    vy_ast.Module\n        Top-level Vyper AST node\n    '
    return vy_ast.parse_to_ast_with_settings(source_code, source_id, contract_name)

def generate_unfolded_ast(contract_path: Path | PurePath, vyper_module: vy_ast.Module, input_bundle: InputBundle) -> vy_ast.Module:
    if False:
        i = 10
        return i + 15
    vy_ast.validation.validate_literal_nodes(vyper_module)
    vy_ast.folding.replace_builtin_functions(vyper_module)
    with input_bundle.search_path(contract_path.parent):
        validate_semantics(vyper_module, input_bundle)
    return vyper_module

def generate_folded_ast(contract_path: Path, vyper_module: vy_ast.Module, input_bundle: InputBundle, storage_layout_overrides: StorageLayout=None) -> Tuple[vy_ast.Module, StorageLayout]:
    if False:
        return 10
    '\n    Perform constant folding operations on the Vyper AST.\n\n    Arguments\n    ---------\n    vyper_module : vy_ast.Module\n        Top-level Vyper AST node\n\n    Returns\n    -------\n    vy_ast.Module\n        Folded Vyper AST\n    StorageLayout\n        Layout of variables in storage\n    '
    vy_ast.validation.validate_literal_nodes(vyper_module)
    vyper_module_folded = copy.deepcopy(vyper_module)
    vy_ast.folding.fold(vyper_module_folded)
    with input_bundle.search_path(contract_path.parent):
        validate_semantics(vyper_module_folded, input_bundle)
    symbol_tables = set_data_positions(vyper_module_folded, storage_layout_overrides)
    return (vyper_module_folded, symbol_tables)

def generate_ir_nodes(global_ctx: GlobalContext, optimize: OptimizationLevel) -> tuple[IRnode, IRnode]:
    if False:
        return 10
    '\n    Generate the intermediate representation (IR) from the contextualized AST.\n\n    This phase also includes IR-level optimizations.\n\n    This function returns three values: deployment bytecode, runtime bytecode\n    and the function signatures of the contract\n\n    Arguments\n    ---------\n    global_ctx : GlobalContext\n        Contextualized Vyper AST\n\n    Returns\n    -------\n    (IRnode, IRnode)\n        IR to generate deployment bytecode\n        IR to generate runtime bytecode\n    '
    with anchor_opt_level(optimize):
        (ir_nodes, ir_runtime) = module.generate_ir_for_module(global_ctx)
    if optimize != OptimizationLevel.NONE:
        ir_nodes = optimizer.optimize(ir_nodes)
        ir_runtime = optimizer.optimize(ir_runtime)
    return (ir_nodes, ir_runtime)

def generate_assembly(ir_nodes: IRnode, optimize: Optional[OptimizationLevel]=None) -> list:
    if False:
        return 10
    '\n    Generate assembly instructions from IR.\n\n    Arguments\n    ---------\n    ir_nodes : str\n        Top-level IR nodes. Can be deployment or runtime IR.\n\n    Returns\n    -------\n    list\n        List of assembly instructions.\n    '
    optimize = optimize or OptimizationLevel.default()
    assembly = compile_ir.compile_to_assembly(ir_nodes, optimize=optimize)
    if _find_nested_opcode(assembly, 'DEBUG'):
        warnings.warn('This code contains DEBUG opcodes! The DEBUG opcode will only work in a supported EVM! It will FAIL on all other nodes!')
    return assembly

def _find_nested_opcode(assembly, key):
    if False:
        print('Hello World!')
    if key in assembly:
        return True
    else:
        sublists = [sub for sub in assembly if isinstance(sub, list)]
        return any((_find_nested_opcode(x, key) for x in sublists))

def generate_bytecode(assembly: list, insert_compiler_metadata: bool) -> bytes:
    if False:
        print('Hello World!')
    '\n    Generate bytecode from assembly instructions.\n\n    Arguments\n    ---------\n    assembly : list\n        Assembly instructions. Can be deployment or runtime assembly.\n\n    Returns\n    -------\n    bytes\n        Final compiled bytecode.\n    '
    return compile_ir.assembly_to_evm(assembly, insert_compiler_metadata=insert_compiler_metadata)[0]
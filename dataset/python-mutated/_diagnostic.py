"""Diagnostic components for TorchScript based ONNX export, i.e. `torch.onnx.export`."""
from __future__ import annotations
import contextlib
import gzip
from collections.abc import Generator
from typing import List, Optional
import torch
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
from torch.utils import cpp_backtrace

def _cpp_call_stack(frames_to_skip: int=0, frames_to_log: int=32) -> infra.Stack:
    if False:
        i = 10
        return i + 15
    'Returns the current C++ call stack.\n\n    This function utilizes `torch.utils.cpp_backtrace` to get the current C++ call stack.\n    The returned C++ call stack is a concatenated string of the C++ call stack frames.\n    Each frame is separated by a newline character, in the same format of\n    r"frame #[0-9]+: (?P<frame_info>.*)". More info at `c10/util/Backtrace.cpp`.\n\n    '
    frames = cpp_backtrace.get_cpp_backtrace(frames_to_skip, frames_to_log).split('\n')
    frame_messages = []
    for frame in frames:
        segments = frame.split(':', 1)
        if len(segments) == 2:
            frame_messages.append(segments[1].strip())
        else:
            frame_messages.append('<unknown frame>')
    return infra.Stack(frames=[infra.StackFrame(location=infra.Location(message=message)) for message in frame_messages])

class TorchScriptOnnxExportDiagnostic(infra.Diagnostic):
    """Base class for all export diagnostics.

    This class is used to represent all export diagnostics. It is a subclass of
    infra.Diagnostic, and adds additional methods to add more information to the
    diagnostic.
    """
    python_call_stack: Optional[infra.Stack] = None
    cpp_call_stack: Optional[infra.Stack] = None

    def __init__(self, *args, frames_to_skip: int=1, cpp_stack: bool=False, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.python_call_stack = self.record_python_call_stack(frames_to_skip=frames_to_skip)
        if cpp_stack:
            self.cpp_call_stack = self.record_cpp_call_stack(frames_to_skip=frames_to_skip)

    def record_cpp_call_stack(self, frames_to_skip: int) -> infra.Stack:
        if False:
            return 10
        'Records the current C++ call stack in the diagnostic.'
        stack = _cpp_call_stack(frames_to_skip=frames_to_skip)
        stack.message = 'C++ call stack'
        self.with_stack(stack)
        return stack

class ExportDiagnosticEngine:
    """PyTorch ONNX Export diagnostic engine.

    The only purpose of creating this class instead of using `DiagnosticContext` directly
    is to provide a background context for `diagnose` calls inside exporter.

    By design, one `torch.onnx.export` call should initialize one diagnostic context.
    All `diagnose` calls inside exporter should be made in the context of that export.
    However, since diagnostic context is currently being accessed via a global variable,
    there is no guarantee that the context is properly initialized. Therefore, we need
    to provide a default background context to fallback to, otherwise any invocation of
    exporter internals, e.g. unit tests, will fail due to missing diagnostic context.
    This can be removed once the pipeline for context to flow through the exporter is
    established.
    """
    contexts: List[infra.DiagnosticContext]
    _background_context: infra.DiagnosticContext

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.contexts = []
        self._background_context = infra.DiagnosticContext(name='torch.onnx', version=torch.__version__)

    @property
    def background_context(self) -> infra.DiagnosticContext:
        if False:
            while True:
                i = 10
        return self._background_context

    def create_diagnostic_context(self, name: str, version: str, options: Optional[infra.DiagnosticOptions]=None) -> infra.DiagnosticContext:
        if False:
            return 10
        'Creates a new diagnostic context.\n\n        Args:\n            name: The subject name for the diagnostic context.\n            version: The subject version for the diagnostic context.\n            options: The options for the diagnostic context.\n\n        Returns:\n            A new diagnostic context.\n        '
        if options is None:
            options = infra.DiagnosticOptions()
        context: infra.DiagnosticContext[infra.Diagnostic] = infra.DiagnosticContext(name, version, options)
        self.contexts.append(context)
        return context

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Clears all diagnostic contexts.'
        self.contexts.clear()
        self._background_context.diagnostics.clear()

    def to_json(self) -> str:
        if False:
            print('Hello World!')
        return formatter.sarif_to_json(self.sarif_log())

    def dump(self, file_path: str, compress: bool=False) -> None:
        if False:
            print('Hello World!')
        'Dumps the SARIF log to a file.'
        if compress:
            with gzip.open(file_path, 'wt') as f:
                f.write(self.to_json())
        else:
            with open(file_path, 'w') as f:
                f.write(self.to_json())

    def sarif_log(self):
        if False:
            i = 10
            return i + 15
        log = sarif.SarifLog(version=sarif_version.SARIF_VERSION, schema_uri=sarif_version.SARIF_SCHEMA_LINK, runs=[context.sarif() for context in self.contexts])
        log.runs.append(self._background_context.sarif())
        return log
engine = ExportDiagnosticEngine()
_context = engine.background_context

@contextlib.contextmanager
def create_export_diagnostic_context() -> Generator[infra.DiagnosticContext, None, None]:
    if False:
        return 10
    'Create a diagnostic context for export.\n\n    This is a workaround for code robustness since diagnostic context is accessed by\n    export internals via global variable. See `ExportDiagnosticEngine` for more details.\n    '
    global _context
    assert _context == engine.background_context, 'Export context is already set. Nested export is not supported.'
    _context = engine.create_diagnostic_context('torch.onnx.export', torch.__version__)
    try:
        yield _context
    finally:
        _context = engine.background_context

def diagnose(rule: infra.Rule, level: infra.Level, message: Optional[str]=None, frames_to_skip: int=2, **kwargs) -> TorchScriptOnnxExportDiagnostic:
    if False:
        i = 10
        return i + 15
    'Creates a diagnostic and record it in the global diagnostic context.\n\n    This is a wrapper around `context.log` that uses the global diagnostic\n    context.\n    '
    diagnostic = TorchScriptOnnxExportDiagnostic(rule, level, message, frames_to_skip=frames_to_skip, **kwargs)
    export_context().log(diagnostic)
    return diagnostic

def export_context() -> infra.DiagnosticContext:
    if False:
        while True:
            i = 10
    global _context
    return _context
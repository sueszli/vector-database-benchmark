import os
import sys
import warnings
import torch
from typing import List, Dict, Optional
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestScriptModuleInstanceAttributeTypeAnnotation(JitTestCase):

    def test_annotated_falsy_base_type(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.x: int = 0

            def forward(self, x: int):
                if False:
                    return 10
                self.x = x
                return 1
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), (1,))
        assert len(w) == 0

    def test_annotated_nonempty_container(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.x: List[int] = [1, 2, 3]

            def forward(self, x: List[int]):
                if False:
                    return 10
                self.x = x
                return 1
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_empty_tensor(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.x: torch.Tensor = torch.empty(0)

            def forward(self, x: torch.Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = x
                return self.x
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), (torch.rand(2, 3),))
        assert len(w) == 0

    def test_annotated_with_jit_attribute(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.x = torch.jit.Attribute([], List[int])

            def forward(self, x: List[int]):
                if False:
                    print('Hello World!')
                self.x = x
                return self.x
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_annotation_only(self):
        if False:
            return 10

        class M(torch.nn.Module):
            x: List[int]

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.x = []

            def forward(self, y: List[int]):
                if False:
                    return 10
                self.x = y
                return self.x
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_annotation_and_init_annotation(self):
        if False:
            return 10

        class M(torch.nn.Module):
            x: List[int]

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.x: List[int] = []

            def forward(self, y: List[int]):
                if False:
                    return 10
                self.x = y
                return self.x
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_jit_annotation(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):
            x: List[int]

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.x: List[int] = torch.jit.annotate(List[int], [])

            def forward(self, y: List[int]):
                if False:
                    i = 10
                    return i + 15
                self.x = y
                return self.x
        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_empty_list(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.x: List[int] = []

            def forward(self, x: List[int]):
                if False:
                    while True:
                        i = 10
                self.x = x
                return 1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Tried to set nonexistent attribute', 'self.x = x'):
            with self.assertWarnsRegex(UserWarning, "doesn't support instance-level annotations on empty non-base types"):
                torch.jit.script(M())

    def test_annotated_empty_dict(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.x: Dict[str, int] = {}

            def forward(self, x: Dict[str, int]):
                if False:
                    i = 10
                    return i + 15
                self.x = x
                return 1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Tried to set nonexistent attribute', 'self.x = x'):
            with self.assertWarnsRegex(UserWarning, "doesn't support instance-level annotations on empty non-base types"):
                torch.jit.script(M())

    def test_annotated_empty_optional(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.x: Optional[str] = None

            def forward(self, x: Optional[str]):
                if False:
                    return 10
                self.x = x
                return 1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Wrong type for attribute assignment', 'self.x = x'):
            with self.assertWarnsRegex(UserWarning, "doesn't support instance-level annotations on empty non-base types"):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_list(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.x = torch.jit.annotate(List[int], [])

            def forward(self, x: List[int]):
                if False:
                    return 10
                self.x = x
                return 1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Tried to set nonexistent attribute', 'self.x = x'):
            with self.assertWarnsRegex(UserWarning, "doesn't support instance-level annotations on empty non-base types"):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_dict(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.x = torch.jit.annotate(Dict[str, int], {})

            def forward(self, x: Dict[str, int]):
                if False:
                    i = 10
                    return i + 15
                self.x = x
                return 1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Tried to set nonexistent attribute', 'self.x = x'):
            with self.assertWarnsRegex(UserWarning, "doesn't support instance-level annotations on empty non-base types"):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_optional(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.x = torch.jit.annotate(Optional[str], None)

            def forward(self, x: Optional[str]):
                if False:
                    while True:
                        i = 10
                self.x = x
                return 1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Wrong type for attribute assignment', 'self.x = x'):
            with self.assertWarnsRegex(UserWarning, "doesn't support instance-level annotations on empty non-base types"):
                torch.jit.script(M())

    def test_annotated_with_torch_jit_import(self):
        if False:
            return 10
        from torch import jit

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.x = jit.annotate(Optional[str], None)

            def forward(self, x: Optional[str]):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = x
                return 1
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Wrong type for attribute assignment', 'self.x = x'):
            with self.assertWarnsRegex(UserWarning, "doesn't support instance-level annotations on empty non-base types"):
                torch.jit.script(M())
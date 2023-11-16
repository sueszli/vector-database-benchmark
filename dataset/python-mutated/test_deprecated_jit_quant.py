import torch
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.common_utils import suppress_warnings
from torch.testing._internal.jit_utils import JitTestCase
from typing import Tuple
import copy

class TestDeprecatedJitQuantized(JitTestCase):

    @skipIfNoFBGEMM
    def test_rnn_cell_quantized(self):
        if False:
            i = 10
            return i + 15
        (d_in, d_hid) = (2, 2)
        for cell in [torch.nn.LSTMCell(d_in, d_hid).float(), torch.nn.GRUCell(d_in, d_hid).float(), torch.nn.RNNCell(d_in, d_hid).float()]:
            if isinstance(cell, torch.nn.LSTMCell):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRUCell):
                num_chunks = 3
            elif isinstance(cell, torch.nn.RNNCell):
                num_chunks = 1
            vals = [[100, -155], [100, -155], [-155, 100], [-155, 100], [100, -155], [-155, 100], [-155, 100], [100, -155]]
            vals = vals[:d_hid * num_chunks]
            cell.weight_ih = torch.nn.Parameter(torch.tensor(vals, dtype=torch.float), requires_grad=False)
            cell.weight_hh = torch.nn.Parameter(torch.tensor(vals, dtype=torch.float), requires_grad=False)
            ref = copy.deepcopy(cell)
            cell = torch.jit.quantized.quantize_rnn_cell_modules(cell)
            x = torch.tensor([[100, -155], [-155, 100], [100, -155]], dtype=torch.float)
            h0_vals = [[-155, 100], [-155, 155], [100, -155]]
            hx = torch.tensor(h0_vals, dtype=torch.float)
            if isinstance(cell, torch.jit.quantized.QuantizedLSTMCell):
                cx = torch.tensor(h0_vals, dtype=torch.float)
                hiddens = (hx, cx)
            else:
                hiddens = hx
            if isinstance(cell, torch.jit.quantized.QuantizedLSTMCell):

                class ScriptWrapper(torch.jit.ScriptModule):

                    def __init__(self, cell):
                        if False:
                            for i in range(10):
                                print('nop')
                        super().__init__()
                        self.cell = cell

                    @torch.jit.script_method
                    def forward(self, x: torch.Tensor, hiddens: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
                        if False:
                            print('Hello World!')
                        return self.cell(x, hiddens)
            else:

                class ScriptWrapper(torch.jit.ScriptModule):

                    def __init__(self, cell):
                        if False:
                            return 10
                        super().__init__()
                        self.cell = cell

                    @torch.jit.script_method
                    def forward(self, x: torch.Tensor, hiddens: torch.Tensor) -> torch.Tensor:
                        if False:
                            return 10
                        return self.cell(x, hiddens)
            cell = ScriptWrapper(cell)
            outs = cell(x, hiddens)
            cell = self.getExportImportCopyWithPacking(cell)
            outs = cell(x, hiddens)
            ref_outs = ref(x, hiddens)
            self.assertEqual(len(outs), len(ref_outs))
            for (out, ref_out) in zip(outs, ref_outs):
                torch.testing.assert_close(out, ref_out)

    @skipIfNoFBGEMM
    def test_rnn_quantized(self):
        if False:
            print('Hello World!')
        (d_in, d_hid) = (2, 2)
        for cell in [torch.nn.LSTM(d_in, d_hid).float(), torch.nn.GRU(d_in, d_hid).float()]:
            vals = [[100, -155], [100, -155], [-155, 100], [-155, 100], [100, -155], [-155, 100], [-155, 100], [100, -155]]
            if isinstance(cell, torch.nn.LSTM):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRU):
                num_chunks = 3
            vals = vals[:d_hid * num_chunks]
            cell.weight_ih_l0 = torch.nn.Parameter(torch.tensor(vals, dtype=torch.float), requires_grad=False)
            cell.weight_hh_l0 = torch.nn.Parameter(torch.tensor(vals, dtype=torch.float), requires_grad=False)
            ref = copy.deepcopy(cell)
            cell_int8 = torch.jit.quantized.quantize_rnn_modules(cell, dtype=torch.int8)
            cell_fp16 = torch.jit.quantized.quantize_rnn_modules(cell, dtype=torch.float16)
            niter = 10
            x = torch.tensor([[100, -155], [-155, 100], [100, -155]], dtype=torch.float).unsqueeze(0).repeat(niter, 1, 1)
            h0_vals = [[-155, 100], [-155, 155], [100, -155]]
            hx = torch.tensor(h0_vals, dtype=torch.float).unsqueeze(0)
            cx = torch.tensor(h0_vals, dtype=torch.float).unsqueeze(0)
            if isinstance(ref, torch.nn.LSTM):
                hiddens = (hx, cx)
            elif isinstance(ref, torch.nn.GRU):
                hiddens = hx
            (ref_out, ref_hid) = ref(x, hiddens)
            (output_int8, final_hiddens_int8) = cell_int8(x, hiddens)
            torch.testing.assert_close(output_int8, ref_out)
            for (out, ref) in zip(final_hiddens_int8, ref_hid):
                torch.testing.assert_close(out, ref)
            (output_fp16, final_hiddens_fp16) = cell_fp16(x, hiddens)
            torch.testing.assert_close(output_fp16, ref_out)
            for (out, ref) in zip(final_hiddens_fp16, ref_hid):
                torch.testing.assert_close(out, ref)

            def compare_quantized_unquantized(ScriptWrapper, cell):
                if False:
                    while True:
                        i = 10
                wrapper = ScriptWrapper(cell)
                (script_out, script_hid) = wrapper(x, hiddens)
                torch.testing.assert_close(script_out, ref_out)
                for (out, ref) in zip(script_hid, ref_hid):
                    torch.testing.assert_close(out, ref)
                export_import_wrapper = self.getExportImportCopyWithPacking(wrapper)
                (ei_out, ei_hid) = export_import_wrapper(x, hiddens)
                torch.testing.assert_close(ei_out, ref_out)
                for (out, ref) in zip(ei_hid, ref_hid):
                    torch.testing.assert_close(out, ref)
            if isinstance(cell, torch.jit.quantized.QuantizedGRU):

                class ScriptWrapper(torch.jit.ScriptModule):

                    def __init__(self, cell):
                        if False:
                            for i in range(10):
                                print('nop')
                        super().__init__()
                        self.cell = cell

                    @torch.jit.script_method
                    def forward(self, x: torch.Tensor, hiddens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                        if False:
                            for i in range(10):
                                print('nop')
                        return self.cell(x, hiddens)
                compare_quantized_unquantized(ScriptWrapper, cell)
            elif isinstance(cell, torch.jit.quantized.QuantizedLSTM):
                for cell in [cell_int8, cell_fp16]:

                    class ScriptWrapper(torch.jit.ScriptModule):

                        def __init__(self, cell):
                            if False:
                                return 10
                            super().__init__()
                            self.cell = cell

                        @torch.jit.script_method
                        def forward(self, x, hiddens):
                            if False:
                                return 10
                            return self.cell(x, hiddens)
                    compare_quantized_unquantized(ScriptWrapper, cell)
    if 'fbgemm' in torch.backends.quantized.supported_engines:

        @suppress_warnings
        def test_quantization_modules(self):
            if False:
                print('Hello World!')
            (K1, N1) = (2, 2)

            class FooBar(torch.nn.Module):

                def __init__(self):
                    if False:
                        while True:
                            i = 10
                    super().__init__()
                    self.linear1 = torch.nn.Linear(K1, N1).float()

                def forward(self, x):
                    if False:
                        for i in range(10):
                            print('nop')
                    x = self.linear1(x)
                    return x
            fb = FooBar()
            fb.linear1.weight = torch.nn.Parameter(torch.tensor([[-150, 100], [100, -150]], dtype=torch.float), requires_grad=False)
            fb.linear1.bias = torch.nn.Parameter(torch.zeros_like(fb.linear1.bias), requires_grad=False)
            x = (torch.rand(1, K1).float() - 0.5) / 10.0
            value = torch.tensor([[100, -150]], dtype=torch.float)
            y_ref = fb(value)
            fb_int8 = torch.jit.quantized.quantize_linear_modules(fb)
            traced_int8 = torch.jit.trace(fb_int8, (x,))
            fb_int8 = self.getExportImportCopyWithPacking(traced_int8)
            y_int8 = fb_int8(value)
            fb_fp16 = torch.jit.quantized.quantize_linear_modules(fb, torch.float16)
            traced_fp16 = torch.jit.trace(fb_fp16, (x,))
            fb_fp16 = self.getExportImportCopyWithPacking(traced_fp16)
            y_fp16 = fb_fp16(value)
            torch.testing.assert_close(y_int8, y_ref, rtol=0.0001, atol=0.001)
            torch.testing.assert_close(y_fp16, y_ref, rtol=0.0001, atol=0.001)

    @skipIfNoFBGEMM
    def test_erase_class_tensor_shapes(self):
        if False:
            i = 10
            return i + 15

        class Linear(torch.nn.Module):

            def __init__(self, in_features, out_features):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                qweight = torch._empty_affine_quantized([out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8)
                self._packed_weight = torch.ops.quantized.linear_prepack(qweight)

            @torch.jit.export
            def __getstate__(self):
                if False:
                    i = 10
                    return i + 15
                return (torch.ops.quantized.linear_unpack(self._packed_weight)[0], self.training)

            def forward(self):
                if False:
                    print('Hello World!')
                return self._packed_weight

            @torch.jit.export
            def __setstate__(self, state):
                if False:
                    return 10
                self._packed_weight = torch.ops.quantized.linear_prepack(state[0])
                self.training = state[1]

            @property
            def weight(self):
                if False:
                    i = 10
                    return i + 15
                return torch.ops.quantized.linear_unpack(self._packed_weight)[0]

            @weight.setter
            def weight(self, w):
                if False:
                    i = 10
                    return i + 15
                self._packed_weight = torch.ops.quantized.linear_prepack(w)
        with torch._jit_internal._disable_emit_hooks():
            x = torch.jit.script(Linear(10, 10))
            torch._C._jit_pass_erase_shape_information(x.graph)
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_quantization.py TESTNAME\n\ninstead.')
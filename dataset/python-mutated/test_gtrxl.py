import pytest
import torch
from ding.torch_utils import GTrXL, GRUGatingUnit

@pytest.mark.unittest
class TestGTrXL:

    def test_GTrXl(self):
        if False:
            print('Hello World!')
        dim_size = 128
        seq_len = 64
        bs = 32
        embedding_dim = 256
        layer_num = 5
        mem_len = 40
        memory = [None, torch.rand(layer_num + 1, mem_len, bs, embedding_dim)]
        batch_first = [False, True]
        for i in range(2):
            m = memory[i]
            bf = batch_first[i]
            model = GTrXL(input_dim=dim_size, head_dim=2, embedding_dim=embedding_dim, memory_len=mem_len, head_num=2, mlp_num=2, layer_num=layer_num)
            input = torch.rand(seq_len, bs, dim_size)
            if bf:
                input = torch.transpose(input, 1, 0)
            input.requires_grad_(True)
            if m is None:
                model.reset_memory(batch_size=bs)
            else:
                model.reset_memory(state=m)
            output = model(input, batch_first=bf)
            target = torch.randn(output['logit'].shape)
            mse_loss = torch.nn.MSELoss()
            target = torch.randn(output['logit'].shape)
            loss = mse_loss(output['logit'], target)
            assert input.grad is None
            loss.backward()
            assert isinstance(input.grad, torch.Tensor)
            if bf is False:
                assert output['logit'].shape == (seq_len, bs, embedding_dim)
            else:
                assert output['logit'].shape == (bs, seq_len, embedding_dim)
            assert output['memory'].shape == (layer_num + 1, mem_len, bs, embedding_dim)
            memory_out = output['memory']
            if m is not None:
                assert torch.all(torch.eq(memory_out, m))

    def test_memory(self):
        if False:
            print('Hello World!')
        dim_size = 128
        seq_len = 4
        bs = 16
        embedding_dim = 128
        layer_num = 3
        mem_len = 8
        model = GTrXL(input_dim=dim_size, head_dim=2, embedding_dim=embedding_dim, memory_len=mem_len, head_num=2, mlp_num=2, layer_num=layer_num)
        memories = []
        outs = []
        for i in range(4):
            input = torch.rand(seq_len, bs, dim_size)
            output = model(input)
            memories.append(output['memory'])
            outs.append(output['logit'])
        assert sum(memories[0].flatten()) == 0
        assert torch.all(torch.eq(memories[1][-1][4:], outs[0]))
        assert sum(memories[1][-1][:4].flatten()) == 0
        assert torch.all(torch.eq(memories[2][-1][4:], outs[1]))
        assert torch.all(torch.eq(memories[2][-1][:4], outs[0]))
        assert torch.all(torch.eq(memories[3][-1][4:], outs[2]))
        assert torch.all(torch.eq(memories[3][-1][:4], outs[1]))

    def test_gru(self):
        if False:
            i = 10
            return i + 15
        input_dim = 32
        gru = GRUGatingUnit(input_dim, 1.0)
        x = torch.rand((4, 12, 32))
        y = torch.rand((4, 12, 32))
        out = gru(x, y)
        assert out.shape == x.shape
        gru = GRUGatingUnit(input_dim, 100000.0)
        out = gru(x, y)
        torch.testing.assert_close(out, x)
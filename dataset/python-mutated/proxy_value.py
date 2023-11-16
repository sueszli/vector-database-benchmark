from typing import Union
import torch

class ProxyValue:

    def __init__(self, data, proxy: Union[torch.fx.Proxy, torch.fx.Node]):
        if False:
            return 10
        self.data = data
        self.proxy_or_node = proxy

    @property
    def node(self) -> torch.fx.Node:
        if False:
            return 10
        if isinstance(self.proxy_or_node, torch.fx.Node):
            return self.proxy_or_node
        assert isinstance(self.proxy_or_node, torch.fx.Proxy)
        return self.proxy_or_node.node

    @property
    def proxy(self) -> torch.fx.Proxy:
        if False:
            while True:
                i = 10
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise RuntimeError(f"ProxyValue doesn't have attached Proxy object. Node: {self.proxy_or_node.format_node()}")
        return self.proxy_or_node

    def to_tensor(self) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        assert isinstance(self.data, torch.Tensor)
        return self.data

    def is_tensor(self) -> bool:
        if False:
            print('Hello World!')
        return isinstance(self.data, torch.Tensor)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        yield from self.data

    def __bool__(self) -> bool:
        if False:
            return 10
        return bool(self.data)
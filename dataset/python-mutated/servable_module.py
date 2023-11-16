from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple
import torch
from torch import Tensor

class ServableModule(ABC, torch.nn.Module):
    """The ServableModule provides a simple API to make your model servable.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Here is an example of how to use the ``ServableModule`` module.

    .. code-block:: python

        from typing import Dict, Any, Callable

        import torch

        from lightning.pytorch import Trainer
        from lightning.pytorch.demos.boring_classes import BoringModel
        from lightning.pytorch.serve.servable_module_validator import ServableModule, ServableModuleValidator


        class ServableBoringModel(BoringModel, ServableModule):
            def configure_payload(self) -> Dict[str, Any]:
                return {"body": {"x": list(range(32))}}

            def configure_serialization(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
                def deserialize(x):
                    return torch.tensor(x, dtype=torch.float)

                def serialize(x):
                    return x.tolist()

                return {"x": deserialize}, {"output": serialize}

            def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"output": torch.tensor([0, 1])}

            def configure_response(self):
                return {"output": [0, 1]}


        serve_cb = ServableModuleValidator()
        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=0,
            callbacks=[serve_cb],
        )
        trainer.fit(ServableBoringModel())
        assert serve_cb.resp.json() == {"output": [0, 1]}

    """

    @abstractmethod
    def configure_payload(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Returns a request payload as a dictionary.'

    @abstractmethod
    def configure_serialization(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
        if False:
            i = 10
            return i + 15
        'Returns a tuple of dictionaries.\n\n        The first dictionary contains the name of the ``serve_step`` input variables name as its keys\n        and the associated de-serialization function (e.g function to convert a payload to tensors).\n\n        The second dictionary contains the name of the ``serve_step`` output variables name as its keys\n        and the associated serialization function (e.g function to convert a tensors into payload).\n\n        '

    @abstractmethod
    def serve_step(self, *args: Tensor, **kwargs: Tensor) -> Dict[str, Tensor]:
        if False:
            while True:
                i = 10
        'Returns the predictions of your model as a dictionary.\n\n        .. code-block:: python\n\n            def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:\n                return {"predictions": self(x)}\n\n        Args:\n            args: The output from de-serializer functions provided by the ``configure_serialization`` hook.\n            kwargs: The keyword output of the de-serializer functions provided by the ``configure_serialization`` hook.\n\n        Return:\n            - ``dict`` - A dictionary with their associated tensors.\n\n        '

    @abstractmethod
    def configure_response(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Returns a response to validate the server response.'
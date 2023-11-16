from typing import Any
from unittest import mock
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.model_summary import summarize
from tests_pytorch.helpers.runif import RunIf

@RunIf(rich=True)
def test_rich_model_summary_callback():
    if False:
        i = 10
        return i + 15
    trainer = Trainer(callbacks=RichProgressBar())
    assert any((isinstance(cb, RichModelSummary) for cb in trainer.callbacks))
    assert isinstance(trainer.progress_bar_callback, RichProgressBar)

def test_rich_progress_bar_import_error(monkeypatch):
    if False:
        print('Hello World!')
    import lightning.pytorch.callbacks.rich_model_summary as imports
    monkeypatch.setattr(imports, '_RICH_AVAILABLE', False)
    with pytest.raises(ModuleNotFoundError, match='`RichModelSummary` requires `rich` to be installed.'):
        RichModelSummary()

@RunIf(rich=True)
@mock.patch('rich.console.Console.print', autospec=True)
@mock.patch('rich.table.Table.add_row', autospec=True)
def test_rich_summary_tuples(mock_table_add_row, mock_console):
    if False:
        while True:
            i = 10
    'Ensure that tuples are converted into string, and print is called correctly.'
    model_summary = RichModelSummary()

    class TestModel(BoringModel):

        @property
        def example_input_array(self) -> Any:
            if False:
                while True:
                    i = 10
            return torch.randn(4, 32)
    model = TestModel()
    summary = summarize(model)
    summary_data = summary._get_summary_data()
    model_summary.summarize(summary_data=summary_data, total_parameters=1, trainable_parameters=1, model_size=1)
    assert mock_console.call_count == 2
    (args, kwargs) = mock_table_add_row.call_args_list[0]
    assert args[1:] == ('0', 'layer', 'Linear', '66  ', '[4, 32]', '[4, 2]')
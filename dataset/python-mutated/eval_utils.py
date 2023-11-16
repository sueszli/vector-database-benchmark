import torch
import torch.nn.functional as F

def _replace_dropout_for_eval(m: torch.fx.GraphModule):
    if False:
        return 10
    '\n    Replace the aten training dropout pattern with a noop, intended for eval.\n\n    For models with dropout torch ops (nn.Dropout, F.dropout), calling model.eval()\n    effectively turns these dropout ops into noops. For exported models, however,\n    this is not done automatically, since the aten dropout patterns previously generated\n    for training remain in the graph. Here we rewrite these dropout patterns with noops\n    to avoid incorrectly applying further dropout during eval.\n\n    See https://github.com/pytorch/pytorch/issues/103681.\n    '
    from .utils import get_aten_graph_module
    m.graph.eliminate_dead_code()
    m.recompile()

    def dropout_train(x):
        if False:
            for i in range(10):
                print('nop')
        return F.dropout(x, p=0.5, training=True)

    def dropout_eval(x):
        if False:
            while True:
                i = 10
        return F.dropout(x, p=0.5, training=False)
    example_inputs = (torch.randn(1),)
    match_pattern = get_aten_graph_module(dropout_train, example_inputs)
    replacement_pattern = get_aten_graph_module(dropout_eval, example_inputs)
    from torch.fx.subgraph_rewriter import replace_pattern_with_filters
    replace_pattern_with_filters(m, match_pattern, replacement_pattern, match_filters=[], ignore_literals=True)
    m.recompile()

def _move_exported_model_to_eval(model: torch.fx.GraphModule):
    if False:
        return 10
    '\n    Move an exported GraphModule to eval mode.\n\n    This is equivalent to model.eval() but only for certain special ops like dropout.\n    QAT users should call this before performing inference on the model.\n    '
    _replace_dropout_for_eval(model)
    return model
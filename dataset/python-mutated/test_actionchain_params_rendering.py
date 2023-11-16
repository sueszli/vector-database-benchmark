from __future__ import absolute_import
import unittest2
import mock
from action_chain_runner import action_chain_runner as acr
from st2common.exceptions.action import ParameterRenderingFailedException
from st2common.models.system.actionchain import Node

class ActionChainRunnerResolveParamsTests(unittest2.TestCase):

    def test_render_params_action_context(self):
        if False:
            for i in range(10):
                print('nop')
        runner = acr.get_runner()
        chain_context = {'parent': {'execution_id': 'some_awesome_exec_id', 'user': 'dad'}, 'user': 'son', 'k1': 'v1'}
        task_params = {'exec_id': {'default': '{{action_context.parent.execution_id}}'}, 'k2': {}, 'foo': {'default': 1}}
        action_node = Node(name='test_action_context_params', ref='core.local', params=task_params)
        rendered_params = runner._resolve_params(action_node, {}, {}, {}, chain_context)
        self.assertEqual(rendered_params['exec_id']['default'], 'some_awesome_exec_id')

    def test_render_params_action_context_non_existent_member(self):
        if False:
            while True:
                i = 10
        runner = acr.get_runner()
        chain_context = {'parent': {'execution_id': 'some_awesome_exec_id', 'user': 'dad'}, 'user': 'son', 'k1': 'v1'}
        task_params = {'exec_id': {'default': '{{action_context.parent.yo_gimme_tha_key}}'}, 'k2': {}, 'foo': {'default': 1}}
        action_node = Node(name='test_action_context_params', ref='core.local', params=task_params)
        try:
            runner._resolve_params(action_node, {}, {}, {}, chain_context)
            self.fail('Should have thrown an instance of %s' % ParameterRenderingFailedException)
        except ParameterRenderingFailedException:
            pass

    def test_render_params_with_config(self):
        if False:
            print('Hello World!')
        with mock.patch('st2common.util.config_loader.ContentPackConfigLoader') as config_loader:
            config_loader().get_config.return_value = {'amazing_config_value_fo_lyfe': 'no'}
            runner = acr.get_runner()
            chain_context = {'parent': {'execution_id': 'some_awesome_exec_id', 'user': 'dad', 'pack': 'mom'}, 'user': 'son'}
            task_params = {'config_val': '{{config_context.amazing_config_value_fo_lyfe}}'}
            action_node = Node(name='test_action_context_params', ref='core.local', params=task_params)
            rendered_params = runner._resolve_params(action_node, {}, {}, {}, chain_context)
            self.assertEqual(rendered_params['config_val'], 'no')

    def test_init_params_vars_with_unicode_value(self):
        if False:
            print('Hello World!')
        chain_spec = {'vars': {'unicode_var': '٩(̾●̮̮̃̾•̃̾)۶ ٩(̾●̮̮̃̾•̃̾)۶ ćšž', 'unicode_var_param': '{{ param }}'}, 'chain': [{'name': 'c1', 'ref': 'core.local', 'parameters': {'cmd': 'echo {{ unicode_var }}'}}]}
        chain_holder = acr.ChainHolder(chainspec=chain_spec, chainname='foo')
        chain_holder.init_vars(action_parameters={'param': '٩(̾●̮̮̃̾•̃̾)۶'})
        expected = {'unicode_var': '٩(̾●̮̮̃̾•̃̾)۶ ٩(̾●̮̮̃̾•̃̾)۶ ćšž', 'unicode_var_param': '٩(̾●̮̮̃̾•̃̾)۶'}
        self.assertEqual(chain_holder.vars, expected)
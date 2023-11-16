from __future__ import annotations
import jinja2
from airflow.models.dag import DAG
from airflow.template.templater import Templater
from airflow.utils.context import Context

class TestTemplater:

    def test_get_template_env(self):
        if False:
            for i in range(10):
                print('nop')
        templater = Templater()
        dag = DAG(dag_id='test_dag', render_template_as_native_obj=True)
        env = templater.get_template_env(dag)
        assert isinstance(env, jinja2.Environment)
        assert not env.sandboxed
        templater = Templater()
        env = templater.get_template_env()
        assert isinstance(env, jinja2.Environment)
        assert env.sandboxed

    def test_prepare_template(self):
        if False:
            while True:
                i = 10
        templater = Templater()
        templater.prepare_template()

    def test_resolve_template_files_logs_exception(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        templater = Templater()
        templater.message = 'template_file.txt'
        templater.template_fields = ['message']
        templater.template_ext = ['.txt']
        templater.resolve_template_files()
        assert "Failed to resolve template field 'message'" in caplog.text

    def test_render_template(self):
        if False:
            while True:
                i = 10
        context = Context({'name': 'world'})
        templater = Templater()
        templater.message = 'Hello {{ name }}'
        templater.template_fields = ['message']
        templater.template_ext = ['.txt']
        rendered_content = templater.render_template(templater.message, context)
        assert rendered_content == 'Hello world'
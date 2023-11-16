import os.path
import tempfile
from io import StringIO
from django.core.management import call_command
from sentry.constants import MODULE_ROOT
from sentry.testutils.cases import TestCase

class TestGenerateControlsiloUrls(TestCase):

    def call_command(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        out = StringIO()
        call_command('generate_controlsilo_urls', *args, stdout=out, stderr=StringIO, **kwargs)
        return out.getvalue()

    def test_skip_includes(self):
        if False:
            print('Hello World!')
        result = self.call_command(format='js')
        assert "new RegExp('^api/0/$')" not in result

    def test_render_text(self):
        if False:
            i = 10
            return i + 15
        result = self.call_command(format='text')
        assert '^api/0/users/$' in result

    def test_render_code(self):
        if False:
            return 10
        result = self.call_command(format='js')
        assert "new RegExp('^api/0/users/$')," in result
        assert "new RegExp('^api/0/internal/integration-proxy/\\\\S*$')," in result
        assert 'const patterns' in result
        assert 'export default patterns;' in result

    def test_write_file(self):
        if False:
            i = 10
            return i + 15
        with tempfile.NamedTemporaryFile() as tf:
            self.call_command(format='js', output=tf.name)
            tf.seek(0)
            result = tf.read().decode('utf8')
        assert 'This is generated code' in result
        assert "new RegExp('^api/0/users/$')," in result
        assert 'const patterns' in result
        assert 'export default patterns;' in result

    def test_no_missing_urls(self):
        if False:
            i = 10
            return i + 15
        pattern_file = 'static/app/data/controlsiloUrlPatterns.ts'
        project_root = os.path.dirname(os.path.dirname(MODULE_ROOT))
        pattern_filepath = os.path.join(project_root, pattern_file)
        with open(pattern_filepath) as f:
            current_state = f.read()
        result = self.call_command(format='js')
        for line in result.splitlines():
            msg = f'\n            New control silo URL patterns detected!\n\n            The pattern: {line}\n\n            Does not exist in the current pattern inventory. You should regenerate\n            the pattern inventory with:\n\n            getsentry django generate_controlsilo_urls --format=js --output={pattern_file}\n\n            This command needs to be run in a getsentry environment\n            in order to not lose patterns that are important for sentry.io\n            '
            assert line in current_state, msg
from . import Framework

class Markdown(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.text = 'MyTitle\n=======\n\nIssue #1'
        self.repo = self.g.get_user().get_repo('PyGithub')

    def testRenderMarkdown(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.g.render_markdown(self.text), '<h1><a name="mytitle" class="anchor" href="#mytitle"><span class="mini-icon mini-icon-link"></span></a>MyTitle</h1><p>Issue #1</p>')

    def testRenderGithubFlavoredMarkdown(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.g.render_markdown(self.text, self.repo), '<h1>MyTitle</h1><p>Issue <a href="https://github.com/jacquev6/PyGithub/issues/1" class="issue-link" title="Gitub -&gt; Github everywhere">#1</a></p>')
from ..models import GitHubCore
"\ngithub3.repos.issue_import\n==========================\n\nThis module contains the ImportedIssue object for Github's import issue API\n\n"

class ImportedIssue(GitHubCore):
    """
    The :class:`ImportedIssue <ImportedIssue>` object. This represents
    information from the Import Issue API.

    See also: https://gist.github.com/jonmagic/5282384165e0f86ef105
    """
    IMPORT_CUSTOM_HEADERS = {'Accept': 'application/vnd.github.golden-comet-preview+json'}

    def _update_attributes(self, json):
        if False:
            i = 10
            return i + 15
        self.id = json.get('id', None)
        self.status = json.get('status', None)
        self.url = json.get('url', None)
        self.created_at = json.get('created_at', None)
        self.updated_at = json.get('updated_at', None)
        self.import_issues_url = json.get('import_issues_url')
        self.repository_url = json.get('repository_url', None)
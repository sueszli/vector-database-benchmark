from google.cloud import dataform_v1beta1

def sample_pull_git_commits():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    author = dataform_v1beta1.CommitAuthor()
    author.name = 'name_value'
    author.email_address = 'email_address_value'
    request = dataform_v1beta1.PullGitCommitsRequest(name='name_value', author=author)
    client.pull_git_commits(request=request)
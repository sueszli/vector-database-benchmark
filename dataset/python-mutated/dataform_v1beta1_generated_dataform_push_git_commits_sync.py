from google.cloud import dataform_v1beta1

def sample_push_git_commits():
    if False:
        while True:
            i = 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.PushGitCommitsRequest(name='name_value')
    client.push_git_commits(request=request)
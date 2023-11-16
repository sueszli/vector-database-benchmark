from google.cloud import dataform_v1beta1

def sample_commit_workspace_changes():
    if False:
        for i in range(10):
            print('nop')
    client = dataform_v1beta1.DataformClient()
    author = dataform_v1beta1.CommitAuthor()
    author.name = 'name_value'
    author.email_address = 'email_address_value'
    request = dataform_v1beta1.CommitWorkspaceChangesRequest(name='name_value', author=author)
    client.commit_workspace_changes(request=request)
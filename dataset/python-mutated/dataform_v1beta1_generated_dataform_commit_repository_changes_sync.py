from google.cloud import dataform_v1beta1

def sample_commit_repository_changes():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    commit_metadata = dataform_v1beta1.CommitMetadata()
    commit_metadata.author.name = 'name_value'
    commit_metadata.author.email_address = 'email_address_value'
    request = dataform_v1beta1.CommitRepositoryChangesRequest(name='name_value', commit_metadata=commit_metadata)
    client.commit_repository_changes(request=request)
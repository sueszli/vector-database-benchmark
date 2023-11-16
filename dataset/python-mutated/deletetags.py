from boto3.resources.action import CustomModeledAction

def inject_delete_tags(event_emitter, **kwargs):
    if False:
        while True:
            i = 10
    action_model = {'request': {'operation': 'DeleteTags', 'params': [{'target': 'Resources[0]', 'source': 'identifier', 'name': 'Id'}]}}
    action = CustomModeledAction('delete_tags', action_model, delete_tags, event_emitter)
    action.inject(**kwargs)

def delete_tags(self, **kwargs):
    if False:
        print('Hello World!')
    kwargs['Resources'] = [self.id]
    return self.meta.client.delete_tags(**kwargs)
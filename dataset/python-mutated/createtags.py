def inject_create_tags(event_name, class_attributes, **kwargs):
    if False:
        i = 10
        return i + 15
    'This injects a custom create_tags method onto the ec2 service resource\n\n    This is needed because the resource model is not able to express\n    creating multiple tag resources based on the fact you can apply a set\n    of tags to multiple ec2 resources.\n    '
    class_attributes['create_tags'] = create_tags

def create_tags(self, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    self.meta.client.create_tags(**kwargs)
    resources = kwargs.get('Resources', [])
    tags = kwargs.get('Tags', [])
    tag_resources = []
    for resource in resources:
        for tag in tags:
            tag_resource = self.Tag(resource, tag['Key'], tag['Value'])
            tag_resources.append(tag_resource)
    return tag_resources
def validate_tag_update(update):
    if False:
        while True:
            i = 10
    '\n    Property: ResourceUpdateConstraint.TagUpdateOnProvisionedProduct\n    '
    valid_tag_update_values = ['ALLOWED', 'NOT_ALLOWED']
    if update not in valid_tag_update_values:
        raise ValueError('{} is not a valid tag update value'.format(update))
    return update
from dagster import asset

@asset(group_name='cereal_assets')
def nabisco_cereals():
    if False:
        i = 10
        return i + 15
    return [1, 2, 3]
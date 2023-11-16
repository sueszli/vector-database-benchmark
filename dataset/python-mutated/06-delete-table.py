"""
Purpose

Deletes the Amazon DynamoDB table used in the demonstration.
"""
import boto3

def delete_dax_table(dyn_resource=None):
    if False:
        return 10
    '\n    Deletes the demonstration table.\n\n    :param dyn_resource: Either a Boto3 or DAX resource.\n    '
    if dyn_resource is None:
        dyn_resource = boto3.resource('dynamodb')
    table = dyn_resource.Table('TryDaxTable')
    table.delete()
    print(f'Deleting {table.name}...')
    table.wait_until_not_exists()
if __name__ == '__main__':
    delete_dax_table()
    print('Table deleted!')
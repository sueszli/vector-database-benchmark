import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from .. import _utilities
from . import outputs
__all__ = ['SqlResourceSqlContainerArgs', 'SqlResourceSqlContainer']

@pulumi.input_type
class SqlResourceSqlContainerArgs:

    def __init__(__self__):
        if False:
            return 10
        '\n        The set of arguments for constructing a SqlResourceSqlContainer resource.\n        '
        pass

class SqlResourceSqlContainer(pulumi.CustomResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, __props__=None):
        if False:
            i = 10
            return i + 15
        '\n        An Azure Cosmos DB container.\n        API Version: 2021-03-15.\n\n        ## Example Usage\n        ### CosmosDBSqlContainerCreateUpdate\n\n        ```python\n        import pulumi\n        import pulumi_azure_native as azure_native\n\n        sql_resource_sql_container = azure_native.documentdb.SqlResourceSqlContainer("sqlResourceSqlContainer",\n            account_name="ddb1",\n            container_name="containerName",\n            database_name="databaseName",\n            location="West US",\n            options=azure_native.documentdb.CreateUpdateOptionsArgs(),\n            resource=azure_native.documentdb.SqlContainerResourceArgs(\n                conflict_resolution_policy=azure_native.documentdb.ConflictResolutionPolicyArgs(\n                    conflict_resolution_path="/path",\n                    mode="LastWriterWins",\n                ),\n                default_ttl=100,\n                id="containerName",\n                indexing_policy=azure_native.documentdb.IndexingPolicyArgs(\n                    automatic=True,\n                    excluded_paths=[],\n                    included_paths=[azure_native.documentdb.IncludedPathArgs(\n                        indexes=[\n                            azure_native.documentdb.IndexesArgs(\n                                data_type="String",\n                                kind="Range",\n                                precision=-1,\n                            ),\n                            azure_native.documentdb.IndexesArgs(\n                                data_type="Number",\n                                kind="Range",\n                                precision=-1,\n                            ),\n                        ],\n                        path="/*",\n                    )],\n                    indexing_mode="consistent",\n                ),\n                partition_key=azure_native.documentdb.ContainerPartitionKeyArgs(\n                    kind="Hash",\n                    paths=["/AccountNumber"],\n                ),\n                unique_key_policy=azure_native.documentdb.UniqueKeyPolicyArgs(\n                    unique_keys=[azure_native.documentdb.UniqueKeyArgs(\n                        paths=["/testPath"],\n                    )],\n                ),\n            ),\n            resource_group_name="rg1",\n            tags={})\n\n        ```\n\n        ## Import\n\n        An existing resource can be imported using its type token, name, and identifier, e.g.\n\n        ```sh\n        $ pulumi import azure-native:documentdb:SqlResourceSqlContainer containerName /subscriptions/subid/resourceGroups/rg1/providers/Microsoft.DocumentDB/databaseAccounts/ddb1/sqlDatabases/databaseName/sqlContainers/containerName \n        ```\n\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: Optional[SqlResourceSqlContainerArgs]=None, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            i = 10
            return i + 15
        '\n        An Azure Cosmos DB container.\n        API Version: 2021-03-15.\n\n        ## Example Usage\n        ### CosmosDBSqlContainerCreateUpdate\n\n        ```python\n        import pulumi\n        import pulumi_azure_native as azure_native\n\n        sql_resource_sql_container = azure_native.documentdb.SqlResourceSqlContainer("sqlResourceSqlContainer",\n            account_name="ddb1",\n            container_name="containerName",\n            database_name="databaseName",\n            location="West US",\n            options=azure_native.documentdb.CreateUpdateOptionsArgs(),\n            resource=azure_native.documentdb.SqlContainerResourceArgs(\n                conflict_resolution_policy=azure_native.documentdb.ConflictResolutionPolicyArgs(\n                    conflict_resolution_path="/path",\n                    mode="LastWriterWins",\n                ),\n                default_ttl=100,\n                id="containerName",\n                indexing_policy=azure_native.documentdb.IndexingPolicyArgs(\n                    automatic=True,\n                    excluded_paths=[],\n                    included_paths=[azure_native.documentdb.IncludedPathArgs(\n                        indexes=[\n                            azure_native.documentdb.IndexesArgs(\n                                data_type="String",\n                                kind="Range",\n                                precision=-1,\n                            ),\n                            azure_native.documentdb.IndexesArgs(\n                                data_type="Number",\n                                kind="Range",\n                                precision=-1,\n                            ),\n                        ],\n                        path="/*",\n                    )],\n                    indexing_mode="consistent",\n                ),\n                partition_key=azure_native.documentdb.ContainerPartitionKeyArgs(\n                    kind="Hash",\n                    paths=["/AccountNumber"],\n                ),\n                unique_key_policy=azure_native.documentdb.UniqueKeyPolicyArgs(\n                    unique_keys=[azure_native.documentdb.UniqueKeyArgs(\n                        paths=["/testPath"],\n                    )],\n                ),\n            ),\n            resource_group_name="rg1",\n            tags={})\n\n        ```\n\n        ## Import\n\n        An existing resource can be imported using its type token, name, and identifier, e.g.\n\n        ```sh\n        $ pulumi import azure-native:documentdb:SqlResourceSqlContainer containerName /subscriptions/subid/resourceGroups/rg1/providers/Microsoft.DocumentDB/databaseAccounts/ddb1/sqlDatabases/databaseName/sqlContainers/containerName \n        ```\n\n        :param str resource_name: The name of the resource.\n        :param SqlResourceSqlContainerArgs args: The arguments to use to populate this resource\'s properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        '
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        (resource_args, opts) = _utilities.get_resource_args_opts(SqlResourceSqlContainerArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, __props__=None):
        if False:
            return 10
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is None:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = SqlResourceSqlContainerArgs.__new__(SqlResourceSqlContainerArgs)
            __props__.__dict__['resource'] = None
        super(SqlResourceSqlContainer, __self__).__init__('azure-native:documentdb:SqlResourceSqlContainer', resource_name, __props__, opts)

    @staticmethod
    def get(resource_name: str, id: pulumi.Input[str], opts: Optional[pulumi.ResourceOptions]=None) -> 'SqlResourceSqlContainer':
        if False:
            print('Hello World!')
        "\n        Get an existing SqlResourceSqlContainer resource's state with the given name, id, and optional extra\n        properties used to qualify the lookup.\n\n        :param str resource_name: The unique name of the resulting resource.\n        :param pulumi.Input[str] id: The unique provider ID of the resource to lookup.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        opts = pulumi.ResourceOptions.merge(opts, pulumi.ResourceOptions(id=id))
        __props__ = SqlResourceSqlContainerArgs.__new__(SqlResourceSqlContainerArgs)
        __props__.__dict__['resource'] = None
        return SqlResourceSqlContainer(resource_name, opts=opts, __props__=__props__)

    @property
    @pulumi.getter
    def resource(self) -> pulumi.Output[Optional['outputs.SqlContainerGetPropertiesResponseResource']]:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'resource')
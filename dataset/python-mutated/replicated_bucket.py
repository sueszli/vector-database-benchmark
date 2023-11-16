import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
from . import gcp as _gcp
import pulumi_google_native
__all__ = ['ReplicatedBucketArgs', 'ReplicatedBucket']

@pulumi.input_type
class ReplicatedBucketArgs:

    def __init__(__self__, *, destination_region: pulumi.Input[str]):
        if False:
            print('Hello World!')
        '\n        The set of arguments for constructing a ReplicatedBucket resource.\n        :param pulumi.Input[str] destination_region: Region to which data should be replicated.\n        '
        pulumi.set(__self__, 'destination_region', destination_region)

    @property
    @pulumi.getter(name='destinationRegion')
    def destination_region(self) -> pulumi.Input[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Region to which data should be replicated.\n        '
        return pulumi.get(self, 'destination_region')

    @destination_region.setter
    def destination_region(self, value: pulumi.Input[str]):
        if False:
            i = 10
            return i + 15
        pulumi.set(self, 'destination_region', value)

class ReplicatedBucket(pulumi.ComponentResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, destination_region: Optional[pulumi.Input[str]]=None, __props__=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a ReplicatedBucket resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        :param pulumi.Input[str] destination_region: Region to which data should be replicated.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: ReplicatedBucketArgs, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            i = 10
            return i + 15
        "\n        Create a ReplicatedBucket resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param ReplicatedBucketArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        (resource_args, opts) = _utilities.get_resource_args_opts(ReplicatedBucketArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, destination_region: Optional[pulumi.Input[str]]=None, __props__=None):
        if False:
            print('Hello World!')
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is not None:
            raise ValueError('ComponentResource classes do not support opts.id')
        else:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = ReplicatedBucketArgs.__new__(ReplicatedBucketArgs)
            if destination_region is None and (not opts.urn):
                raise TypeError("Missing required property 'destination_region'")
            __props__.__dict__['destination_region'] = destination_region
            __props__.__dict__['location_policy'] = None
        super(ReplicatedBucket, __self__).__init__('example:index:ReplicatedBucket', resource_name, __props__, opts, remote=True)

    @property
    @pulumi.getter(name='locationPolicy')
    def location_policy(self) -> pulumi.Output[Optional['_gcp.gke.outputs.NodePoolAutoscaling']]:
        if False:
            print('Hello World!')
        '\n        test stuff\n        '
        return pulumi.get(self, 'location_policy')
import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from . import _utilities
from ._inputs import *
import pulumi_aws
__all__ = ['StaticPageArgs', 'StaticPage']

@pulumi.input_type
class StaticPageArgs:

    def __init__(__self__, *, index_content: pulumi.Input[str], foo: Optional['FooArgs']=None):
        if False:
            while True:
                i = 10
        '\n        The set of arguments for constructing a StaticPage resource.\n        :param pulumi.Input[str] index_content: The HTML content for index.html.\n        '
        pulumi.set(__self__, 'index_content', index_content)
        if foo is not None:
            pulumi.set(__self__, 'foo', foo)

    @property
    @pulumi.getter(name='indexContent')
    def index_content(self) -> pulumi.Input[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The HTML content for index.html.\n        '
        return pulumi.get(self, 'index_content')

    @index_content.setter
    def index_content(self, value: pulumi.Input[str]):
        if False:
            while True:
                i = 10
        pulumi.set(self, 'index_content', value)

    @property
    @pulumi.getter
    def foo(self) -> Optional['FooArgs']:
        if False:
            while True:
                i = 10
        return pulumi.get(self, 'foo')

    @foo.setter
    def foo(self, value: Optional['FooArgs']):
        if False:
            return 10
        pulumi.set(self, 'foo', value)

class StaticPage(pulumi.ComponentResource):

    @overload
    def __init__(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, foo: Optional[pulumi.InputType['FooArgs']]=None, index_content: Optional[pulumi.Input[str]]=None, __props__=None):
        if False:
            return 10
        '\n        Create a StaticPage resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        :param pulumi.Input[str] index_content: The HTML content for index.html.\n        '
        ...

    @overload
    def __init__(__self__, resource_name: str, args: StaticPageArgs, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            print('Hello World!')
        "\n        Create a StaticPage resource with the given unique name, props, and options.\n        :param str resource_name: The name of the resource.\n        :param StaticPageArgs args: The arguments to use to populate this resource's properties.\n        :param pulumi.ResourceOptions opts: Options for the resource.\n        "
        ...

    def __init__(__self__, resource_name: str, *args, **kwargs):
        if False:
            return 10
        (resource_args, opts) = _utilities.get_resource_args_opts(StaticPageArgs, pulumi.ResourceOptions, *args, **kwargs)
        if resource_args is not None:
            __self__._internal_init(resource_name, opts, **resource_args.__dict__)
        else:
            __self__._internal_init(resource_name, *args, **kwargs)

    def _internal_init(__self__, resource_name: str, opts: Optional[pulumi.ResourceOptions]=None, foo: Optional[pulumi.InputType['FooArgs']]=None, index_content: Optional[pulumi.Input[str]]=None, __props__=None):
        if False:
            return 10
        opts = pulumi.ResourceOptions.merge(_utilities.get_resource_opts_defaults(), opts)
        if not isinstance(opts, pulumi.ResourceOptions):
            raise TypeError('Expected resource options to be a ResourceOptions instance')
        if opts.id is not None:
            raise ValueError('ComponentResource classes do not support opts.id')
        else:
            if __props__ is not None:
                raise TypeError('__props__ is only valid when passed in combination with a valid opts.id to get an existing resource')
            __props__ = StaticPageArgs.__new__(StaticPageArgs)
            __props__.__dict__['foo'] = foo
            if index_content is None and (not opts.urn):
                raise TypeError("Missing required property 'index_content'")
            __props__.__dict__['index_content'] = index_content
            __props__.__dict__['bucket'] = None
            __props__.__dict__['website_url'] = None
        super(StaticPage, __self__).__init__('xyz:index:StaticPage', resource_name, __props__, opts, remote=True)

    @property
    @pulumi.getter
    def bucket(self) -> pulumi.Output['pulumi_aws.s3.Bucket']:
        if False:
            print('Hello World!')
        '\n        The bucket resource.\n        '
        return pulumi.get(self, 'bucket')

    @property
    @pulumi.getter(name='websiteUrl')
    def website_url(self) -> pulumi.Output[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The website URL.\n        '
        return pulumi.get(self, 'website_url')
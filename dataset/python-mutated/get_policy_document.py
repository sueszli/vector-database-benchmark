import copy
import warnings
import pulumi
import pulumi.runtime
from typing import Any, Mapping, Optional, Sequence, Union, overload
from ... import _utilities
from ... import x as _x
__all__ = ['GetPolicyDocumentResult', 'AwaitableGetPolicyDocumentResult', 'get_policy_document', 'get_policy_document_output']

@pulumi.output_type
class GetPolicyDocumentResult:

    def __init__(__self__, id=None, json=None, statements=None):
        if False:
            return 10
        if id and (not isinstance(id, str)):
            raise TypeError("Expected argument 'id' to be a str")
        pulumi.set(__self__, 'id', id)
        if json and (not isinstance(json, str)):
            raise TypeError("Expected argument 'json' to be a str")
        pulumi.set(__self__, 'json', json)
        if statements and (not isinstance(statements, list)):
            raise TypeError("Expected argument 'statements' to be a list")
        pulumi.set(__self__, 'statements', statements)

    @property
    @pulumi.getter
    def id(self) -> str:
        if False:
            print('Hello World!')
        return pulumi.get(self, 'id')

    @property
    @pulumi.getter
    def json(self) -> str:
        if False:
            return 10
        return pulumi.get(self, 'json')

    @property
    @pulumi.getter
    def statements(self) -> Optional[Sequence['_x.outputs.GetPolicyDocumentStatementResult']]:
        if False:
            i = 10
            return i + 15
        return pulumi.get(self, 'statements')

class AwaitableGetPolicyDocumentResult(GetPolicyDocumentResult):

    def __await__(self):
        if False:
            while True:
                i = 10
        if False:
            yield self
        return GetPolicyDocumentResult(id=self.id, json=self.json, statements=self.statements)

def get_policy_document(statements: Optional[Sequence[pulumi.InputType['_x.GetPolicyDocumentStatementArgs']]]=None, opts: Optional[pulumi.InvokeOptions]=None) -> AwaitableGetPolicyDocumentResult:
    if False:
        return 10
    '\n    Use this data source to access information about an existing resource.\n    '
    __args__ = dict()
    __args__['statements'] = statements
    opts = pulumi.InvokeOptions.merge(_utilities.get_invoke_opts_defaults(), opts)
    __ret__ = pulumi.runtime.invoke('aws:x/iam/getPolicyDocument:getPolicyDocument', __args__, opts=opts, typ=GetPolicyDocumentResult).value
    return AwaitableGetPolicyDocumentResult(id=pulumi.get(__ret__, 'id'), json=pulumi.get(__ret__, 'json'), statements=pulumi.get(__ret__, 'statements'))

@_utilities.lift_output_func(get_policy_document)
def get_policy_document_output(statements: Optional[pulumi.Input[Optional[Sequence[pulumi.InputType['_x.GetPolicyDocumentStatementArgs']]]]]=None, opts: Optional[pulumi.InvokeOptions]=None) -> pulumi.Output[GetPolicyDocumentResult]:
    if False:
        while True:
            i = 10
    '\n    Use this data source to access information about an existing resource.\n    '
    ...
from asyncio import ensure_future
from typing import Optional, Any, List, Callable
from copy import deepcopy
from .output import Output, Input
from .resource import CustomResource, ResourceOptions

class StackReferenceOutputDetails:
    """
    Records the output of a StackReference.
    At most one of the value and secret_value fields will be set.
    """
    value = Optional[Any]
    '\n    Output value returned by the StackReference.\n    None if the value is a secret or if it does not exist.\n    '
    secret_value = Optional[Any]
    '\n    Secret value returned by the StackReference.\n    None if the value is not a secret or if it does not exist.\n    '

    def __init__(self, value: Optional[Any]=None, secret_value: Optional[Any]=None) -> None:
        if False:
            print('Hello World!')
        '\n        :param Optional[Any] value:\n            Non-secret output value, if any.\n        :param Optional[Any] secret_value:\n            Secret output value, if any.\n        '
        self.value = value
        self.secret_value = secret_value

class StackReference(CustomResource):
    """
    Manages a reference to a Pulumi stack. The referenced stack's outputs are available via its "outputs" property or
    the "output" method.
    """
    name: Output[str]
    '\n    The name of the referenced stack.\n    '
    outputs: Output[dict]
    '\n    The outputs of the referenced stack.\n    '
    secret_output_names: Output[List[str]]
    '\n    The names of any stack outputs which contain secrets.\n    '

    def __init__(self, name: str, stack_name: Optional[str]=None, opts: Optional[ResourceOptions]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param str name: The unique name of the stack reference.\n        :param Optional[str] stack_name: The name of the stack to reference. If not provided, defaults to the name of\n               this resource.\n        :param Optional[ResourceOptions] opts: An optional set of resource options for this resource.\n        '
        target_stack = stack_name if stack_name is not None else name
        opts = ResourceOptions.merge(opts, ResourceOptions(id=target_stack))
        super().__init__('pulumi:pulumi:StackReference', name, {'name': target_stack, 'outputs': None, 'secret_output_names': None}, opts)

    def get_output(self, name: Input[str]) -> Output[Any]:
        if False:
            while True:
                i = 10
        '\n        Fetches the value of the named stack output, or None if the stack output was not found.\n\n        :param Input[str] name: The name of the stack output to fetch.\n        '
        value: Output[Any] = Output.all(Output.from_input(name), self.outputs).apply(lambda l: l[1].get(l[0]))
        is_secret = ensure_future(self.__is_secret_name(name))
        return Output(value.resources(), value.future(), value.is_known(), is_secret)

    def require_output(self, name: Input[str]) -> Output[Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetches the value of the named stack output, or raises a KeyError if the output was not\n        found.\n\n        :param Input[str] name: The name of the stack output to fetch.\n        '
        value = Output.all(Output.from_input(name), self.outputs).apply(lambda l: l[1][l[0]])
        is_secret = ensure_future(self.__is_secret_name(name))
        return Output(value.resources(), value.future(), value.is_known(), is_secret)

    async def get_output_details(self, name: str) -> StackReferenceOutputDetails:
        """
        Fetches the value of the named stack output
        and builds a StackReferenceOutputDetails object from it.

        The returned object has its `value` or `secret_value` fields set
        depending on whether the output is a secret.
        Neither field is set if the output was not found.
        """
        is_secret = await ensure_future(self.__is_secret_name(name))
        output_val = self.outputs.apply(lambda os: os[name])
        if not await output_val.is_known():
            return StackReferenceOutputDetails()
        value = await output_val.future()
        if is_secret:
            return StackReferenceOutputDetails(secret_value=value)
        return StackReferenceOutputDetails(value=value)

    def translate_output_property(self, prop: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Provides subclasses of Resource an opportunity to translate names of output properties\n        into a format of their choosing before writing those properties to the resource object.\n\n        :param str prop: A property name.\n        :return: A potentially transformed property name.\n        :rtype: str\n        '
        return 'secret_output_names' if prop == 'secretOutputNames' else prop

    async def __is_secret_name(self, name: Input[str]) -> bool:
        if not (await Output.from_input(name).is_known() and await self.secret_output_names.is_known()):
            return await self.outputs.is_secret()
        names = await self.secret_output_names.future()
        if names is None:
            return await self.outputs.is_secret()
        return await Output.from_input(name).future() in names
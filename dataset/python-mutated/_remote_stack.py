from typing import List, Optional
from pulumi.automation._cmd import OnOutput
from pulumi.automation._output import OutputMap
from pulumi.automation._stack import DestroyResult, OnEvent, PreviewResult, RefreshResult, Stack, UpResult, UpdateSummary
from pulumi.automation._workspace import Deployment

class RemoteStack:
    """
    RemoteStack is an isolated, independencly configurable instance of a Pulumi program that is
    operated on remotely (up/preview/refresh/destroy).
    """
    __stack: Stack

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        return self.__stack.name

    def __init__(self, stack: Stack):
        if False:
            print('Hello World!')
        self.__stack = stack

    def up(self, on_output: Optional[OnOutput]=None, on_event: Optional[OnEvent]=None) -> UpResult:
        if False:
            print('Hello World!')
        '\n        Creates or updates the resources in a stack by executing the program in the Workspace.\n        https://www.pulumi.com/docs/cli/commands/pulumi_up/\n\n        :param on_output: A function to process the stdout stream.\n        :param on_event: A function to process structured events from the Pulumi event stream.\n        :returns: UpResult\n        '
        return self.__stack.up(on_output=on_output, on_event=on_event)

    def preview(self, on_output: Optional[OnOutput]=None, on_event: Optional[OnEvent]=None) -> PreviewResult:
        if False:
            while True:
                i = 10
        '\n        Performs a dry-run update to a stack, returning pending changes.\n        https://www.pulumi.com/docs/cli/commands/pulumi_preview/\n\n        :param on_output: A function to process the stdout stream.\n        :param on_event: A function to process structured events from the Pulumi event stream.\n        :returns: PreviewResult\n        '
        return self.__stack.preview(on_output=on_output, on_event=on_event)

    def refresh(self, on_output: Optional[OnOutput]=None, on_event: Optional[OnEvent]=None) -> RefreshResult:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compares the current stack’s resource state with the state known to exist in the actual\n        cloud provider. Any such changes are adopted into the current stack.\n\n        :param on_output: A function to process the stdout stream.\n        :param on_event: A function to process structured events from the Pulumi event stream.\n        :returns: RefreshResult\n        '
        return self.__stack.refresh(on_output=on_output, on_event=on_event)

    def destroy(self, on_output: Optional[OnOutput]=None, on_event: Optional[OnEvent]=None) -> DestroyResult:
        if False:
            print('Hello World!')
        '\n        Destroy deletes all resources in a stack, leaving all history and configuration intact.\n\n        :param on_output: A function to process the stdout stream.\n        :param on_event: A function to process structured events from the Pulumi event stream.\n        :returns: DestroyResult\n        '
        return self.__stack.destroy(on_output=on_output, on_event=on_event)

    def outputs(self) -> OutputMap:
        if False:
            print('Hello World!')
        '\n        Gets the current set of Stack outputs from the last Stack.up().\n\n        :returns: OutputMap\n        '
        return self.__stack.outputs()

    def history(self, page_size: Optional[int]=None, page: Optional[int]=None) -> List[UpdateSummary]:
        if False:
            return 10
        '\n        Returns a list summarizing all previous and current results from Stack lifecycle operations\n        (up/preview/refresh/destroy).\n\n        :param page_size: Paginate history entries (used in combination with page), defaults to all.\n        :param page: Paginate history entries (used in combination with page_size), defaults to all.\n        :param show_secrets: Show config secrets when they appear in history.\n\n        :returns: List[UpdateSummary]\n        '
        return self.__stack.history(page_size=page_size, page=page, show_secrets=False)

    def cancel(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Cancel stops a stack's currently running update. It returns an error if no update is currently running.\n        Note that this operation is _very dangerous_, and may leave the stack in an inconsistent state\n        if a resource operation was pending when the update was canceled.\n        This command is not supported for local backends.\n        "
        self.__stack.cancel()

    def export_stack(self) -> Deployment:
        if False:
            print('Hello World!')
        "\n        export_stack exports the deployment state of the stack.\n        This can be combined with Stack.import_state to edit a stack's state (such as recovery from failed deployments).\n\n        :returns: Deployment\n        "
        return self.__stack.export_stack()

    def import_stack(self, state: Deployment) -> None:
        if False:
            return 10
        "\n        import_stack imports the specified deployment state into a pre-existing stack.\n        This can be combined with Stack.export_state to edit a stack's state (such as recovery from failed deployments).\n\n        :param state: The deployment state to import.\n        "
        self.__stack.import_stack(state=state)
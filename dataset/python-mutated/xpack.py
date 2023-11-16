import typing as t
from elastic_transport import ObjectApiResponse
from ._base import NamespacedClient
from .utils import _rewrite_parameters

class XPackClient(NamespacedClient):

    def __getattr__(self, attr_name: str) -> t.Any:
        if False:
            print('Hello World!')
        return getattr(self.client, attr_name)

    @_rewrite_parameters()
    async def info(self, *, accept_enterprise: t.Optional[bool]=None, categories: t.Optional[t.Sequence[str]]=None, error_trace: t.Optional[bool]=None, filter_path: t.Optional[t.Union[str, t.Sequence[str]]]=None, human: t.Optional[bool]=None, pretty: t.Optional[bool]=None) -> ObjectApiResponse[t.Any]:
        """
        Retrieves information about the installed X-Pack features.

        `<https://www.elastic.co/guide/en/elasticsearch/reference/master/info-api.html>`_

        :param accept_enterprise: If this param is used it must be set to true
        :param categories: A comma-separated list of the information categories to include
            in the response. For example, `build,license,features`.
        """
        __path = '/_xpack'
        __query: t.Dict[str, t.Any] = {}
        if accept_enterprise is not None:
            __query['accept_enterprise'] = accept_enterprise
        if categories is not None:
            __query['categories'] = categories
        if error_trace is not None:
            __query['error_trace'] = error_trace
        if filter_path is not None:
            __query['filter_path'] = filter_path
        if human is not None:
            __query['human'] = human
        if pretty is not None:
            __query['pretty'] = pretty
        __headers = {'accept': 'application/json'}
        return await self.perform_request('GET', __path, params=__query, headers=__headers)

    @_rewrite_parameters()
    async def usage(self, *, error_trace: t.Optional[bool]=None, filter_path: t.Optional[t.Union[str, t.Sequence[str]]]=None, human: t.Optional[bool]=None, master_timeout: t.Optional[t.Union['t.Literal[-1]', 't.Literal[0]', str]]=None, pretty: t.Optional[bool]=None) -> ObjectApiResponse[t.Any]:
        """
        Retrieves usage information about the installed X-Pack features.

        `<https://www.elastic.co/guide/en/elasticsearch/reference/master/usage-api.html>`_

        :param master_timeout: Period to wait for a connection to the master node. If
            no response is received before the timeout expires, the request fails and
            returns an error.
        """
        __path = '/_xpack/usage'
        __query: t.Dict[str, t.Any] = {}
        if error_trace is not None:
            __query['error_trace'] = error_trace
        if filter_path is not None:
            __query['filter_path'] = filter_path
        if human is not None:
            __query['human'] = human
        if master_timeout is not None:
            __query['master_timeout'] = master_timeout
        if pretty is not None:
            __query['pretty'] = pretty
        __headers = {'accept': 'application/json'}
        return await self.perform_request('GET', __path, params=__query, headers=__headers)
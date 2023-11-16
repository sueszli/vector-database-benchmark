import re
from collections import defaultdict
from typing import Any, AnyStr, Dict, List, Mapping, Match, NamedTuple, Optional, Tuple
from urllib.parse import parse_qs, unquote
from botocore.model import OperationModel, ServiceModel, StructureShape
from werkzeug.datastructures import Headers, MultiDict
from werkzeug.exceptions import MethodNotAllowed, NotFound
from werkzeug.routing import Map, MapAdapter, PathConverter, Rule
from localstack.http import Request
from localstack.http.request import get_raw_path

class GreedyPathConverter(PathConverter):
    """
    This converter makes sure that the path ``/mybucket//mykey`` can be matched to the pattern
    ``<Bucket>/<path:Key>`` and will result in `Key` being `/mykey`.
    """
    regex = '.*?'
    part_isolating = False
    'From the werkzeug docs: If a custom converter can match a forward slash, /, it should have the\n    attribute part_isolating set to False. This will ensure that rules using the custom converter are\n    correctly matched.'

class _HttpOperation(NamedTuple):
    """Useful intermediary representation of the 'http' block of an operation to make code cleaner"""
    operation: OperationModel
    path: str
    method: str
    query_args: Mapping[str, List[str]]
    header_args: List[str]
    deprecated: bool

    @staticmethod
    def from_operation(op: OperationModel) -> '_HttpOperation':
        if False:
            i = 10
            return i + 15
        if (auth_path := op.http.get('authPath')):
            (path, sep, query) = op.http.get('requestUri', '').partition('?')
            uri = f"{auth_path.rstrip('/')}{sep}{query}"
        else:
            uri = op.http.get('requestUri')
        method = op.http.get('method')
        deprecated = op.deprecated
        path_query = uri.split('?')
        path = path_query[0]
        header_args = []
        query_args: Dict[str, List[str]] = {}
        if len(path_query) > 1:
            query_args: Dict[str, List[str]] = parse_qs(path_query[1], keep_blank_values=True)
            query_args = {k: filter(None, v) for (k, v) in query_args.items()}
        input_shape = op.input_shape
        if isinstance(input_shape, StructureShape):
            for required_member in input_shape.required_members:
                member_shape = input_shape.members[required_member]
                location = member_shape.serialization.get('location')
                if location is not None:
                    if location == 'header':
                        header_name = member_shape.serialization.get('name')
                        header_args.append(header_name)
                    elif location == 'querystring':
                        query_name = member_shape.serialization.get('name')
                        if query_name not in query_args:
                            query_args[query_name] = []
        return _HttpOperation(op, path, method, query_args, header_args, deprecated)

class _RequiredArgsRule:
    """
    Specific Rule implementation which checks if a set of certain required header and query parameters are matched by
    a specific request.
    """
    endpoint: Any
    required_query_args: Optional[Mapping[str, List[Any]]]
    required_header_args: List[str]
    match_score: int

    def __init__(self, operation: _HttpOperation) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.endpoint = operation.operation
        self.required_query_args = operation.query_args or {}
        self.required_header_args = operation.header_args or []
        self.match_score = 10 + 10 * len(self.required_query_args) + 10 * len(self.required_header_args)
        if operation.deprecated:
            self.match_score -= 5

    def matches(self, query_args: MultiDict, headers: Headers) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns true if the given query args and the given headers of a request match the required query args and\n        headers of this rule.\n        :param query_args: query arguments of the incoming request\n        :param headers: headers of the incoming request\n        :return: True if the query args and headers match the required args of this rule\n        '
        if self.required_query_args:
            for (key, values) in self.required_query_args.items():
                if key not in query_args:
                    return False
                if values:
                    query_arg_values = query_args.getlist(key)
                    for value in values:
                        if value not in query_arg_values:
                            return False
        if self.required_header_args:
            for key in self.required_header_args:
                if key not in headers:
                    return False
        return True

class _StrictMethodRule(Rule):
    """
    Small extension to Werkzeug's Rule class which reverts unwanted assumptions made by Werkzeug.
    Reverted assumptions:
    - Werkzeug automatically matches HEAD requests to the corresponding GET request (i.e. Werkzeug's rule automatically
      adds the HEAD HTTP method to a rule which should only match GET requests). This is implemented to simplify
      implementing an app compliant with HTTP (where a HEAD request needs to return the headers of a corresponding GET
      request), but it is unwanted for our strict rule matching in here.
    """

    def __init__(self, string: str, method: str, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(string=string, methods=[method], **kwargs)
        self.methods = {method.upper()}

class _RequestMatchingRule(_StrictMethodRule):
    """
    A Werkzeug Rule extension which initially acts as a normal rule (i.e. matches a path and method).

    This rule matches if one of its sub-rules _might_ match.
    It cannot be assumed that one of the fine-grained rules matches, just because this rule initially matches.
    If this rule matches, the caller _must_ call `match_request` in order to find the actual fine-grained matching rule.
    The result of `match_request` is only meaningful if this wrapping rule also matches.
    """

    def __init__(self, string: str, operations: List[_HttpOperation], method: str, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(string=string, method=method, **kwargs)
        rules = [_RequiredArgsRule(op) for op in operations]
        self.rules = sorted(rules, key=lambda rule: rule.match_score, reverse=True)

    def match_request(self, request: Request) -> _RequiredArgsRule:
        if False:
            print('Hello World!')
        "\n        Function which needs to be called by a caller if the _RequestMatchingRule already matched using Werkzeug's\n        default matching mechanism.\n\n        :param request: to perform the fine-grained matching on\n        :return: matching fine-grained rule\n        :raises: NotFound if none of the fine-grained rules matches\n        "
        for rule in self.rules:
            if rule.matches(request.args, request.headers):
                return rule
        raise NotFound()
_path_param_regex = re.compile('({.+?})')
_rule_replacements = {'-': '_0_'}
_rule_replacement_table = str.maketrans(_rule_replacements)

def _transform_path_params_to_rule_vars(match: Match[AnyStr]) -> str:
    if False:
        while True:
            i = 10
    '\n    Transforms a request URI path param to a valid Werkzeug Rule string variable placeholder.\n    This transformation function should be used in combination with _path_param_regex on the request URIs (without any\n    query params).\n\n    :param match: Regex match which contains a single group. The match group is a request URI path param, including the\n                    surrounding curly braces.\n    :return: Werkzeug rule string variable placeholder which is semantically equal to the given request URI path param\n\n    '
    request_uri_variable: str = match.group(0)[1:-1]
    greedy_prefix = ''
    if request_uri_variable.endswith('+'):
        greedy_prefix = 'path:'
        request_uri_variable = request_uri_variable.strip('+')
    escaped_request_uri_variable = request_uri_variable.translate(_rule_replacement_table)
    return f'<{greedy_prefix}{escaped_request_uri_variable}>'

def _post_process_arg_name(arg_key: str) -> str:
    if False:
        print('Hello World!')
    '\n    Reverses previous manipulations to the path parameters names (like replacing forbidden characters with\n    placeholders).\n    :param arg_key: Path param key name extracted using Werkzeug rules\n    :return: Post-processed ("un-sanitized") path param key\n    '
    result = arg_key
    for (original, substitution) in _rule_replacements.items():
        result = result.replace(substitution, original)
    return result

def _create_service_map(service: ServiceModel) -> Map:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a Werkzeug Map object with all rules necessary for the specific service.\n    :param service: botocore service model to create the rules for\n    :return: a Map instance which is used to perform the in-service operation routing\n    '
    ops = [service.operation_model(op_name) for op_name in service.operation_names]
    rules = []
    path_index: Dict[(str, str), List[_HttpOperation]] = defaultdict(list)
    for op in ops:
        http_op = _HttpOperation.from_operation(op)
        path_index[http_op.path, http_op.method].append(http_op)
    for ((path, method), ops) in path_index.items():
        rule_string = _path_param_regex.sub(_transform_path_params_to_rule_vars, path)
        if len(ops) == 1:
            op = ops[0]
            rules.append(_StrictMethodRule(string=rule_string, method=method, endpoint=op.operation))
        else:
            rules.append(_RequestMatchingRule(string=rule_string, method=method, operations=ops))
    return Map(rules=rules, strict_slashes=False, merge_slashes=False, converters={'path': GreedyPathConverter})

class RestServiceOperationRouter:
    """
    A router implementation which abstracts the (quite complex) routing of incoming HTTP requests to a specific
    operation within a "REST" service (rest-xml, rest-json).
    """
    _map: Map

    def __init__(self, service: ServiceModel):
        if False:
            return 10
        self._map = _create_service_map(service)

    def match(self, request: Request) -> Tuple[OperationModel, Mapping[str, Any]]:
        if False:
            i = 10
            return i + 15
        "\n        Matches the given request to the operation it targets (or raises an exception if no operation matches).\n\n        :param request: The request of which the targeting operation needs to be found\n        :return: A tuple with the matched operation and the (already parsed) path params\n        :raises: Werkzeug's NotFound exception in case the given request does not match any operation\n        "
        matcher: MapAdapter = self._map.bind(request.host)
        try:
            method = request.method if request.method != 'OPTIONS' else 'GET'
            path = get_raw_path(request)
            path = path.rstrip('/')
            (rule, args) = matcher.match(path, method=method, return_rule=True)
        except MethodNotAllowed as e:
            raise NotFound() from e
        if isinstance(rule, _RequestMatchingRule):
            rule = rule.match_request(request)
        args = {_post_process_arg_name(k): unquote(v) for (k, v) in args.items()}
        operation: OperationModel = rule.endpoint
        return (operation, args)
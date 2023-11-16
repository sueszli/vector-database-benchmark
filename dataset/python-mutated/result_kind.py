from pyflink.java_gateway import get_gateway
__all__ = ['ResultKind']

class ResultKind(object):
    """
    ResultKind defines the types of the result.

    :data:`SUCCESS`:

    The statement (e.g. DDL, USE) executes successfully, and the result only contains a simple "OK".

    :data:`SUCCESS_WITH_CONTENT`:

    The statement (e.g. DML, DQL, SHOW) executes successfully, and the result contains important
    content.

    .. versionadded:: 1.11.0
    """
    SUCCESS = 0
    SUCCESS_WITH_CONTENT = 1

    @staticmethod
    def _from_j_result_kind(j_result_kind):
        if False:
            return 10
        gateway = get_gateway()
        JResultKind = gateway.jvm.org.apache.flink.table.api.ResultKind
        if j_result_kind == JResultKind.SUCCESS:
            return ResultKind.SUCCESS
        elif j_result_kind == JResultKind.SUCCESS_WITH_CONTENT:
            return ResultKind.SUCCESS_WITH_CONTENT
        else:
            raise Exception('Unsupported Java result kind: %s' % j_result_kind)
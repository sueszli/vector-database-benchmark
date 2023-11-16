__all__ = ['JobID']

class JobID(object):
    """
    Unique (at least statistically unique) identifier for a Flink Job. Jobs in Flink correspond
    to dataflow graphs.

    Jobs act simultaneously as sessions, because jobs can be created and submitted incrementally
    in different parts. Newer fragments of a graph can be attached to existing graphs, thereby
    extending the current data flow graphs.

    .. versionadded:: 1.11.0
    """

    def __init__(self, j_job_id):
        if False:
            i = 10
            return i + 15
        self._j_job_id = j_job_id

    def __str__(self):
        if False:
            while True:
                i = 10
        return self._j_job_id.toString()
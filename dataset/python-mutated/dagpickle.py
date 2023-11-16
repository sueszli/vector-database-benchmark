from __future__ import annotations
from typing import TYPE_CHECKING
import dill
from sqlalchemy import BigInteger, Column, Integer, PickleType
from airflow.models.base import Base
from airflow.utils import timezone
from airflow.utils.sqlalchemy import UtcDateTime
if TYPE_CHECKING:
    from airflow.models.dag import DAG

class DagPickle(Base):
    """
    Represents a version of a DAG and becomes a source of truth for a BackfillJob execution.

    Dags can originate from different places (user repos, main repo, ...) and also get executed
    in different places (different executors).  A pickle is a native python serialized object,
    and in this case gets stored in the database for the duration of the job.

    The executors pick up the DagPickle id and read the dag definition from the database.
    """
    id = Column(Integer, primary_key=True)
    pickle = Column(PickleType(pickler=dill))
    created_dttm = Column(UtcDateTime, default=timezone.utcnow)
    pickle_hash = Column(BigInteger)
    __tablename__ = 'dag_pickle'

    def __init__(self, dag: DAG) -> None:
        if False:
            i = 10
            return i + 15
        self.dag_id = dag.dag_id
        if hasattr(dag, 'template_env'):
            dag.template_env = None
        self.pickle_hash = hash(dag)
        self.pickle = dag
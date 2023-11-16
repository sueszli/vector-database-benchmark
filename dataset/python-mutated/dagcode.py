from __future__ import annotations
import logging
import os
import struct
from datetime import datetime
from typing import TYPE_CHECKING, Collection, Iterable
from sqlalchemy import BigInteger, Column, String, Text, delete, select
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.sql.expression import literal
from airflow.exceptions import AirflowException, DagCodeNotFound
from airflow.models.base import Base
from airflow.utils import timezone
from airflow.utils.file import correct_maybe_zipped, open_maybe_zipped
from airflow.utils.session import NEW_SESSION, provide_session
from airflow.utils.sqlalchemy import UtcDateTime
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
log = logging.getLogger(__name__)

class DagCode(Base):
    """A table for DAGs code.

    dag_code table contains code of DAG files synchronized by scheduler.

    For details on dag serialization see SerializedDagModel
    """
    __tablename__ = 'dag_code'
    fileloc_hash = Column(BigInteger, nullable=False, primary_key=True, autoincrement=False)
    fileloc = Column(String(2000), nullable=False)
    last_updated = Column(UtcDateTime, nullable=False)
    source_code = Column(Text().with_variant(MEDIUMTEXT(), 'mysql'), nullable=False)

    def __init__(self, full_filepath: str, source_code: str | None=None):
        if False:
            for i in range(10):
                print('nop')
        self.fileloc = full_filepath
        self.fileloc_hash = DagCode.dag_fileloc_hash(self.fileloc)
        self.last_updated = timezone.utcnow()
        self.source_code = source_code or DagCode.code(self.fileloc)

    @provide_session
    def sync_to_db(self, session: Session=NEW_SESSION) -> None:
        if False:
            print('Hello World!')
        'Write code into database.\n\n        :param session: ORM Session\n        '
        self.bulk_sync_to_db([self.fileloc], session)

    @classmethod
    @provide_session
    def bulk_sync_to_db(cls, filelocs: Iterable[str], session: Session=NEW_SESSION) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write code in bulk into database.\n\n        :param filelocs: file paths of DAGs to sync\n        :param session: ORM Session\n        '
        filelocs = set(filelocs)
        filelocs_to_hashes = {fileloc: DagCode.dag_fileloc_hash(fileloc) for fileloc in filelocs}
        existing_orm_dag_codes = session.scalars(select(DagCode).filter(DagCode.fileloc_hash.in_(filelocs_to_hashes.values())).with_for_update(of=DagCode)).all()
        if existing_orm_dag_codes:
            existing_orm_dag_codes_map = {orm_dag_code.fileloc: orm_dag_code for orm_dag_code in existing_orm_dag_codes}
        else:
            existing_orm_dag_codes_map = {}
        existing_orm_dag_codes_by_fileloc_hashes = {orm.fileloc_hash: orm for orm in existing_orm_dag_codes}
        existing_orm_filelocs = {orm.fileloc for orm in existing_orm_dag_codes_by_fileloc_hashes.values()}
        if not existing_orm_filelocs.issubset(filelocs):
            conflicting_filelocs = existing_orm_filelocs.difference(filelocs)
            hashes_to_filelocs = {DagCode.dag_fileloc_hash(fileloc): fileloc for fileloc in filelocs}
            message = ''
            for fileloc in conflicting_filelocs:
                filename = hashes_to_filelocs[DagCode.dag_fileloc_hash(fileloc)]
                message += f"Filename '{filename}' causes a hash collision in the database with '{fileloc}'. Please rename the file."
            raise AirflowException(message)
        existing_filelocs = {dag_code.fileloc for dag_code in existing_orm_dag_codes}
        missing_filelocs = filelocs.difference(existing_filelocs)
        for fileloc in missing_filelocs:
            orm_dag_code = DagCode(fileloc, cls._get_code_from_file(fileloc))
            session.add(orm_dag_code)
        for fileloc in existing_filelocs:
            current_version = existing_orm_dag_codes_by_fileloc_hashes[filelocs_to_hashes[fileloc]]
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(correct_maybe_zipped(fileloc)), tz=timezone.utc)
            if file_mod_time > current_version.last_updated:
                orm_dag_code = existing_orm_dag_codes_map[fileloc]
                orm_dag_code.last_updated = file_mod_time
                orm_dag_code.source_code = cls._get_code_from_file(orm_dag_code.fileloc)
                session.merge(orm_dag_code)

    @classmethod
    @provide_session
    def remove_deleted_code(cls, alive_dag_filelocs: Collection[str], processor_subdir: str, session: Session=NEW_SESSION) -> None:
        if False:
            i = 10
            return i + 15
        'Delete code not included in alive_dag_filelocs.\n\n        :param alive_dag_filelocs: file paths of alive DAGs\n        :param processor_subdir: dag processor subdir\n        :param session: ORM Session\n        '
        alive_fileloc_hashes = [cls.dag_fileloc_hash(fileloc) for fileloc in alive_dag_filelocs]
        log.debug('Deleting code from %s table ', cls.__tablename__)
        session.execute(delete(cls).where(cls.fileloc_hash.notin_(alive_fileloc_hashes), cls.fileloc.notin_(alive_dag_filelocs), cls.fileloc.contains(processor_subdir)).execution_options(synchronize_session='fetch'))

    @classmethod
    @provide_session
    def has_dag(cls, fileloc: str, session: Session=NEW_SESSION) -> bool:
        if False:
            print('Hello World!')
        'Check a file exist in dag_code table.\n\n        :param fileloc: the file to check\n        :param session: ORM Session\n        '
        fileloc_hash = cls.dag_fileloc_hash(fileloc)
        return session.scalars(select(literal(True)).where(cls.fileloc_hash == fileloc_hash)).one_or_none() is not None

    @classmethod
    def get_code_by_fileloc(cls, fileloc: str) -> str:
        if False:
            i = 10
            return i + 15
        'Return source code for a given fileloc.\n\n        :param fileloc: file path of a DAG\n        :return: source code as string\n        '
        return cls.code(fileloc)

    @classmethod
    def code(cls, fileloc) -> str:
        if False:
            print('Hello World!')
        'Return source code for this DagCode object.\n\n        :return: source code as string\n        '
        return cls._get_code_from_db(fileloc)

    @staticmethod
    def _get_code_from_file(fileloc):
        if False:
            while True:
                i = 10
        with open_maybe_zipped(fileloc, 'r') as f:
            code = f.read()
        return code

    @classmethod
    @provide_session
    def _get_code_from_db(cls, fileloc, session: Session=NEW_SESSION) -> str:
        if False:
            i = 10
            return i + 15
        dag_code = session.scalar(select(cls).where(cls.fileloc_hash == cls.dag_fileloc_hash(fileloc)))
        if not dag_code:
            raise DagCodeNotFound()
        else:
            code = dag_code.source_code
        return code

    @staticmethod
    def dag_fileloc_hash(full_filepath: str) -> int:
        if False:
            return 10
        'Hashing file location for indexing.\n\n        :param full_filepath: full filepath of DAG file\n        :return: hashed full_filepath\n        '
        import hashlib
        return struct.unpack('>Q', hashlib.sha1(full_filepath.encode('utf-8')).digest()[-8:])[0] >> 8
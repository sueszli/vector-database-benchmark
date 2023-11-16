from typing import Any, Dict, List, Optional, Tuple, Union, cast
import attr
from synapse.api.constants import LoginType
from synapse.api.errors import StoreError
from synapse.storage._base import SQLBaseStore, db_to_json
from synapse.storage.database import LoggingTransaction
from synapse.types import JsonDict
from synapse.util import json_encoder, stringutils

@attr.s(slots=True, auto_attribs=True)
class UIAuthSessionData:
    session_id: str
    clientdict: JsonDict
    uri: str
    method: str
    description: str

class UIAuthWorkerStore(SQLBaseStore):
    """
    Manage user interactive authentication sessions.
    """

    async def create_ui_auth_session(self, clientdict: JsonDict, uri: str, method: str, description: str) -> UIAuthSessionData:
        """
        Creates a new user interactive authentication session.

        The session can be used to track the stages necessary to authenticate a
        user across multiple HTTP requests.

        Args:
            clientdict:
                The dictionary from the client root level, not the 'auth' key.
            uri:
                The URI this session was initiated with, this is checked at each
                stage of the authentication to ensure that the asked for
                operation has not changed.
            method:
                The method this session was initiated with, this is checked at each
                stage of the authentication to ensure that the asked for
                operation has not changed.
            description:
                A string description of the operation that the current
                authentication is authorising.
        Returns:
            The newly created session.
        Raises:
            StoreError if a unique session ID cannot be generated.
        """
        clientdict_json = json_encoder.encode(clientdict)
        attempts = 0
        while attempts < 5:
            session_id = stringutils.random_string(24)
            try:
                await self.db_pool.simple_insert(table='ui_auth_sessions', values={'session_id': session_id, 'clientdict': clientdict_json, 'uri': uri, 'method': method, 'description': description, 'serverdict': '{}', 'creation_time': self.hs.get_clock().time_msec()}, desc='create_ui_auth_session')
                return UIAuthSessionData(session_id, clientdict, uri, method, description)
            except self.db_pool.engine.module.IntegrityError:
                attempts += 1
        raise StoreError(500, "Couldn't generate a session ID.")

    async def get_ui_auth_session(self, session_id: str) -> UIAuthSessionData:
        """Retrieve a UI auth session.

        Args:
            session_id: The ID of the session.
        Returns:
            A dict containing the device information.
        Raises:
            StoreError if the session is not found.
        """
        result = await self.db_pool.simple_select_one(table='ui_auth_sessions', keyvalues={'session_id': session_id}, retcols=('clientdict', 'uri', 'method', 'description'), desc='get_ui_auth_session')
        return UIAuthSessionData(session_id, clientdict=db_to_json(result[0]), uri=result[1], method=result[2], description=result[3])

    async def mark_ui_auth_stage_complete(self, session_id: str, stage_type: str, result: Union[str, bool, JsonDict]) -> None:
        """
        Mark a session stage as completed.

        Args:
            session_id: The ID of the corresponding session.
            stage_type: The completed stage type.
            result: The result of the stage verification.
        Raises:
            StoreError if the session cannot be found.
        """
        try:
            await self.db_pool.simple_upsert(table='ui_auth_sessions_credentials', keyvalues={'session_id': session_id, 'stage_type': stage_type}, values={'result': json_encoder.encode(result)}, desc='mark_ui_auth_stage_complete')
        except self.db_pool.engine.module.IntegrityError:
            raise StoreError(400, 'Unknown session ID: %s' % (session_id,))

    async def get_completed_ui_auth_stages(self, session_id: str) -> Dict[str, Union[str, bool, JsonDict]]:
        """
        Retrieve the completed stages of a UI authentication session.

        Args:
            session_id: The ID of the session.
        Returns:
            The completed stages mapped to the result of the verification of
            that auth-type.
        """
        results = {}
        rows = cast(List[Tuple[str, str]], await self.db_pool.simple_select_list(table='ui_auth_sessions_credentials', keyvalues={'session_id': session_id}, retcols=('stage_type', 'result'), desc='get_completed_ui_auth_stages'))
        for (stage_type, result) in rows:
            results[stage_type] = db_to_json(result)
        return results

    async def set_ui_auth_clientdict(self, session_id: str, clientdict: JsonDict) -> None:
        """
        Store an updated clientdict for a given session ID.

        Args:
            session_id: The ID of this session as returned from check_auth
            clientdict:
                The dictionary from the client root level, not the 'auth' key.
        """
        clientdict_json = json_encoder.encode(clientdict)
        await self.db_pool.simple_update_one(table='ui_auth_sessions', keyvalues={'session_id': session_id}, updatevalues={'clientdict': clientdict_json}, desc='set_ui_auth_client_dict')

    async def set_ui_auth_session_data(self, session_id: str, key: str, value: Any) -> None:
        """
        Store a key-value pair into the sessions data associated with this
        request. This data is stored server-side and cannot be modified by
        the client.

        Args:
            session_id: The ID of this session as returned from check_auth
            key: The key to store the data under
            value: The data to store
        Raises:
            StoreError if the session cannot be found.
        """
        await self.db_pool.runInteraction('set_ui_auth_session_data', self._set_ui_auth_session_data_txn, session_id, key, value)

    def _set_ui_auth_session_data_txn(self, txn: LoggingTransaction, session_id: str, key: str, value: Any) -> None:
        if False:
            return 10
        result = self.db_pool.simple_select_one_onecol_txn(txn, table='ui_auth_sessions', keyvalues={'session_id': session_id}, retcol='serverdict')
        serverdict = db_to_json(result)
        serverdict[key] = value
        self.db_pool.simple_update_one_txn(txn, table='ui_auth_sessions', keyvalues={'session_id': session_id}, updatevalues={'serverdict': json_encoder.encode(serverdict)})

    async def get_ui_auth_session_data(self, session_id: str, key: str, default: Optional[Any]=None) -> Any:
        """
        Retrieve data stored with set_session_data

        Args:
            session_id: The ID of this session as returned from check_auth
            key: The key to store the data under
            default: Value to return if the key has not been set
        Raises:
            StoreError if the session cannot be found.
        """
        result = await self.db_pool.simple_select_one_onecol(table='ui_auth_sessions', keyvalues={'session_id': session_id}, retcol='serverdict', desc='get_ui_auth_session_data')
        serverdict = db_to_json(result)
        return serverdict.get(key, default)

    async def add_user_agent_ip_to_ui_auth_session(self, session_id: str, user_agent: str, ip: str) -> None:
        """Add the given user agent / IP to the tracking table"""
        await self.db_pool.simple_upsert(table='ui_auth_sessions_ips', keyvalues={'session_id': session_id, 'user_agent': user_agent, 'ip': ip}, values={}, desc='add_user_agent_ip_to_ui_auth_session')

    async def get_user_agents_ips_to_ui_auth_session(self, session_id: str) -> List[Tuple[str, str]]:
        """Get the given user agents / IPs used during the ui auth process

        Returns:
            List of user_agent/ip pairs
        """
        return cast(List[Tuple[str, str]], await self.db_pool.simple_select_list(table='ui_auth_sessions_ips', keyvalues={'session_id': session_id}, retcols=('user_agent', 'ip'), desc='get_user_agents_ips_to_ui_auth_session'))

    async def delete_old_ui_auth_sessions(self, expiration_time: int) -> None:
        """
        Remove sessions which were last used earlier than the expiration time.

        Args:
            expiration_time: The latest time that is still considered valid.
                This is an epoch time in milliseconds.

        """
        await self.db_pool.runInteraction('delete_old_ui_auth_sessions', self._delete_old_ui_auth_sessions_txn, expiration_time)

    def _delete_old_ui_auth_sessions_txn(self, txn: LoggingTransaction, expiration_time: int) -> None:
        if False:
            i = 10
            return i + 15
        sql = 'SELECT session_id FROM ui_auth_sessions WHERE creation_time <= ?'
        txn.execute(sql, [expiration_time])
        session_ids = [r[0] for r in txn.fetchall()]
        self.db_pool.simple_delete_many_txn(txn, table='ui_auth_sessions_ips', column='session_id', values=session_ids, keyvalues={})
        rows = cast(List[Tuple[str]], self.db_pool.simple_select_many_txn(txn, table='ui_auth_sessions_credentials', column='session_id', iterable=session_ids, keyvalues={'stage_type': LoginType.REGISTRATION_TOKEN}, retcols=['result']))
        token_counts: Dict[str, int] = {}
        for r in rows:
            token = db_to_json(r[0])
            if isinstance(token, str):
                token_counts[token] = token_counts.get(token, 0) + 1
        if len(token_counts) > 0:
            token_rows = cast(List[Tuple[str, int]], self.db_pool.simple_select_many_txn(txn, table='registration_tokens', column='token', iterable=list(token_counts.keys()), keyvalues={}, retcols=['token', 'pending']))
            for (token, pending) in token_rows:
                new_pending = pending - token_counts[token]
                self.db_pool.simple_update_one_txn(txn, table='registration_tokens', keyvalues={'token': token}, updatevalues={'pending': new_pending})
        self.db_pool.simple_delete_many_txn(txn, table='ui_auth_sessions_credentials', column='session_id', values=session_ids, keyvalues={})
        self.db_pool.simple_delete_many_txn(txn, table='ui_auth_sessions', column='session_id', values=session_ids, keyvalues={})

class UIAuthStore(UIAuthWorkerStore):
    pass
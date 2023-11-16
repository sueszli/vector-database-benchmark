# Copyright 2015, 2016 OpenMarket Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Type

from synapse.api.errors import CodeMessageException
from synapse.metrics.background_process_metrics import run_as_background_process
from synapse.storage import DataStore
from synapse.types import StrCollection
from synapse.util import Clock

if TYPE_CHECKING:
    from synapse.notifier import Notifier
    from synapse.replication.tcp.handler import ReplicationCommandHandler

logger = logging.getLogger(__name__)


class NotRetryingDestination(Exception):
    def __init__(self, retry_last_ts: int, retry_interval: int, destination: str):
        """Raised by the limiter (and federation client) to indicate that we are
        are deliberately not attempting to contact a given server.

        Args:
            retry_last_ts: the unix ts in milliseconds of our last attempt
                to contact the server.  0 indicates that the last attempt was
                successful or that we've never actually attempted to connect.
            retry_interval: the time in milliseconds to wait until the next
                attempt.
            destination: the domain in question
        """

        msg = f"Not retrying server {destination} because we tried it recently retry_last_ts={retry_last_ts} and we won't check for another retry_interval={retry_interval}ms."
        super().__init__(msg)

        self.retry_last_ts = retry_last_ts
        self.retry_interval = retry_interval
        self.destination = destination


async def get_retry_limiter(
    destination: str,
    clock: Clock,
    store: DataStore,
    ignore_backoff: bool = False,
    **kwargs: Any,
) -> "RetryDestinationLimiter":
    """For a given destination check if we have previously failed to
    send a request there and are waiting before retrying the destination.
    If we are not ready to retry the destination, this will raise a
    NotRetryingDestination exception. Otherwise, will return a Context Manager
    that will mark the destination as down if an exception is thrown (excluding
    CodeMessageException with code < 500)

    Args:
        destination: name of homeserver
        clock: timing source
        store: datastore
        ignore_backoff: true to ignore the historical backoff data and
            try the request anyway. We will still reset the retry_interval on success.

    Example usage:

        try:
            limiter = await get_retry_limiter(destination, clock, store)
            with limiter:
                response = await do_request()
        except NotRetryingDestination:
            # We aren't ready to retry that destination.
            raise
    """
    failure_ts = None
    retry_last_ts, retry_interval = (0, 0)

    retry_timings = await store.get_destination_retry_timings(destination)

    if retry_timings:
        failure_ts = retry_timings.failure_ts
        retry_last_ts = retry_timings.retry_last_ts
        retry_interval = retry_timings.retry_interval

        now = int(clock.time_msec())

        if not ignore_backoff and retry_last_ts + retry_interval > now:
            raise NotRetryingDestination(
                retry_last_ts=retry_last_ts,
                retry_interval=retry_interval,
                destination=destination,
            )

    # if we are ignoring the backoff data, we should also not increment the backoff
    # when we get another failure - otherwise a server can very quickly reach the
    # maximum backoff even though it might only have been down briefly
    backoff_on_failure = not ignore_backoff

    return RetryDestinationLimiter(
        destination,
        clock,
        store,
        failure_ts,
        retry_interval,
        backoff_on_failure=backoff_on_failure,
        **kwargs,
    )


async def filter_destinations_by_retry_limiter(
    destinations: StrCollection,
    clock: Clock,
    store: DataStore,
    retry_due_within_ms: int = 0,
) -> StrCollection:
    """Filter down the list of destinations to only those that will are either
    alive or due for a retry (within `retry_due_within_ms`)
    """
    if not destinations:
        return destinations

    retry_timings = await store.get_destination_retry_timings_batch(destinations)

    now = int(clock.time_msec())

    return [
        destination
        for destination, timings in retry_timings.items()
        if timings is None
        or timings.retry_last_ts + timings.retry_interval <= now + retry_due_within_ms
    ]


class RetryDestinationLimiter:
    def __init__(
        self,
        destination: str,
        clock: Clock,
        store: DataStore,
        failure_ts: Optional[int],
        retry_interval: int,
        backoff_on_404: bool = False,
        backoff_on_failure: bool = True,
        notifier: Optional["Notifier"] = None,
        replication_client: Optional["ReplicationCommandHandler"] = None,
        backoff_on_all_error_codes: bool = False,
    ):
        """Marks the destination as "down" if an exception is thrown in the
        context, except for CodeMessageException with code < 500.

        If no exception is raised, marks the destination as "up".

        Args:
            destination
            clock
            store
            failure_ts: when this destination started failing (in ms since
                the epoch), or zero if the last request was successful
            retry_interval: The next retry interval taken from the
                database in milliseconds, or zero if the last request was
                successful.
            backoff_on_404: Back off if we get a 404
            backoff_on_failure: set to False if we should not increase the
                retry interval on a failure.
            notifier: A notifier used to mark servers as up.
            replication_client A replication client used to mark servers as up.
            backoff_on_all_error_codes: Whether we should back off on any
                error code.
        """
        self.clock = clock
        self.store = store
        self.destination = destination

        self.failure_ts = failure_ts
        self.retry_interval = retry_interval
        self.backoff_on_404 = backoff_on_404
        self.backoff_on_failure = backoff_on_failure
        self.backoff_on_all_error_codes = backoff_on_all_error_codes

        self.notifier = notifier
        self.replication_client = replication_client

        self.destination_min_retry_interval_ms = (
            self.store.hs.config.federation.destination_min_retry_interval_ms
        )
        self.destination_retry_multiplier = (
            self.store.hs.config.federation.destination_retry_multiplier
        )
        self.destination_max_retry_interval_ms = (
            self.store.hs.config.federation.destination_max_retry_interval_ms
        )

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        success = exc_type is None
        valid_err_code = False
        if exc_type is None:
            valid_err_code = True
        elif not issubclass(exc_type, Exception):
            # avoid treating exceptions which don't derive from Exception as
            # failures; this is mostly so as not to catch defer._DefGen.
            valid_err_code = True
        elif isinstance(exc_val, CodeMessageException):
            # Some error codes are perfectly fine for some APIs, whereas other
            # APIs may expect to never received e.g. a 404. It's important to
            # handle 404 as some remote servers will return a 404 when the HS
            # has been decommissioned.
            # If we get a 401, then we should probably back off since they
            # won't accept our requests for at least a while.
            # 429 is us being aggressively rate limited, so lets rate limit
            # ourselves.
            if self.backoff_on_all_error_codes:
                valid_err_code = False
            elif exc_val.code == 404 and self.backoff_on_404:
                valid_err_code = False
            elif exc_val.code in (401, 429):
                valid_err_code = False
            elif exc_val.code < 500:
                valid_err_code = True
            else:
                valid_err_code = False

        # Whether previous requests to the destination had been failing.
        previously_failing = bool(self.failure_ts)

        if success:
            # We connected successfully.
            if not self.retry_interval:
                return

            logger.debug(
                "Connection to %s was successful; clearing backoff", self.destination
            )
            self.failure_ts = None
            retry_last_ts = 0
            self.retry_interval = 0
        elif valid_err_code:
            # We got a potentially valid error code back. We don't reset the
            # timers though, as the other side might actually be down anyway
            # (e.g. some deprovisioned servers will always return a 404 or 403,
            # and we don't want to keep resetting the retry timers for them).
            return
        elif not self.backoff_on_failure:
            return
        else:
            # We couldn't connect.
            if self.retry_interval:
                self.retry_interval = int(
                    self.retry_interval
                    * self.destination_retry_multiplier
                    * random.uniform(0.8, 1.4)
                )

                if self.retry_interval >= self.destination_max_retry_interval_ms:
                    self.retry_interval = self.destination_max_retry_interval_ms
            else:
                self.retry_interval = self.destination_min_retry_interval_ms

            logger.info(
                "Connection to %s was unsuccessful (%s(%s)); backoff now %i",
                self.destination,
                exc_type,
                exc_val,
                self.retry_interval,
            )
            retry_last_ts = int(self.clock.time_msec())

            if self.failure_ts is None:
                self.failure_ts = retry_last_ts

        # Whether the current request to the destination had been failing.
        currently_failing = bool(self.failure_ts)

        async def store_retry_timings() -> None:
            try:
                await self.store.set_destination_retry_timings(
                    self.destination,
                    self.failure_ts,
                    retry_last_ts,
                    self.retry_interval,
                )

                # If the server was previously failing, but is no longer.
                if previously_failing and not currently_failing:
                    if self.notifier:
                        # Inform the relevant places that the remote server is back up.
                        self.notifier.notify_remote_server_up(self.destination)

                    if self.replication_client:
                        # Inform other workers that the remote server is up.
                        self.replication_client.send_remote_server_up(self.destination)

            except Exception:
                logger.exception("Failed to store destination_retry_timings")

        # we deliberately do this in the background.
        run_as_background_process("store_retry_timings", store_retry_timings)

from enum import Enum
from typing import Optional
from pyflink.common import Duration
from pyflink.datastream.checkpoint_storage import CheckpointStorage, _from_j_checkpoint_storage
from pyflink.datastream.checkpointing_mode import CheckpointingMode
from pyflink.java_gateway import get_gateway
__all__ = ['CheckpointConfig', 'ExternalizedCheckpointCleanup']

class CheckpointConfig(object):
    """
    Configuration that captures all checkpointing related settings.

    :data:`DEFAULT_MODE`:

    The default checkpoint mode: exactly once.

    :data:`DEFAULT_TIMEOUT`:

    The default timeout of a checkpoint attempt: 10 minutes.

    :data:`DEFAULT_MIN_PAUSE_BETWEEN_CHECKPOINTS`:

    The default minimum pause to be made between checkpoints: none.

    :data:`DEFAULT_MAX_CONCURRENT_CHECKPOINTS`:

    The default limit of concurrently happening checkpoints: one.
    """
    DEFAULT_MODE = CheckpointingMode.EXACTLY_ONCE
    DEFAULT_TIMEOUT = 10 * 60 * 1000
    DEFAULT_MIN_PAUSE_BETWEEN_CHECKPOINTS = 0
    DEFAULT_MAX_CONCURRENT_CHECKPOINTS = 1

    def __init__(self, j_checkpoint_config):
        if False:
            i = 10
            return i + 15
        self._j_checkpoint_config = j_checkpoint_config

    def is_checkpointing_enabled(self) -> bool:
        if False:
            return 10
        '\n        Checks whether checkpointing is enabled.\n\n        :return: True if checkpointing is enables, false otherwise.\n        '
        return self._j_checkpoint_config.isCheckpointingEnabled()

    def get_checkpointing_mode(self) -> CheckpointingMode:
        if False:
            print('Hello World!')
        '\n        Gets the checkpointing mode (exactly-once vs. at-least-once).\n\n        .. seealso:: :func:`set_checkpointing_mode`\n\n        :return: The :class:`CheckpointingMode`.\n        '
        return CheckpointingMode._from_j_checkpointing_mode(self._j_checkpoint_config.getCheckpointingMode())

    def set_checkpointing_mode(self, checkpointing_mode: CheckpointingMode) -> 'CheckpointConfig':
        if False:
            i = 10
            return i + 15
        '\n        Sets the checkpointing mode (:data:`CheckpointingMode.EXACTLY_ONCE` vs.\n        :data:`CheckpointingMode.AT_LEAST_ONCE`).\n\n        Example:\n        ::\n\n            >>> config.set_checkpointing_mode(CheckpointingMode.AT_LEAST_ONCE)\n\n        :param checkpointing_mode: The :class:`CheckpointingMode`.\n        '
        self._j_checkpoint_config.setCheckpointingMode(CheckpointingMode._to_j_checkpointing_mode(checkpointing_mode))
        return self

    def get_checkpoint_interval(self) -> int:
        if False:
            return 10
        '\n        Gets the interval in which checkpoints are periodically scheduled.\n\n        This setting defines the base interval. Checkpoint triggering may be delayed by the settings\n        :func:`get_max_concurrent_checkpoints` and :func:`get_min_pause_between_checkpoints`.\n\n        :return: The checkpoint interval, in milliseconds.\n        '
        return self._j_checkpoint_config.getCheckpointInterval()

    def set_checkpoint_interval(self, checkpoint_interval: int) -> 'CheckpointConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the interval in which checkpoints are periodically scheduled.\n\n        This setting defines the base interval. Checkpoint triggering may be delayed by the settings\n        :func:`set_max_concurrent_checkpoints` and :func:`set_min_pause_between_checkpoints`.\n\n        :param checkpoint_interval: The checkpoint interval, in milliseconds.\n        '
        self._j_checkpoint_config.setCheckpointInterval(checkpoint_interval)
        return self

    def get_checkpoint_timeout(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the maximum time that a checkpoint may take before being discarded.\n\n        :return: The checkpoint timeout, in milliseconds.\n        '
        return self._j_checkpoint_config.getCheckpointTimeout()

    def set_checkpoint_timeout(self, checkpoint_timeout: int) -> 'CheckpointConfig':
        if False:
            while True:
                i = 10
        '\n        Sets the maximum time that a checkpoint may take before being discarded.\n\n        :param checkpoint_timeout: The checkpoint timeout, in milliseconds.\n        '
        self._j_checkpoint_config.setCheckpointTimeout(checkpoint_timeout)
        return self

    def get_min_pause_between_checkpoints(self) -> int:
        if False:
            return 10
        '\n        Gets the minimal pause between checkpointing attempts. This setting defines how soon the\n        checkpoint coordinator may trigger another checkpoint after it becomes possible to trigger\n        another checkpoint with respect to the maximum number of concurrent checkpoints\n        (see :func:`get_max_concurrent_checkpoints`).\n\n        :return: The minimal pause before the next checkpoint is triggered.\n        '
        return self._j_checkpoint_config.getMinPauseBetweenCheckpoints()

    def set_min_pause_between_checkpoints(self, min_pause_between_checkpoints: int) -> 'CheckpointConfig':
        if False:
            print('Hello World!')
        '\n        Sets the minimal pause between checkpointing attempts. This setting defines how soon the\n        checkpoint coordinator may trigger another checkpoint after it becomes possible to trigger\n        another checkpoint with respect to the maximum number of concurrent checkpoints\n        (see :func:`set_max_concurrent_checkpoints`).\n\n        If the maximum number of concurrent checkpoints is set to one, this setting makes\n        effectively sure that a minimum amount of time passes where no checkpoint is in progress\n        at all.\n\n        :param min_pause_between_checkpoints: The minimal pause before the next checkpoint is\n                                              triggered.\n        '
        self._j_checkpoint_config.setMinPauseBetweenCheckpoints(min_pause_between_checkpoints)
        return self

    def get_max_concurrent_checkpoints(self) -> int:
        if False:
            return 10
        '\n        Gets the maximum number of checkpoint attempts that may be in progress at the same time.\n        If this value is *n*, then no checkpoints will be triggered while *n* checkpoint attempts\n        are currently in flight. For the next checkpoint to be triggered, one checkpoint attempt\n        would need to finish or expire.\n\n        :return: The maximum number of concurrent checkpoint attempts.\n        '
        return self._j_checkpoint_config.getMaxConcurrentCheckpoints()

    def set_max_concurrent_checkpoints(self, max_concurrent_checkpoints: int) -> 'CheckpointConfig':
        if False:
            return 10
        '\n        Sets the maximum number of checkpoint attempts that may be in progress at the same time.\n        If this value is *n*, then no checkpoints will be triggered while *n* checkpoint attempts\n        are currently in flight. For the next checkpoint to be triggered, one checkpoint attempt\n        would need to finish or expire.\n\n        :param max_concurrent_checkpoints: The maximum number of concurrent checkpoint attempts.\n        '
        self._j_checkpoint_config.setMaxConcurrentCheckpoints(max_concurrent_checkpoints)
        return self

    def is_fail_on_checkpointing_errors(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        This determines the behaviour of tasks if there is an error in their local checkpointing.\n        If this returns true, tasks will fail as a reaction. If this returns false, task will only\n        decline the failed checkpoint.\n\n        :return: ``True`` if failing on checkpointing errors, false otherwise.\n        '
        return self._j_checkpoint_config.isFailOnCheckpointingErrors()

    def set_fail_on_checkpointing_errors(self, fail_on_checkpointing_errors: bool) -> 'CheckpointConfig':
        if False:
            return 10
        '\n        Sets the expected behaviour for tasks in case that they encounter an error in their\n        checkpointing procedure. If this is set to true, the task will fail on checkpointing error.\n        If this is set to false, the task will only decline a the checkpoint and continue running.\n        The default is true.\n\n        Example:\n        ::\n\n            >>> config.set_fail_on_checkpointing_errors(False)\n\n        :param fail_on_checkpointing_errors: ``True`` if failing on checkpointing errors,\n                                             false otherwise.\n        '
        self._j_checkpoint_config.setFailOnCheckpointingErrors(fail_on_checkpointing_errors)
        return self

    def get_tolerable_checkpoint_failure_number(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Get the defined number of consecutive checkpoint failures that will be tolerated, before the\n        whole job is failed over.\n\n        :return: The maximum number of tolerated checkpoint failures.\n        '
        return self._j_checkpoint_config.getTolerableCheckpointFailureNumber()

    def set_tolerable_checkpoint_failure_number(self, tolerable_checkpoint_failure_number: int) -> 'CheckpointConfig':
        if False:
            i = 10
            return i + 15
        '\n        This defines how many consecutive checkpoint failures will be tolerated, before the whole\n        job is failed over. The default value is `0`, which means no checkpoint failures will be\n        tolerated, and the job will fail on first reported checkpoint failure.\n\n        Example:\n        ::\n\n            >>> config.set_tolerable_checkpoint_failure_number(2)\n\n        :param tolerable_checkpoint_failure_number: The maximum number of tolerated checkpoint\n                                                    failures.\n        '
        self._j_checkpoint_config.setTolerableCheckpointFailureNumber(tolerable_checkpoint_failure_number)
        return self

    def enable_externalized_checkpoints(self, cleanup_mode: 'ExternalizedCheckpointCleanup') -> 'CheckpointConfig':
        if False:
            while True:
                i = 10
        '\n        Sets the mode for externalized checkpoint clean-up. Externalized checkpoints will be enabled\n        automatically unless the mode is set to\n        :data:`ExternalizedCheckpointCleanup.NO_EXTERNALIZED_CHECKPOINTS`.\n\n        Externalized checkpoints write their meta data out to persistent storage and are **not**\n        automatically cleaned up when the owning job fails or is suspended (terminating with job\n        status ``FAILED`` or ``SUSPENDED``). In this case, you have to manually clean up the\n        checkpoint state, both the meta data and actual program state.\n\n        The :class:`ExternalizedCheckpointCleanup` mode defines how an externalized checkpoint\n        should be cleaned up on job cancellation. If you choose to retain externalized checkpoints\n        on cancellation you have to handle checkpoint clean-up manually when you cancel the job as\n        well (terminating with job status ``CANCELED``).\n\n        The target directory for externalized checkpoints is configured via\n        ``org.apache.flink.configuration.CheckpointingOptions#CHECKPOINTS_DIRECTORY``.\n\n        Example:\n        ::\n\n            >>> config.enable_externalized_checkpoints(\n            ...     ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION)\n\n        :param cleanup_mode: Externalized checkpoint clean-up behaviour, the mode could be\n                             :data:`ExternalizedCheckpointCleanup.DELETE_ON_CANCELLATION`,\n                             :data:`ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION` or\n                             :data:`ExternalizedCheckpointCleanup.NO_EXTERNALIZED_CHECKPOINTS`\n\n        .. note:: Deprecated in 1.15. Use :func:`set_externalized_checkpoint_cleanup` instead.\n        '
        self._j_checkpoint_config.enableExternalizedCheckpoints(ExternalizedCheckpointCleanup._to_j_externalized_checkpoint_cleanup(cleanup_mode))
        return self

    def set_externalized_checkpoint_cleanup(self, cleanup_mode: 'ExternalizedCheckpointCleanup') -> 'CheckpointConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the mode for externalized checkpoint clean-up. Externalized checkpoints will be enabled\n        automatically unless the mode is set to\n        :data:`ExternalizedCheckpointCleanup.NO_EXTERNALIZED_CHECKPOINTS`.\n\n        Externalized checkpoints write their meta data out to persistent storage and are **not**\n        automatically cleaned up when the owning job fails or is suspended (terminating with job\n        status ``FAILED`` or ``SUSPENDED``). In this case, you have to manually clean up the\n        checkpoint state, both the meta data and actual program state.\n\n        The :class:`ExternalizedCheckpointCleanup` mode defines how an externalized checkpoint\n        should be cleaned up on job cancellation. If you choose to retain externalized checkpoints\n        on cancellation you have to handle checkpoint clean-up manually when you cancel the job as\n        well (terminating with job status ``CANCELED``).\n\n        The target directory for externalized checkpoints is configured via\n        ``org.apache.flink.configuration.CheckpointingOptions#CHECKPOINTS_DIRECTORY``.\n\n        Example:\n        ::\n\n            >>> config.set_externalized_checkpoint_cleanup(\n            ...     ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION)\n\n        :param cleanup_mode: Externalized checkpoint clean-up behaviour, the mode could be\n                             :data:`ExternalizedCheckpointCleanup.DELETE_ON_CANCELLATION`,\n                             :data:`ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION` or\n                             :data:`ExternalizedCheckpointCleanup.NO_EXTERNALIZED_CHECKPOINTS`\n        '
        self._j_checkpoint_config.setExternalizedCheckpointCleanup(ExternalizedCheckpointCleanup._to_j_externalized_checkpoint_cleanup(cleanup_mode))
        return self

    def is_externalized_checkpoints_enabled(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns whether checkpoints should be persisted externally.\n\n        :return: ``True`` if checkpoints should be externalized, false otherwise.\n        '
        return self._j_checkpoint_config.isExternalizedCheckpointsEnabled()

    def get_externalized_checkpoint_cleanup(self) -> Optional['ExternalizedCheckpointCleanup']:
        if False:
            return 10
        '\n        Returns the cleanup behaviour for externalized checkpoints.\n\n        :return: The cleanup behaviour for externalized checkpoints or ``None`` if none is\n                 configured.\n        '
        cleanup_mode = self._j_checkpoint_config.getExternalizedCheckpointCleanup()
        if cleanup_mode is None:
            return None
        else:
            return ExternalizedCheckpointCleanup._from_j_externalized_checkpoint_cleanup(cleanup_mode)

    def is_unaligned_checkpoints_enabled(self) -> bool:
        if False:
            return 10
        '\n        Returns whether unaligned checkpoints are enabled.\n\n        :return: ``True`` if unaligned checkpoints are enabled.\n        '
        return self._j_checkpoint_config.isUnalignedCheckpointsEnabled()

    def enable_unaligned_checkpoints(self, enabled: bool=True) -> 'CheckpointConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Enables unaligned checkpoints, which greatly reduce checkpointing times under backpressure.\n\n        Unaligned checkpoints contain data stored in buffers as part of the checkpoint state, which\n        allows checkpoint barriers to overtake these buffers. Thus, the checkpoint duration becomes\n        independent of the current throughput as checkpoint barriers are effectively not embedded\n        into the stream of data anymore.\n\n        Unaligned checkpoints can only be enabled if :func:`get_checkpointing_mode` is\n        :data:`CheckpointingMode.EXACTLY_ONCE`.\n\n        :param enabled: ``True`` if a checkpoints should be taken in unaligned mode.\n        '
        self._j_checkpoint_config.enableUnalignedCheckpoints(enabled)
        return self

    def disable_unaligned_checkpoints(self) -> 'CheckpointConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Enables unaligned checkpoints, which greatly reduce checkpointing times under backpressure\n        (experimental).\n\n        Unaligned checkpoints contain data stored in buffers as part of the checkpoint state, which\n        allows checkpoint barriers to overtake these buffers. Thus, the checkpoint duration becomes\n        independent of the current throughput as checkpoint barriers are effectively not embedded\n        into the stream of data anymore.\n\n        Unaligned checkpoints can only be enabled if :func:`get_checkpointing_mode` is\n        :data:`CheckpointingMode.EXACTLY_ONCE`.\n        '
        self.enable_unaligned_checkpoints(False)
        return self

    def set_alignment_timeout(self, alignment_timeout: Duration) -> 'CheckpointConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Only relevant if :func:`enable_unaligned_checkpoints` is enabled.\n\n        If ``alignment_timeout`` has value equal to ``0``, checkpoints will always start unaligned.\n        If ``alignment_timeout`` has value greater then ``0``, checkpoints will start aligned. If\n        during checkpointing, checkpoint start delay exceeds this ``alignment_timeout``, alignment\n        will timeout and checkpoint will start working as unaligned checkpoint.\n\n        :param alignment_timeout: The duration until the aligned checkpoint will be converted into\n                                  an unaligned checkpoint.\n        '
        self._j_checkpoint_config.setAlignmentTimeout(alignment_timeout._j_duration)
        return self

    def get_alignment_timeout(self) -> 'Duration':
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the alignment timeout, as configured via :func:`set_alignment_timeout` or\n        ``org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions#ALIGNMENT_TIMEOUT``.\n\n        :return: the alignment timeout.\n        '
        return Duration(self._j_checkpoint_config.getAlignmentTimeout())

    def set_force_unaligned_checkpoints(self, force_unaligned_checkpoints: bool=True) -> 'CheckpointConfig':
        if False:
            return 10
        '\n        Checks whether unaligned checkpoints are forced, despite currently non-checkpointable\n        iteration feedback or custom partitioners.\n\n        :param force_unaligned_checkpoints: The flag to force unaligned checkpoints.\n        '
        self._j_checkpoint_config.setForceUnalignedCheckpoints(force_unaligned_checkpoints)
        return self

    def is_force_unaligned_checkpoints(self) -> 'bool':
        if False:
            while True:
                i = 10
        '\n        Checks whether unaligned checkpoints are forced, despite iteration feedback or custom\n        partitioners.\n\n        :return: True, if unaligned checkpoints are forced, false otherwise.\n        '
        return self._j_checkpoint_config.isForceUnalignedCheckpoints()

    def set_checkpoint_storage(self, storage: CheckpointStorage) -> 'CheckpointConfig':
        if False:
            i = 10
            return i + 15
        '\n        Checkpoint storage defines how stat backends checkpoint their state for fault\n        tolerance in streaming applications. Various implementations store their checkpoints\n        in different fashions and have different requirements and availability guarantees.\n\n        For example, `JobManagerCheckpointStorage` stores checkpoints in the memory of the\n        JobManager. It is lightweight and without additional dependencies but is not highly\n        available and only supports small state sizes. This checkpoint storage policy is convenient\n        for local testing and development.\n\n        The `FileSystemCheckpointStorage` stores checkpoints in a filesystem. For systems like\n        HDFS, NFS Drivs, S3, and GCS, this storage policy supports large state size, in the\n        magnitude of many terabytes while providing a highly available foundation for stateful\n        applications. This checkpoint storage policy is recommended for most production deployments.\n        '
        self._j_checkpoint_config.setCheckpointStorage(storage._j_checkpoint_storage)
        return self

    def set_checkpoint_storage_dir(self, checkpoint_path: str) -> 'CheckpointConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Configures the application to write out checkpoint snapshots to the configured directory.\n        See `FileSystemCheckpointStorage` for more details on checkpointing to a file system.\n        '
        self._j_checkpoint_config.setCheckpointStorage(checkpoint_path)
        return self

    def get_checkpoint_storage(self) -> Optional[CheckpointStorage]:
        if False:
            i = 10
            return i + 15
        '\n        The checkpoint storage that has been configured for the Job, or None if\n        none has been set.\n        '
        j_storage = self._j_checkpoint_config.getCheckpointStorage()
        if j_storage is None:
            return None
        else:
            return _from_j_checkpoint_storage(j_storage)

class ExternalizedCheckpointCleanup(Enum):
    """
    Cleanup behaviour for externalized checkpoints when the job is cancelled.

    :data:`DELETE_ON_CANCELLATION`:

    Delete externalized checkpoints on job cancellation.

    All checkpoint state will be deleted when you cancel the owning
    job, both the meta data and actual program state. Therefore, you
    cannot resume from externalized checkpoints after the job has been
    cancelled.

    Note that checkpoint state is always kept if the job terminates
    with state ``FAILED``.

    :data:`RETAIN_ON_CANCELLATION`:

    Retain externalized checkpoints on job cancellation.

    All checkpoint state is kept when you cancel the owning job. You
    have to manually delete both the checkpoint meta data and actual
    program state after cancelling the job.

    Note that checkpoint state is always kept if the job terminates
    with state ``FAILED``.

    :data:`NO_EXTERNALIZED_CHECKPOINTS`:

    Externalized checkpoints are disabled completely.
    """
    DELETE_ON_CANCELLATION = 0
    RETAIN_ON_CANCELLATION = 1
    NO_EXTERNALIZED_CHECKPOINTS = 2

    @staticmethod
    def _from_j_externalized_checkpoint_cleanup(j_cleanup_mode) -> 'ExternalizedCheckpointCleanup':
        if False:
            return 10
        return ExternalizedCheckpointCleanup[j_cleanup_mode.name()]

    def _to_j_externalized_checkpoint_cleanup(self):
        if False:
            print('Hello World!')
        gateway = get_gateway()
        JExternalizedCheckpointCleanup = gateway.jvm.org.apache.flink.streaming.api.environment.CheckpointConfig.ExternalizedCheckpointCleanup
        return getattr(JExternalizedCheckpointCleanup, self.name)
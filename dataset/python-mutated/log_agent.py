import logging
from typing import Tuple
import concurrent.futures
import ray.dashboard.modules.log.log_utils as log_utils
import ray.dashboard.modules.log.log_consts as log_consts
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.ray_constants import env_integer
import asyncio
import grpc
import io
import os
from pathlib import Path
from ray.core.generated import reporter_pb2
from ray.core.generated import reporter_pb2_grpc
from ray._private.ray_constants import LOG_PREFIX_TASK_ATTEMPT_START, LOG_PREFIX_TASK_ATTEMPT_END
logger = logging.getLogger(__name__)
routes = dashboard_optional_utils.DashboardAgentRouteTable
BLOCK_SIZE = 1 << 16
DEFAULT_KEEP_ALIVE_INTERVAL_SEC = 1
RAY_DASHBOARD_LOG_TASK_LOG_SEARCH_MAX_WORKER_COUNT = env_integer('RAY_DASHBOARD_LOG_TASK_LOG_SEARCH_MAX_WORKER_COUNT', default=2)

def find_offset_of_content_in_file(file: io.BufferedIOBase, content: bytes, start_offset: int=0) -> int:
    if False:
        print('Hello World!')
    'Find the offset of the first occurrence of content in a file.\n\n    Args:\n        file: File object\n        content: Content to find\n        start_offset: Start offset to read from, inclusive.\n\n    Returns:\n        Offset of the first occurrence of content in a file.\n    '
    logger.debug(f'Finding offset of content {content} in file')
    file.seek(start_offset, io.SEEK_SET)
    offset = start_offset
    while True:
        block_data = file.read(BLOCK_SIZE)
        if block_data == b'':
            return -1
        block_offset = block_data.find(content)
        if block_offset != -1:
            return offset + block_offset
        offset += len(block_data)

def find_end_offset_file(file: io.BufferedIOBase) -> int:
    if False:
        print('Hello World!')
    '\n    Find the offset of the end of a file without changing the file pointer.\n\n    Args:\n        file: File object\n\n    Returns:\n        Offset of the end of a file.\n    '
    old_pos = file.tell()
    file.seek(0, io.SEEK_END)
    end = file.tell()
    file.seek(old_pos, io.SEEK_SET)
    return end

def find_end_offset_next_n_lines_from_offset(file: io.BufferedIOBase, start_offset: int, n: int) -> int:
    if False:
        return 10
    '\n    Find the offsets of next n lines from a start offset.\n\n    Args:\n        file: File object\n        start_offset: Start offset to read from, inclusive.\n        n: Number of lines to find.\n\n    Returns:\n        Offset of the end of the next n line (exclusive)\n    '
    file.seek(start_offset)
    end_offset = None
    for _ in range(n):
        line = file.readline()
        if not line:
            break
        end_offset = file.tell()
    logger.debug(f'Found next {n} lines from {start_offset} offset')
    return end_offset if end_offset is not None else file.seek(0, io.SEEK_END)

def find_start_offset_last_n_lines_from_offset(file: io.BufferedIOBase, offset: int, n: int, block_size: int=BLOCK_SIZE) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Find the offset of the beginning of the line of the last X lines from an offset.\n\n    Args:\n        file: File object\n        offset: Start offset from which to find last X lines, -1 means end of file.\n            The offset is exclusive, i.e. data at the offset is not included\n            in the result.\n        n: Number of lines to find\n        block_size: Block size to read from file\n\n    Returns:\n        Offset of the beginning of the line of the last X lines from a start offset.\n    '
    logger.debug(f'Finding last {n} lines from {offset} offset')
    if offset == -1:
        offset = file.seek(0, io.SEEK_END)
    else:
        file.seek(offset, io.SEEK_SET)
    if n == 0:
        return offset
    nbytes_from_end = 0
    file.seek(max(0, offset - 1), os.SEEK_SET)
    if file.read(1) != b'\n':
        n -= 1
    lines_more = n
    read_offset = max(0, offset - block_size)
    prev_offset = offset
    while lines_more >= 0 and read_offset >= 0:
        file.seek(read_offset, 0)
        block_data = file.read(min(block_size, prev_offset - read_offset))
        num_lines = block_data.count(b'\n')
        if num_lines > lines_more:
            lines = block_data.split(b'\n', num_lines - lines_more)
            nbytes_from_end += len(lines[-1])
            break
        lines_more -= num_lines
        nbytes_from_end += len(block_data)
        if read_offset == 0:
            break
        prev_offset = read_offset
        read_offset = max(0, read_offset - block_size)
    offset_read_start = offset - nbytes_from_end
    assert offset_read_start >= 0, f'Read start offset({offset_read_start}) should be non-negative'
    return offset_read_start

async def _stream_log_in_chunk(context: grpc.aio.ServicerContext, file: io.BufferedIOBase, start_offset: int, end_offset: int=-1, keep_alive_interval_sec: int=-1, block_size: int=BLOCK_SIZE):
    """Streaming log in chunk from start to end offset.

    Stream binary file content in chunks from start offset to an end
    offset if provided, else to the end of the file.

    Args:
        context: gRPC server side context
        file: Binary file to stream
        start_offset: File offset where streaming starts
        end_offset: If -1, implying streaming til the EOF.
        keep_alive_interval_sec: Duration for which streaming will be
            retried when reaching the file end, -1 means no retry.
        block_size: Number of bytes per chunk, exposed for testing

    Return:
        Async generator of StreamReply
    """
    assert 'b' in file.mode, 'Only binary file is supported.'
    assert not (keep_alive_interval_sec >= 0 and end_offset != -1), 'Keep-alive is not allowed when specifying an end offset'
    file.seek(start_offset, 0)
    cur_offset = start_offset
    while not context.done():
        if end_offset != -1:
            to_read = min(end_offset - cur_offset, block_size)
        else:
            to_read = block_size
        bytes = file.read(to_read)
        if bytes == b'':
            if keep_alive_interval_sec >= 0:
                await asyncio.sleep(keep_alive_interval_sec)
                continue
            break
        logger.debug(f'Sending {len(bytes)} bytes at {cur_offset}')
        yield reporter_pb2.StreamLogReply(data=bytes)
        cur_offset += len(bytes)
        if end_offset != -1 and cur_offset >= end_offset:
            break

class LogAgent(dashboard_utils.DashboardAgentModule):

    def __init__(self, dashboard_agent):
        if False:
            while True:
                i = 10
        super().__init__(dashboard_agent)
        log_utils.register_mimetypes()
        routes.static('/logs', self._dashboard_agent.log_dir, show_index=True)

    async def run(self, server):
        pass

    @staticmethod
    def is_minimal_module():
        if False:
            i = 10
            return i + 15
        return False
_task_log_search_worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=RAY_DASHBOARD_LOG_TASK_LOG_SEARCH_MAX_WORKER_COUNT)

class LogAgentV1Grpc(dashboard_utils.DashboardAgentModule):

    def __init__(self, dashboard_agent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(dashboard_agent)

    async def run(self, server):
        if server:
            reporter_pb2_grpc.add_LogServiceServicer_to_server(self, server)

    @staticmethod
    def is_minimal_module():
        if False:
            print('Hello World!')
        return False

    async def ListLogs(self, request, context):
        """
        Lists all files in the active Ray logs directory.

        Part of `LogService` gRPC.

        NOTE: These RPCs are used by state_head.py, not log_head.py
        """
        path = Path(self._dashboard_agent.log_dir)
        if not path.exists():
            raise FileNotFoundError(f'Could not find log dir at path: {self._dashboard_agent.log_dir}It is unexpected. Please report an issue to Ray Github.')
        log_files = []
        for p in path.glob(request.glob_filter):
            log_files.append(str(p.relative_to(path)))
        return reporter_pb2.ListLogsReply(log_files=log_files)

    @classmethod
    async def _find_task_log_offsets(cls, task_id: str, attempt_number: int, lines: int, f: io.BufferedIOBase) -> Tuple[int, int]:
        """Find the start and end offsets in the log file for a task attempt
        Current task log is in the format of below:

            :job_id:xxx
            :task_name:xxx
            :task_attempt_start:<task_id>-<attempt_number>
            ...
            actual user logs
            ...
            :task_attempt_end:<task_id>-<attempt_number>
            ... (other tasks)


        For async actor tasks, task logs from multiple tasks might however
        be interleaved.
        """
        task_attempt_start_magic_line = f'{LOG_PREFIX_TASK_ATTEMPT_START}{task_id}-{attempt_number}\n'
        task_attempt_magic_line_offset = await asyncio.get_running_loop().run_in_executor(_task_log_search_worker_pool, find_offset_of_content_in_file, f, task_attempt_start_magic_line.encode())
        if task_attempt_magic_line_offset == -1:
            raise FileNotFoundError(f'Log for task attempt({task_id},{attempt_number}) not found')
        start_offset = task_attempt_magic_line_offset + len(task_attempt_start_magic_line)
        task_attempt_end_magic_line = f'{LOG_PREFIX_TASK_ATTEMPT_END}{task_id}-{attempt_number}\n'
        end_offset = await asyncio.get_running_loop().run_in_executor(_task_log_search_worker_pool, find_offset_of_content_in_file, f, task_attempt_end_magic_line.encode(), start_offset)
        if end_offset == -1:
            end_offset = find_end_offset_file(f)
        if lines != -1:
            start_offset = max(find_start_offset_last_n_lines_from_offset(f, end_offset, lines), start_offset)
        return (start_offset, end_offset)

    async def StreamLog(self, request, context):
        """
        Streams the log in real time starting from `request.lines` number of lines from
        the end of the file if `request.keep_alive == True`. Else, it terminates the
        stream once there are no more bytes to read from the log file.

        Part of `LogService` gRPC.

        NOTE: These RPCs are used by state_head.py, not log_head.py
        """
        lines = request.lines if request.lines else 1000
        if not Path(request.log_file_name).is_absolute():
            filepath = Path(self._dashboard_agent.log_dir) / request.log_file_name
        else:
            filepath = Path(request.log_file_name)
        if not filepath.is_file():
            await context.send_initial_metadata([[log_consts.LOG_GRPC_ERROR, log_consts.FILE_NOT_FOUND]])
        else:
            with open(filepath, 'rb') as f:
                await context.send_initial_metadata([])
                start_offset = request.start_offset if request.HasField('start_offset') else 0
                end_offset = request.end_offset if request.HasField('end_offset') else find_end_offset_file(f)
                if lines != -1:
                    start_offset = max(find_start_offset_last_n_lines_from_offset(f, offset=end_offset, n=lines), start_offset)
                keep_alive_interval_sec = -1
                if request.keep_alive:
                    keep_alive_interval_sec = request.interval if request.interval else DEFAULT_KEEP_ALIVE_INTERVAL_SEC
                    end_offset = -1
                logger.info(f'Tailing logs from {start_offset} to {end_offset} for lines={lines}, with keep_alive={keep_alive_interval_sec}')
                async for chunk_res in _stream_log_in_chunk(context=context, file=f, start_offset=start_offset, end_offset=end_offset, keep_alive_interval_sec=keep_alive_interval_sec):
                    yield chunk_res
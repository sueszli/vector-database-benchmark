from __future__ import print_function, absolute_import, division
import collections
import contextlib
import functools
import sys
import os
import re
from .sysinfo import RUNNING_ON_APPVEYOR as APPVEYOR
from .sysinfo import RUNNING_ON_TRAVIS as TRAVIS
from .sysinfo import RESOLVER_NOT_SYSTEM as ARES
from .sysinfo import RESOLVER_ARES
from .sysinfo import RESOLVER_DNSPYTHON
from .sysinfo import RUNNING_ON_CI
from .sysinfo import RUNNING_ON_MUSLLINUX
from .sysinfo import RUN_COVERAGE
from .sysinfo import PYPY
from .sysinfo import PYPY3
from .sysinfo import PY38
from .sysinfo import PY39
from .sysinfo import PY310
from .sysinfo import PY311
from .sysinfo import PY312
from .sysinfo import WIN
from .sysinfo import OSX
from .sysinfo import LIBUV
from .sysinfo import CFFI_BACKEND
from . import flaky
CPYTHON = not PYPY
no_switch_tests = 'test_patched_select.SelectTestCase.test_error_conditions\ntest_patched_ftplib.*.test_all_errors\ntest_patched_ftplib.*.test_getwelcome\ntest_patched_ftplib.*.test_sanitize\ntest_patched_ftplib.*.test_set_pasv\n#test_patched_ftplib.TestIPv6Environment.test_af\ntest_patched_socket.TestExceptions.testExceptionTree\ntest_patched_socket.Urllib2FileobjectTest.testClose\ntest_patched_socket.TestLinuxAbstractNamespace.testLinuxAbstractNamespace\ntest_patched_socket.TestLinuxAbstractNamespace.testMaxName\ntest_patched_socket.TestLinuxAbstractNamespace.testNameOverflow\ntest_patched_socket.FileObjectInterruptedTestCase.*\ntest_patched_urllib.*\ntest_patched_asyncore.HelperFunctionTests.*\ntest_patched_httplib.BasicTest.*\ntest_patched_httplib.HTTPSTimeoutTest.test_attributes\ntest_patched_httplib.HeaderTests.*\ntest_patched_httplib.OfflineTest.*\ntest_patched_httplib.HTTPSTimeoutTest.test_host_port\ntest_patched_httplib.SourceAddressTest.testHTTPSConnectionSourceAddress\ntest_patched_select.SelectTestCase.test_error_conditions\ntest_patched_smtplib.NonConnectingTests.*\ntest_patched_urllib2net.OtherNetworkTests.*\ntest_patched_wsgiref.*\ntest_patched_subprocess.HelperFunctionTests.*\n'
ignore_switch_tests = '\ntest_patched_socket.GeneralModuleTests.*\ntest_patched_httpservers.BaseHTTPRequestHandlerTestCase.*\ntest_patched_queue.*\ntest_patched_signal.SiginterruptTest.*\ntest_patched_urllib2.*\ntest_patched_ssl.*\ntest_patched_signal.BasicSignalTests.*\ntest_patched_threading_local.*\ntest_patched_threading.*\n'

def make_re(tests):
    if False:
        return 10
    tests = [x.strip().replace('\\.', '\\\\.').replace('*', '.*?') for x in tests.split('\n') if x.strip()]
    return re.compile('^%s$' % '|'.join(tests))
no_switch_tests = make_re(no_switch_tests)
ignore_switch_tests = make_re(ignore_switch_tests)

def get_switch_expected(fullname):
    if False:
        i = 10
        return i + 15
    '\n    >>> get_switch_expected(\'test_patched_select.SelectTestCase.test_error_conditions\')\n    False\n    >>> get_switch_expected(\'test_patched_socket.GeneralModuleTests.testCrucialConstants\')\n    False\n    >>> get_switch_expected(\'test_patched_socket.SomeOtherTest.testHello\')\n    True\n    >>> get_switch_expected("test_patched_httplib.BasicTest.test_bad_status_repr")\n    False\n    '
    if ignore_switch_tests.match(fullname) is not None:
        return None
    if no_switch_tests.match(fullname) is not None:
        return False
    return True
disabled_tests = ['test_signal.GenericTests.test_functions_module_attr', 'test_threading.ThreadTests.test_no_refcycle_through_target', 'test_httplib.HTTPSTest.test_local_bad_hostname', 'test_httplib.HTTPSTest.test_local_good_hostname', 'test_httplib.HTTPSTest.test_local_unknown_cert', 'test_threading.ThreadTests.test_PyThreadState_SetAsyncExc', 'test_threading.ThreadTests.test_join_nondaemon_on_shutdown', 'test_urllib2net.TimeoutTest.test_ftp_no_timeout', 'test_urllib2net.TimeoutTest.test_ftp_timeout', 'test_urllib2net.TimeoutTest.test_http_no_timeout', 'test_urllib2net.TimeoutTest.test_http_timeout', 'test_urllib2net.OtherNetworkTests.test_ftp', 'test_urllib2net.OtherNetworkTests.test_urlwithfrag', 'test_urllib2net.OtherNetworkTests.test_sites_no_connection_close', 'test_socket.UDPTimeoutTest.testUDPTimeout', 'test_socket.GeneralModuleTests.testRefCountGetNameInfo', 'test_socket.NetworkConnectionNoServer.test_create_connection_timeout', 'test_asyncore.BaseTestAPI.test_handle_expt', 'test_asyncore.HelperFunctionTests.test_compact_traceback', 'test_signal.WakeupSignalTests.test_wakeup_fd_early', 'test_signal.WakeupSignalTests.test_wakeup_fd_during', 'test_signal.SiginterruptTest.test_without_siginterrupt', 'test_signal.SiginterruptTest.test_siginterrupt_on', 'test_signal.SiginterruptTest.test_siginterrupt_off', 'test_signal.StressTest.test_stress_modifying_handlers', 'test_signal.PosixTests.test_interprocess_signal', 'test_subprocess.ProcessTestCase.test_leak_fast_process_del_killed', 'test_subprocess.ProcessTestCase.test_zombie_fast_process_del', 'test_subprocess.ProcessTestCase.test_no_leaking', 'test_subprocess.ProcessTestCase.test_leaking_fds_on_error', 'test_subprocess.POSIXProcessTestCase.test_stopped', 'test_ssl.ThreadedTests.test_default_ciphers', 'test_ssl.ThreadedTests.test_empty_cert', 'test_ssl.ThreadedTests.test_malformed_cert', 'test_ssl.ThreadedTests.test_malformed_key', 'test_ssl.NetworkedTests.test_non_blocking_connect_ex', 'test_ssl.NetworkedTests.test_algorithms', 'test_ssl.BasicSocketTests.test_random_fork', 'test_ssl.BasicSocketTests.test_dealloc_warn', 'test_ssl.BasicSocketTests.test_connect_ex_error', 'test_urllib2.HandlerTests.test_cookie_redirect', 'test_thread.ThreadRunningTests.test__count', 'test_thread.TestForkInThread.test_forkinthread', 'test_subprocess.POSIXProcessTestCase.test_preexec_errpipe_does_not_double_close_pipes', 'test_ssl.BasicSocketTests.test_parse_cert_CVE_2019_5010', 'test_httplib.HeaderTests.test_headers_debuglevel', 'test_context.ContextTest.test_contextvar_getitem', 'test_context.ContextTest.test_context_var_new_2']
if OSX:
    disabled_tests += ['test_ssl.SimpleBackgroundTests.test_connect_capath', 'test_ssl.SimpleBackgroundTests.test_connect_with_context']
if PYPY:
    disabled_tests += ['test_signal.WakeupSignalTests.test_wakeup_write_error']
if 'thread' in os.getenv('GEVENT_FILE', ''):
    disabled_tests += ['test_subprocess.ProcessTestCase.test_double_close_on_error']
if LIBUV:
    disabled_tests += ['test_signal.InterProcessSignalTests.test_main', 'test_signal.SiginterruptTest.test_siginterrupt_off']
    disabled_tests += ['test_socket.GeneralModuleTests.test_unknown_socket_family_repr', 'test_socket.GeneralModuleTests.test_uknown_socket_family_repr', 'test_selectors.PollSelectorTestCase.test_timeout']
    if OSX:
        disabled_tests += ['test_selectors.PollSelectorTestCase.test_timeout']
    if RUN_COVERAGE:
        disabled_tests += ['test_ftplib.TestFTPClass.test_storlines']
    if sys.platform.startswith('linux'):
        disabled_tests += ['test_asyncore.FileWrapperTest.test_dispatcher']
        if PYPY:
            disabled_tests += ['test_threading.ThreadTests.test_finalize_with_trace', 'test_asyncore.DispatcherWithSendTests_UsePoll.test_send', 'test_asyncore.DispatcherWithSendTests.test_send', 'test_ssl.ContextTests.test__https_verify_envvar', 'test_subprocess.ProcessTestCase.test_check_output', 'test_telnetlib.ReadTests.test_read_eager_A', 'test_urllib2_localnet.TestUrlopen.test_https_with_cafile', 'test_threading.ThreadJoinOnShutdown.test_1_join_on_shutdown', 'test_threading.ThreadingExceptionTests.test_print_exception', 'test_subprocess.ProcessTestCase.test_communicate', 'test_subprocess.ProcessTestCase.test_cwd', 'test_subprocess.ProcessTestCase.test_env', 'test_subprocess.ProcessTestCase.test_stderr_pipe', 'test_subprocess.ProcessTestCase.test_stdout_pipe', 'test_subprocess.ProcessTestCase.test_stdout_stderr_pipe', 'test_subprocess.ProcessTestCase.test_stderr_redirect_with_no_stdout_redirect', 'test_subprocess.ProcessTestCase.test_stdout_filedes_of_stdout', 'test_subprocess.ProcessTestcase.test_stdout_none', 'test_subprocess.ProcessTestcase.test_universal_newlines', 'test_subprocess.ProcessTestcase.test_writes_before_communicate', 'test_subprocess.Win32ProcessTestCase._kill_process', 'test_subprocess.Win32ProcessTestCase._kill_dead_process', 'test_subprocess.Win32ProcessTestCase.test_shell_sequence', 'test_subprocess.Win32ProcessTestCase.test_shell_string', 'test_subprocess.CommandsWithSpaces.with_spaces']
    if WIN:
        disabled_tests += ['test_ssl.ThreadedTests.test_handshake_timeout', 'test_socket.BufferIOTest.testRecvFromIntoBytearray', 'test_socket.BufferIOTest.testRecvFromIntoArray', 'test_socket.BufferIOTest.testRecvIntoArray', 'test_socket.BufferIOTest.testRecvIntoMemoryview', 'test_socket.BufferIOTest.testRecvFromIntoEmptyBuffer', 'test_socket.BufferIOTest.testRecvFromIntoMemoryview', 'test_socket.BufferIOTest.testRecvFromIntoSmallBuffer', 'test_socket.BufferIOTest.testRecvIntoBytearray']
    if PYPY:
        if TRAVIS:
            disabled_tests += ['test_subprocess.ProcessTestCase.test_universal_newlines_communicate']
if RUN_COVERAGE and CFFI_BACKEND:
    disabled_tests += ['test_socket.GeneralModuleTests.test_sendall_interrupted', 'test_socket.TCPTimeoutTest.testInterruptedTimeout', 'test_socketserver.SocketServerTest.test_ForkingUDPServer', 'test_signal.InterProcessSignalTests.test_main']
if PYPY and sys.pypy_version_info[:2] == (7, 3):
    if OSX:
        disabled_tests += ['test_ssl.ThreadedTests.test_alpn_protocols', 'test_ssl.ThreadedTests.test_default_ecdh_curve']
if PYPY3 and TRAVIS:
    disabled_tests += ['test_socket.InheritanceTest.test_SOCK_CLOEXEC']
if PYPY3 and WIN:
    disabled_tests += ['test_socket.GeneralModuleTests.test_socket_fileno', 'test_socket.GeneralModuleTests.test_getaddrinfo_ipv6_scopeid_numeric', 'test_socket.InheritanceTest.test_dup']

def _make_run_with_original(mod_name, func_name):
    if False:
        print('Hello World!')

    @contextlib.contextmanager
    def with_orig():
        if False:
            while True:
                i = 10
        mod = __import__(mod_name)
        now = getattr(mod, func_name)
        from gevent.monkey import get_original
        orig = get_original(mod_name, func_name)
        try:
            setattr(mod, func_name, orig)
            yield
        finally:
            setattr(mod, func_name, now)
    return with_orig

@contextlib.contextmanager
def _gc_at_end():
    if False:
        i = 10
        return i + 15
    try:
        yield
    finally:
        import gc
        gc.collect()
        gc.collect()

@contextlib.contextmanager
def _flaky_socket_timeout():
    if False:
        while True:
            i = 10
    import socket
    try:
        yield
    except socket.timeout:
        flaky.reraiseFlakyTestTimeout()
wrapped_tests = {}

class _PatchedTest(object):

    def __init__(self, test_fqn):
        if False:
            return 10
        self._patcher = wrapped_tests[test_fqn]

    def __call__(self, orig_test_fn):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(orig_test_fn)
        def test(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            with self._patcher():
                return orig_test_fn(*args, **kwargs)
        return test
if OSX:
    disabled_tests += ['test_subprocess.POSIXProcessTestCase.test_run_abort']
if WIN:
    disabled_tests += ['test_ssl.ThreadedTests.test_socketserver', 'test_ssl.ThreadedTests.test_asyncore_server', 'test_socket.NonBlockingTCPTests.testAccept']
    if sys.version_info[:2] <= (3, 9):
        disabled_tests += ['test_context.HamtTest.test_hamt_collision_3', 'test_httplib.BasicTest.test_overflowing_header_limit_after_100']
    wrapped_tests.update({'test_socket.SendfileUsingSendTest.testWithTimeout': _flaky_socket_timeout, 'test_socket.SendfileUsingSendTest.testOffset': _flaky_socket_timeout, 'test_socket.SendfileUsingSendTest.testRegularFile': _flaky_socket_timeout, 'test_socket.SendfileUsingSendTest.testCount': _flaky_socket_timeout})
if PYPY:
    disabled_tests += ['test_subprocess.ProcessTestCase.test_failed_child_execute_fd_leak', 'test_ssl.ThreadedTests.test_compression', 'test_asyncore.TestAPI_UsePoll.test_handle_error', 'test_asyncore.TestAPI_UsePoll.test_handle_read']
    if WIN:
        disabled_tests += ['test_signal.WakeupFDTests.test_invalid_fd', 'test_socket.GeneralModuleTests.test_sock_ioctl']
    disabled_tests += ['test_asyncore.TestAPI_UveIPv6Poll.test_handle_accept', 'test_asyncore.TestAPI_UveIPv6Poll.test_handle_accepted', 'test_asyncore.TestAPI_UveIPv6Poll.test_handle_close', 'test_asyncore.TestAPI_UveIPv6Poll.test_handle_write', 'test_asyncore.TestAPI_UseIPV6Select.test_handle_read', 'test_ssl.ContextTests.test__create_stdlib_context', 'test_ssl.ContextTests.test_create_default_context', 'test_ssl.ContextTests.test_get_ciphers', 'test_ssl.ContextTests.test_options', 'test_ssl.ContextTests.test_constants', 'test_socketserver.SocketServerTest.test_write', 'test_subprocess.ProcessTestcase.test_child_terminated_in_stopped_state', 'test_urllib2_localnet.TestUrlopen.test_https']
disabled_tests += ['test_threading.SubinterpThreadingTests.test_daemon_threads_fatal_error', 'test_threading.ThreadTests.test_tstate_lock', 'test_threading.ThreadTests.test_various_ops', 'test_threading.ThreadTests.test_various_ops_large_stack', 'test_threading.ThreadTests.test_various_ops_small_stack', 'test_subprocess.ProcessTestCase.test_io_buffered_by_default', 'test_subprocess.ProcessTestCase.test_io_unbuffered_works', 'test_subprocess.ProcessTestCase.test_wait_endtime', 'test_subprocess.POSIXProcessTestCase.test_exception_bad_args_0', 'test_subprocess.POSIXProcessTestCase.test_exception_bad_executable', 'test_subprocess.POSIXProcessTestCase.test_exception_cwd', 'test_subprocess.POSIXProcessTestCase.test_exception_errpipe_bad_data', 'test_subprocess.POSIXProcessTestCase.test_exception_errpipe_normal', 'test_subprocess.POSIXProcessTestCase.test_small_errpipe_write_fd', 'test_socket.GeneralModuleTests.test_SocketType_is_socketobject', 'test_socket.GeneralModuleTests.test_dealloc_warn', 'test_socket.GeneralModuleTests.test_repr', 'test_socket.GeneralModuleTests.test_str_for_enums', 'test_socket.GeneralModuleTests.testGetaddrinfo']
if TRAVIS:
    disabled_tests += ['test_subprocess.ProcessTestCase.test_cwd_with_relative_arg', 'test_subprocess.ProcessTestCaseNoPoll.test_cwd_with_relative_arg', 'test_subprocess.ProcessTestCase.test_cwd_with_relative_executable', 'test_subprocess.RunFuncTestCase.test_run_with_shell_timeout_and_capture_output']
disabled_tests += ['test_wsgiref.IntegrationTests.test_interrupted_write']
if PYPY3:
    disabled_tests += ['test_signal.SiginterruptTest.test_siginterrupt_off']
    disabled_tests += ['test_subprocess.POSIXProcessTestCase.test_close_fds_when_max_fd_is_lowered', 'test_ssl.ThreadedTests.test_compression', 'test_ssl.NetworkedBIOTests.test_handshake', 'test_subprocess.ProcessTestCase.test_invalid_env']
    if OSX:
        disabled_tests += ['test_subprocess.POSIXProcessTestCase.test_close_fds', 'test_subprocess.POSIXProcessTestCase.test_close_fds_after_preexec', 'test_subprocess.POSIXProcessTestCase.test_pass_fds', 'test_subprocess.POSIXProcessTestCase.test_pass_fds_inheritable', 'test_subprocess.POSIXProcessTestCase.test_pipe_cloexec', 'test_socket.RecvmsgSCMRightsStreamTest.testCmsgTruncLen0', 'test_socket.RecvmsgSCMRightsStreamTest.testCmsgTruncLen0Plus1', 'test_socket.RecvmsgSCMRightsStreamTest.testCmsgTruncLen1', 'test_socket.RecvmsgSCMRightsStreamTest.testCmsgTruncLen2Minus1', 'test_ssl.ContextTests.test_constructor', 'test_ssl.ContextTests.test_protocol', 'test_ssl.ContextTests.test_session_stats', 'test_ssl.ThreadedTests.test_echo', 'test_ssl.ThreadedTests.test_protocol_sslv23', 'test_ssl.ThreadedTests.test_protocol_sslv3', 'test_ssl.ThreadedTests.test_protocol_tlsv1', 'test_ssl.ThreadedTests.test_protocol_tlsv1_1', 'test_ssl.TestPostHandshakeAuth.test_pha_no_pha_client', 'test_ssl.TestPostHandshakeAuth.test_pha_optional', 'test_ssl.TestPostHandshakeAuth.test_pha_required', 'test_ssl.ThreadedTests.test_npn_protocols', 'test_httpservers.SimpleHTTPServerTestCase.test_undecodable_filename']
    disabled_tests += ['test_threading.ThreadJoinOnShutdown.test_2_join_in_forked_process', 'test_threading.ThreadJoinOnShutdown.test_1_join_in_forked_process']
    if TRAVIS:
        disabled_tests += ['test_threading.ThreadJoinOnShutdown.test_1_join_on_shutdown']
if PYPY:
    wrapped_tests.update({'test_urllib2_localnet.TestUrlopen.test_https_with_cafile': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_command': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_handler': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_head_keep_alive': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_head_via_send_error': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_header_close': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_internal_key_error': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_request_line_trimming': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_return_custom_status': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_return_header_keep_alive': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_send_blank': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_send_error': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_version_bogus': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_version_digits': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_version_invalid': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_version_none': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_version_none_get': _gc_at_end, 'test_httpservers.BaseHTTPServerTestCase.test_get': _gc_at_end, 'test_httpservers.SimpleHTTPServerTestCase.test_get': _gc_at_end, 'test_httpservers.SimpleHTTPServerTestCase.test_head': _gc_at_end, 'test_httpservers.SimpleHTTPServerTestCase.test_invalid_requests': _gc_at_end, 'test_httpservers.SimpleHTTPServerTestCase.test_path_without_leading_slash': _gc_at_end, 'test_httpservers.CGIHTTPServerTestCase.test_invaliduri': _gc_at_end, 'test_httpservers.CGIHTTPServerTestCase.test_issue19435': _gc_at_end, 'test_httplib.TunnelTests.test_connect': _gc_at_end, 'test_httplib.SourceAddressTest.testHTTPConnectionSourceAddress': _gc_at_end, 'test_urllib2_localnet.ProxyAuthTests.test_proxy_with_bad_password_raises_httperror': _gc_at_end, 'test_urllib2_localnet.ProxyAuthTests.test_proxy_with_no_password_raises_httperror': _gc_at_end})
disabled_tests += ['test_subprocess.ProcessTestCase.test_threadsafe_wait', 'test_subprocess.POSIXProcessTestCase.test_preexec_errpipe_does_not_double_close_pipes', 'test_selectors.PollSelectorTestCase.test_above_fd_setsize', 'test_socket.NonBlockingTCPTests.testInitNonBlocking', 'test_socket.NonblockConstantTest.test_SOCK_NONBLOCK', 'test_socket.TestSocketSharing.testShare', 'test_socket.GeneralModuleTests.test_sock_ioctl', 'test_httplib.HeaderTests.test_parse_all_octets']
if OSX:
    disabled_tests += ['test_socket.RecvmsgSCMRightsStreamTest.testFDPassEmpty']
    if TRAVIS:
        disabled_tests += ['test_threading.ThreadTests.test_is_alive_after_fork', 'test_selectors.PollSelectorTestCase.test_timeout']
if TRAVIS:
    disabled_tests += ['test_subprocess.ProcessTestCase.test_double_close_on_error']
disabled_tests += ['test_ssl.ThreadedTests.test_nonblocking_send', 'test_ssl.ThreadedTests.test_socketserver', 'test_socket.GeneralModuleTests.test__sendfile_use_sendfile', 'test_socket.TestExceptions.test_setblocking_invalidfd']
if ARES:
    disabled_tests += ['test_socket.GeneralModuleTests.test_host_resolution', 'test_socket.GeneralModuleTests.test_getnameinfo']
disabled_tests += ['test_threading.MiscTestCase.test__all__']
disabled_tests += ['test_socket.SendfileUsingSendfileTest.testCount', 'test_socket.SendfileUsingSendfileTest.testCountSmall', 'test_socket.SendfileUsingSendfileTest.testCountWithOffset', 'test_socket.SendfileUsingSendfileTest.testOffset', 'test_socket.SendfileUsingSendfileTest.testRegularFile', 'test_socket.SendfileUsingSendfileTest.testWithTimeout', 'test_socket.SendfileUsingSendfileTest.testEmptyFileSend', 'test_socket.SendfileUsingSendfileTest.testNonBlocking', 'test_socket.SendfileUsingSendfileTest.test_errors']
disabled_tests += ['test_socket.GeneralModuleTests.test__sendfile_use_sendfile']
disabled_tests += ['test_socket.LinuxKernelCryptoAPI.test_aead_aes_gcm']
disabled_tests += ['test_subprocess.MiscTests.test_call_keyboardinterrupt_no_kill', 'test_subprocess.MiscTests.test_context_manager_keyboardinterrupt_no_kill', 'test_subprocess.MiscTests.test_run_keyboardinterrupt_no_kill', 'test_socket.NonBlockingTCPTests.testSetBlocking', 'test_ssl.BasicSocketTests.test_private_init', 'test_ssl.ThreadedTests.test_check_hostname_idn', 'test_ssl.SimpleBackgroundTests.test_get_server_certificate', 'test_socket.NetworkConnectionNoServer.test_create_connection', 'test_threading.ThreadTests.test_finalization_shutdown', 'test_threading.ThreadTests.test_shutdown_locks', 'test_threading.ThreadTests.test_old_threading_api', 'test_threading.InterruptMainTests.test_interrupt_main_subthread', 'test_threading.InterruptMainTests.test_interrupt_main_noerror', 'test_ssl.ThreadedTests.test_wrong_cert_tls13']
if APPVEYOR:
    disabled_tests += ['test_selectors.BaseSelectorTestCase.test_timeout']
disabled_tests += ['test_subprocess.RunFuncTestCase.test_run_with_pathlike_path', 'test_subprocess.RunFuncTestCase.test_bufsize_equal_one_binary_mode', 'test_threading.ExceptHookTests.test_excepthook_thread_None']
if sys.version_info[:3] < (3, 8, 1):
    disabled_tests += ['test_ssl.BasicSocketTests.test_parse_all_sans', 'test_ssl.BasicSocketTests.test_parse_cert_CVE_2013_4238']
if sys.version_info[:3] < (3, 8, 10):
    disabled_tests += ['test_ftplib.TestFTPClass.test_makepasv_issue43285_security_disabled', 'test_ftplib.TestFTPClass.test_makepasv_issue43285_security_enabled_default', 'test_httplib.BasicTest.test_dir_with_added_behavior_on_status', 'test_httplib.TunnelTests.test_tunnel_connect_single_send_connection_setup', 'test_ssl.TestSSLDebug.test_msg_callback_deadlock_bpo43577', 'test_ssl.ContextTests.test_load_verify_cadata', 'test_ftplib.TestTLS_FTPClassMixin.test_retrbinary_rest']
if RESOLVER_DNSPYTHON:
    disabled_tests += ['test_socket.GeneralModuleTests.test_getaddrinfo_ipv6_scopeid_symbolic']
disabled_tests += ['test_ssl.BasicSocketTests.test_openssl_version']
if OSX:
    disabled_tests += ['test_socket.RecvmsgIntoTCPTest.testRecvmsgIntoGenerator', 'test_ftp.TestTLS_FTPClassMixin.test_mlsd', 'test_ftp.TestTLS_FTPClassMixin.test_retrlines_too_long', 'test_ftp.TestTLS_FTPClassMixin.test_storlines', 'test_ftp.TestTLS_FTPClassMixin.test_retrbinary_rest']
    if RESOLVER_ARES and PY38 and (not RUNNING_ON_CI):
        disabled_tests += ['test_socket.GeneralModuleTests.test_getaddrinfo_ipv6_scopeid_symbolic']
if PY39:
    disabled_tests += ['test_subprocess.ProcessTestCase.test_repr', 'test_subprocess.POSIXProcessTestTest.test_send_signal_race']
    if sys.version_info[:3] < (3, 9, 5):
        disabled_tests += ['test_ftplib.TestFTPClass.test_makepasv_issue43285_security_disabled', 'test_ftplib.TestFTPClass.test_makepasv_issue43285_security_enabled_default', 'test_httplib.BasicTest.test_dir_with_added_behavior_on_status', 'test_httplib.TunnelTests.test_tunnel_connect_single_send_connection_setup', 'test_ssl.TestSSLDebug.test_msg_callback_deadlock_bpo43577', 'test_ssl.ContextTests.test_load_verify_cadata', 'test_ftplib.TestTLS_FTPClassMixin.test_retrbinary_rest', 'test_ftplib.TestTLS_FTPClassMixin.test_retrlines_too_long']
if PY310:
    disabled_tests += ['test_select.SelectTestCase.test_disallow_instantiation', 'test_threading.ThreadTests.test_disallow_instantiation', 'test_threading.InterruptMainTests.test_can_interrupt_tight_loops', 'test_subprocess.ProcessTestCase.test_pipesize_default', 'test_subprocess.ProcessTestCase.test_pipesizes', 'test_signal.SiginterruptTest.test_siginterrupt_off']
    if TRAVIS:
        disabled_tests += ['test_threading.SubinterpThreadingTests.test_threads_join', 'test_threading.SubinterpThreadingTests.test_threads_join_2']
if PY311:
    disabled_tests += ['test_signal.GenericTests.test_functions_module_attr', 'test_subprocess.ProcessTestCase.test__use_vfork']
if PY312:
    if RUN_COVERAGE:
        disabled_tests += ['test_threading.ThreadTests.test_gettrace_all_threads']
    if WIN:
        disabled_tests += ['test_socket.BasicHyperVTest.testCreateHyperVSocketAddrNotTupleFailure', 'test_socket.BasicHyperVTest.testCreateHyperVSocketAddrServiceIdNotValidUUIDFailure', 'test_socket.BasicHyperVTest.testCreateHyperVSocketAddrVmIdNotValidUUIDFailure']
if TRAVIS:
    disabled_tests += ['test_ssl.ContextTests.test_options', 'test_ssl.ThreadedTests.test_alpn_protocols', 'test_ssl.ThreadedTests.test_default_ecdh_curve', 'test_ssl.ThreadedTests.test_shared_ciphers']
if RUNNING_ON_MUSLLINUX:
    disabled_tests += ['test_threading.ThreadingExceptionTests.test_recursion_limit']

def _build_test_structure(sequence_of_tests):
    if False:
        while True:
            i = 10
    _disabled_tests = frozenset(sequence_of_tests)
    disabled_tests_by_file = collections.defaultdict(set)
    for file_case_meth in _disabled_tests:
        (file_name, _case, _meth) = file_case_meth.split('.')
        by_file = disabled_tests_by_file[file_name]
        by_file.add(file_case_meth)
    return disabled_tests_by_file
_disabled_tests_by_file = _build_test_structure(disabled_tests)
_wrapped_tests_by_file = _build_test_structure(wrapped_tests)

def disable_tests_in_source(source, filename):
    if False:
        i = 10
        return i + 15
    if filename.startswith('./'):
        filename = filename[2:]
    if filename.endswith('.py'):
        filename = filename[:-3]
    my_disabled_tests = _disabled_tests_by_file.get(filename, ())
    my_wrapped_tests = _wrapped_tests_by_file.get(filename, {})
    if my_disabled_tests or my_wrapped_tests:
        pattern = '^import .*'
        replacement = 'from gevent.testing import patched_tests_setup as _GEVENT_PTS;'
        replacement += 'import unittest as _GEVENT_UTS;'
        replacement += '\\g<0>'
        (source, n) = re.subn(pattern, replacement, source, 1, re.MULTILINE)
        print('Added imports', n)
    my_disabled_testcases = set()
    for test in my_disabled_tests:
        testcase = test.split('.')[-1]
        my_disabled_testcases.add(testcase)
        pattern = '^([ \\t]+)def ' + testcase
        replacement = "\\1@_GEVENT_UTS.skip('Removed by patched_tests_setup: %s')\\n" % (test,)
        replacement += '\\g<0>'
        (source, n) = re.subn(pattern, replacement, source, 0, re.MULTILINE)
        print('Skipped %s (%d)' % (testcase, n), file=sys.stderr)
    for test in my_wrapped_tests:
        testcase = test.split('.')[-1]
        if testcase in my_disabled_testcases:
            print('Not wrapping %s because it is skipped' % (test,))
            continue
        pattern = '^([ \\t]+)def ' + testcase
        replacement = "\\1@_GEVENT_PTS._PatchedTest('%s')\\n" % (test,)
        replacement += '\\g<0>'
        (source, n) = re.subn(pattern, replacement, source, 0, re.MULTILINE)
        print('Wrapped %s (%d)' % (testcase, n), file=sys.stderr)
    return source
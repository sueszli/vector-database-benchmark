from sslyze import ServerNetworkLocation, ServerScanRequest, ServerTlsProbingResult
from sslyze.errors import ConnectionToServerFailed
from sslyze.scanner._mass_connectivity_tester import MassConnectivityTester

class TestMassConnectivityTester:

    def test(self):
        if False:
            while True:
                i = 10
        reachable_request1 = ServerScanRequest(server_location=ServerNetworkLocation(hostname='www.google.com', port=443))
        reachable_request2 = ServerScanRequest(server_location=ServerNetworkLocation(hostname='www.cloudflare.com', port=443))
        non_reachable_request1 = ServerScanRequest(server_location=ServerNetworkLocation(hostname='localhost', port=12345))
        non_reachable_request2 = ServerScanRequest(server_location=ServerNetworkLocation(hostname='localhost', port=54321))
        completed_request_uuids = set()

        def server_connectivity_test_completed_callback(server_scan_request: ServerScanRequest, tls_probing_result: ServerTlsProbingResult) -> None:
            if False:
                print('Hello World!')
            completed_request_uuids.add(server_scan_request.uuid)
        error_request_uuids = set()

        def server_connectivity_test_error_callback(server_scan_request: ServerScanRequest, connectivity_error: ConnectionToServerFailed) -> None:
            if False:
                print('Hello World!')
            error_request_uuids.add(server_scan_request.uuid)
        tester = MassConnectivityTester(concurrent_server_scans_count=3)
        tester.start_work([reachable_request1, reachable_request2, non_reachable_request1, non_reachable_request2])
        assert tester.has_started_work
        tester.wait_until_all_work_was_processed(server_connectivity_test_completed_callback=server_connectivity_test_completed_callback, server_connectivity_test_error_callback=server_connectivity_test_error_callback)
        assert len(completed_request_uuids) == 2
        assert {reachable_request1.uuid, reachable_request2.uuid} == completed_request_uuids
        assert len(error_request_uuids) == 2
        assert {non_reachable_request1.uuid, non_reachable_request2.uuid} == error_request_uuids
        tester._scan_requests_queue.join()
        tester._results_queue.join()
        for thread in tester._all_worker_threads:
            assert not thread.is_alive()
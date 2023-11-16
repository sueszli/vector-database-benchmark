import ray
from ray._private import test_utils

@test_utils.wait_for_stdout(strings_to_match=['Adding 1 node(s) of type small-group.'], timeout_s=15)
def main():
    if False:
        for i in range(10):
            print('nop')
    'Submits CPU request.\n    Wait 15 sec for autoscaler scale-up event to get emitted to stdout.\n\n    The autoscaler update interval is 5 sec, so it should be enough to wait 5 seconds.\n    An extra ten seconds are added to the timeout as a generous buffer against\n    flakiness.\n    '
    ray.autoscaler.sdk.request_resources(num_cpus=2)
if __name__ == '__main__':
    ray.init('auto')
    main()
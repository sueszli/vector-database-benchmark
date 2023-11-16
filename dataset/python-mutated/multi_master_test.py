import argparse
import os
from time import sleep, perf_counter
import json
import ray

def test_max_actors_launch(cpus_per_actor, total_actors, num_masters):
    if False:
        while True:
            i = 10
    num_slaves_per_master = total_actors / num_masters - 1

    @ray.remote(num_cpus=cpus_per_actor)
    class Actor:

        def foo(self):
            if False:
                print('Hello World!')
            pass

        def create(self):
            if False:
                print('Hello World!')
            return [Actor.options(max_restarts=-1).remote() for _ in range(num_slaves_per_master)]
    print('Start launch actors')
    actors = [Actor.options(max_restarts=-1, scheduling_strategy='SPREAD').remote() for _ in range(num_masters)]
    slaves_per_master = []
    for master in actors:
        slaves_per_master.append(master.create.remote())
    for slaves in slaves_per_master:
        actors.extend(ray.get(slaves))
    return actors

def test_actor_ready(actors):
    if False:
        for i in range(10):
            print('nop')
    remaining = [actor.foo.remote() for actor in actors]
    ray.get(remaining)

def parse_script_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus-per-actor', type=float, default=0.2)
    parser.add_argument('--total-actors', type=int, default=5000)
    parser.add_argument('--num-masters', type=int, default=50)
    parser.add_argument('--no-report', default=False, action='store_true')
    parser.add_argument('--fail', default=False, action='store_true')
    return parser.parse_known_args()

def main():
    if False:
        while True:
            i = 10
    (args, unknown) = parse_script_args()
    ray.init(address='auto')
    actor_launch_start = perf_counter()
    actors = test_max_actors_launch(args.cpus_per_actor, args.total_actors, args.num_masters)
    actor_launch_end = perf_counter()
    actor_launch_time = actor_launch_end - actor_launch_start
    if args.fail:
        sleep(10)
        return
    actor_ready_start = perf_counter()
    test_actor_ready(actors)
    actor_ready_end = perf_counter()
    actor_ready_time = actor_ready_end - actor_ready_start
    print(f'Actor launch time: {actor_launch_time} ({args.total_actors} actors)')
    print(f'Actor ready time: {actor_ready_time} ({args.total_actors} actors)')
    print(f'Total time: {actor_launch_time + actor_ready_time} ({args.total_actors} actors)')
    if 'TEST_OUTPUT_JSON' in os.environ and (not args.no_report):
        out_file = open(os.environ['TEST_OUTPUT_JSON'], 'w')
        results = {'actor_launch_time': actor_launch_time, 'actor_ready_time': actor_ready_time, 'total_time': actor_launch_time + actor_ready_time, 'num_actors': args.total_actors, 'success': '1'}
        json.dump(results, out_file)
if __name__ == '__main__':
    main()
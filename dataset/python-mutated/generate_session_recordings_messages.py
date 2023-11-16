import argparse
import json
import uuid
from sys import stderr, stdout
from typing import List
import numpy
from faker import Faker
help = 'Generate Kafka messages that simulate session recording data'

def get_parser():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('--count', type=int, default=100, help='The number of session recordings to generate')
    parser.add_argument('--full-snapshot-size-mean', type=int, default=185000, help='The average size of a full snapshot in bytes')
    parser.add_argument('--full-snapshot-size-standard-deviation', type=int, default=160000, help='The standard deviation of the size of a full snapshot in bytes squared')
    parser.add_argument('--full-snapshot-count-mean', type=int, default=2, help='The average number of full snapshots per session')
    parser.add_argument('--full-snapshot-count-standard-deviation', type=int, default=4, help='The standard deviation of the number of full snapshots per session')
    parser.add_argument('--incremental-snapshot-size-mean', type=int, default=14000, help='The average size of an incremental snapshot in bytes')
    parser.add_argument('--incremental-snapshot-size-standard-deviation', type=int, default=50000, help='The standard deviation of the size of an incremental snapshot in bytes squared')
    parser.add_argument('--incremental-snapshot-count-mean', type=int, default=40, help='The average number of incremental snapshots per session')
    parser.add_argument('--incremental-snapshot-count-standard-deviation', type=int, default=214, help='The standard deviation of the number of incremental snapshots per session')
    parser.add_argument('--seed-value', type=int, default=0, help='The seed value to use for the random number generator')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--team-id', type=int, help='The team id to use for the messages.')
    parser.add_argument('--token', type=str, help='The token to use for the messages.')
    return parser

def chunked(data: str, chunk_size: int) -> List[str]:
    if False:
        print('Hello World!')
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def sample_log_normal_distribution(mu: int, sigma: int, count: int):
    if False:
        i = 10
        return i + 15
    '\n    Samples from a log-normal distribution with the given mean and standard\n    deviation of that log distribution.\n    '
    normal_std = numpy.sqrt(numpy.log(1 + (sigma / mu) ** 2))
    normal_mean = numpy.log(mu) - normal_std ** 2 / 2
    return [int(sample) for sample in numpy.random.lognormal(normal_mean, normal_std, count)]

def generate_snapshot_messages(faker: Faker, count: int, full_snapshot_size_mean: int, full_snapshot_size_standard_deviation: int, full_snapshot_count_mean: int, full_snapshot_count_standard_deviation: int, incremental_snapshot_size_mean: int, incremental_snapshot_size_standard_deviation: int, incremental_snapshot_count_mean: int, incremental_snapshot_count_standard_deviation: int, team_id: int, token: str, verbose: bool):
    if False:
        print('Hello World!')
    full_snapshot_count_samples = sample_log_normal_distribution(full_snapshot_count_mean, full_snapshot_count_standard_deviation, count)
    incremental_snapshot_count_samples = sample_log_normal_distribution(incremental_snapshot_count_mean, incremental_snapshot_count_standard_deviation, count)
    full_snapshot_size_samples = sample_log_normal_distribution(full_snapshot_size_mean, full_snapshot_size_standard_deviation, max(full_snapshot_count_samples))
    incremental_snapshot_size_samples = sample_log_normal_distribution(incremental_snapshot_size_mean, incremental_snapshot_size_standard_deviation, max(incremental_snapshot_count_samples))
    now = faker.date_time()
    sent_at = faker.date_time()
    ip = faker.ipv4()
    site_url = faker.url()
    for (full_snapshot_count, incremental_snapshot_count) in zip(full_snapshot_count_samples, incremental_snapshot_count_samples):
        session_id = str(uuid.uuid4())
        distinct_id = str(uuid.uuid4())
        if verbose:
            stderr.write(f'Generating session recording messages for session {session_id} with an average of {full_snapshot_count_mean} full snapshots with a standard deviation of {full_snapshot_count_standard_deviation} and an average of {incremental_snapshot_count_mean} incremental snapshots with a standard deviation of {incremental_snapshot_count_standard_deviation}\n')
        if verbose:
            stderr.write(f'Generating session recording messages for session {session_id} with {full_snapshot_count} full snapshots and {incremental_snapshot_count} incremental snapshots/n')
        for (full_snapshot_index, full_snapshot_size) in enumerate(full_snapshot_size_samples[:full_snapshot_count]):
            full_snapshot_data = faker.pystr(min_chars=full_snapshot_size, max_chars=full_snapshot_size)
            full_snapshot_data_chunks = chunked(full_snapshot_data, 900000)
            for (chunk_index, chunk) in enumerate(full_snapshot_data_chunks):
                chunk_id = str(uuid.uuid4())
                chunk_count = len(full_snapshot_data_chunks)
                snapshot_data = {'chunk_id': chunk_id, 'chunk_index': chunk_index, 'chunk_count': chunk_count, 'data': chunk, 'compression': 'gzip-base64', 'has_full_snapshot': full_snapshot_index == 0}
                data = {'event': '$snapshot', 'properties': {'distinct_id': distinct_id, 'session_id': session_id, 'window_id': session_id, 'snapshot_data': snapshot_data}}
                message = {'uuid': str(uuid.uuid4()), 'distinct_id': distinct_id, 'ip': ip, 'site_url': site_url, 'data': json.dumps(data), 'team_id': team_id, 'now': now.isoformat(), 'sent_at': sent_at.isoformat(), 'token': token}
                stdout.write(json.dumps(message))
                stdout.write('\n')
        for incremental_snapshot_size in incremental_snapshot_size_samples[:incremental_snapshot_count]:
            incremental_snapshot_data = faker.pystr(min_chars=incremental_snapshot_size, max_chars=incremental_snapshot_size)
            incremental_snapshot_data_chunks = chunked(incremental_snapshot_data, 900000)
            for (chunk_index, chunk) in enumerate(incremental_snapshot_data_chunks):
                chunk_id = str(uuid.uuid4())
                chunk_count = len(incremental_snapshot_data_chunks)
                snapshot_data = {'chunk_id': chunk_id, 'chunk_index': chunk_index, 'chunk_count': chunk_count, 'data': chunk, 'compression': 'gzip-base64', 'has_full_snapshot': False}
                data = {'event': '$snapshot', 'properties': {'distinct_id': distinct_id, 'session_id': session_id, 'window_id': session_id, 'snapshot_data': snapshot_data}}
                message = {'uuid': str(uuid.uuid4()), 'distinct_id': distinct_id, 'ip': ip, 'site_url': site_url, 'data': json.dumps(data), 'team_id': team_id, 'now': now.isoformat(), 'sent_at': sent_at.isoformat(), 'token': token}
                stdout.write(json.dumps(message))
                stdout.write('\n')

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse the command line arguments using `get_parser`, generate the snapshot messages, and print\n    them out to stdout as a single JSON object per line. We also initialize\n    Faker and numpy to ensure that the random number generator is seeded with a\n    constant.\n    '
    parser = get_parser()
    args = parser.parse_args()
    Faker.seed(args.seed_value)
    faker = Faker()
    numpy.random.seed(args.seed_value)
    generate_snapshot_messages(faker=faker, count=args.count, full_snapshot_size_mean=args.full_snapshot_size_mean, full_snapshot_size_standard_deviation=args.full_snapshot_size_standard_deviation, full_snapshot_count_mean=args.full_snapshot_count_mean, full_snapshot_count_standard_deviation=args.full_snapshot_count_standard_deviation, incremental_snapshot_size_mean=args.incremental_snapshot_size_mean, incremental_snapshot_size_standard_deviation=args.incremental_snapshot_size_standard_deviation, incremental_snapshot_count_mean=args.incremental_snapshot_count_mean, incremental_snapshot_count_standard_deviation=args.incremental_snapshot_count_standard_deviation, team_id=args.team_id, token=args.token, verbose=args.verbose)
if __name__ == '__main__':
    main()
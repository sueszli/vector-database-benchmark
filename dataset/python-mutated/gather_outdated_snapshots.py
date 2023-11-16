import datetime
import json
import os
import click

def get_outdated_snapshots_for_directory(path: str, date_limit: str, check_sub_directories: bool=True, combine_parametrized=True, show_date=False) -> dict:
    if False:
        return 10
    '\n    Fetches all snapshots that were recorded before the given date_limit\n    :param path: The directory where to look for snapshot files.\n    :param date_limit: All snapshots whose recorded-date is older than date-limit are considered outdated.\n            Format of the date-string must be "DD-MM-YYYY".\n    :param check_sub_directories: Whether to look for snapshots in subdirectories\n    :param combine_parametrized: Whether to combine versions of the same test and treat them as the same or not\n    :return: List of test names whose snapshots (if any) are outdated.\n    '
    result = {'date': date_limit}
    date_limit = datetime.datetime.strptime(date_limit, '%d-%m-%Y')
    outdated_snapshots = {}

    def do_get_outdated_snapshots(path: str):
        if False:
            i = 10
            return i + 15
        if not path.endswith('/'):
            path = f'{path}/'
        for file in os.listdir(path):
            if os.path.isdir(f'{path}{file}') and check_sub_directories:
                do_get_outdated_snapshots(f'{path}{file}')
            elif file.endswith('.snapshot.json'):
                with open(f'{path}{file}') as f:
                    json_content: dict = json.load(f)
                    for (name, recorded_snapshot_data) in json_content.items():
                        recorded_date = recorded_snapshot_data.get('recorded-date')
                        date = datetime.datetime.strptime(recorded_date, '%d-%m-%Y, %H:%M:%S')
                        if date < date_limit:
                            outdated_snapshot_data = dict()
                            if show_date:
                                outdated_snapshot_data['recorded-date'] = recorded_date
                            if combine_parametrized:
                                name = name.split('[')[0]
                            outdated_snapshots[name] = outdated_snapshot_data
    do_get_outdated_snapshots(path)
    result['count'] = len(outdated_snapshots)
    result['outdated_snapshots'] = outdated_snapshots
    return result

@click.command()
@click.argument('path', type=str, required=True)
@click.argument('date_limit', type=str, required=True)
@click.option('--check-sub-dirs', type=bool, required=False, default=True, help='Whether to check sub directories of PATH too')
@click.option('--combine-parametrized', type=bool, required=False, default=True, help='If True, parametrized snapshots are treated as one')
@click.option('--show-date', type=bool, required=False, default=False, help='Should tests have their recording date attached?')
def get_snapshots(path: str, date_limit: str, check_sub_dirs, combine_parametrized, show_date):
    if False:
        print('Hello World!')
    '\n    Fetches all snapshots in PATH that were recorded before the given DATE_LIMIT.\n    Format of the DATE_LIMIT-string must be "DD-MM-YYYY".\n\n    Returns a JSON with the relevant information\n\n    \x08\n    Example usage:\n    python gather_outdated_snapshots.py ../tests/integration 24-12-2022 | jq .\n    '
    snapshots = get_outdated_snapshots_for_directory(path, date_limit, check_sub_dirs, combine_parametrized, show_date)
    snapshots['outdated_snapshots'] = dict(sorted(snapshots['outdated_snapshots'].items()))
    join = ' '.join(snapshots['outdated_snapshots'])
    snapshots['pytest_executable_list'] = join
    print(json.dumps(snapshots, default=str))
if __name__ == '__main__':
    get_snapshots()
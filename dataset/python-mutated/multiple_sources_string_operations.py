def user_controlled_input():
    if False:
        while True:
            i = 10
    return 'evil'
global_query: str = 'SELECT'

class DatabaseSchemaEditor:
    attribute_query: str = 'SELECT'

    def string_operations(self, arg) -> None:
        if False:
            i = 10
            return i + 15
        user_controlled: str = user_controlled_input()
        self.attribute_query.format(user_controlled)

def string_operations(arg) -> None:
    if False:
        print('Hello World!')
    query: str = 'SELECT'
    user_controlled: str = user_controlled_input()
    query.format(user_controlled)
    query.format(data=user_controlled)
    query + user_controlled
    user_controlled + query
    query % user_controlled
    global_query.format(user_controlled)
    DatabaseSchemaEditor.attribute_query.format(user_controlled)
global_query = ''

def format_string_issue_string_literal():
    if False:
        for i in range(10):
            print('nop')
    user_controlled = user_controlled_input()
    f'SELECT{user_controlled}'

def format_string_multiple_issues_string_literal():
    if False:
        for i in range(10):
            print('nop')
    user_controlled = user_controlled_input()
    f'SELECT{user_controlled}'
    f'SELECT{user_controlled}'

def format_string_issue():
    if False:
        while True:
            i = 10
    query: str = 'SELECT'
    user_controlled = user_controlled_input()
    f'{query}{user_controlled}'
    x = 0
    f'{query}{user_controlled}{x}'

def format_string_triggered_user_controlled(arg):
    if False:
        print('Hello World!')
    query: str = 'SELECT'
    f'{query}{arg}'

def format_string_issue_with_triggered_user_controlled():
    if False:
        return 10
    user_controlled = user_controlled_input()
    format_string_triggered_user_controlled(user_controlled)

def format_string_triggered_sql(arg):
    if False:
        while True:
            i = 10
    user_controlled = user_controlled_input()
    f'{user_controlled}{arg}'

def format_string_issue_with_triggered_sql():
    if False:
        return 10
    query: str = 'SELECT'
    format_string_triggered_sql(query)

def format_string_multiple_triggered_user_controlled(arg1, arg2):
    if False:
        i = 10
        return i + 15
    f'{arg1} SELECT {arg2}'

def format_string_issue_with_multiple_triggered_user_controlled():
    if False:
        for i in range(10):
            print('nop')
    user_controlled = user_controlled_input()
    format_string_multiple_triggered_user_controlled(user_controlled, 0)
    format_string_multiple_triggered_user_controlled(0, user_controlled)

def nested_stradd_and_fstring():
    if False:
        return 10
    x: str = user_controlled_input()
    y = 'xyz'
    return 'abc' + f'{x + y}'

def stradd_triggered_user_controlled(arg):
    if False:
        for i in range(10):
            print('nop')
    x: str = user_controlled_input()
    x + arg.f

def test_large_string_add():
    if False:
        return 10
    db_dir = '/mnt'
    wal_dir = '/mnt'
    key_size = 1
    value_size = 2
    block_size = 10
    cache_size = 1
    M = 1
    G = 2
    K = 3
    const_params = ' --db=' + str(db_dir) + ' --wal_dir=' + str(wal_dir) + ' --num_levels=' + str(6) + ' --key_size=' + str(key_size) + ' --value_size=' + str(value_size) + ' --block_size=' + str(block_size) + ' --cache_size=' + str(cache_size) + ' --cache_numshardbits=' + str(6) + ' --compression_type=' + str('snappy') + ' --compression_ratio=' + str(0.5) + ' --write_buffer_size=' + str(int(128 * M)) + ' --max_write_buffer_number=' + str(2) + ' --target_file_size_base=' + str(int(128 * M)) + ' --max_bytes_for_level_base=' + str(int(1 * G)) + ' --sync=' + str(0) + ' --verify_checksum=' + str(1) + ' --delete_obsolete_files_period_micros=' + str(int(60 * M)) + ' --statistics=' + str(1) + ' --stats_per_interval=' + str(1) + ' --stats_interval=' + str(int(1 * M)) + ' --histogram=' + str(1) + ' --memtablerep=' + str('skip_list') + ' --bloom_bits=' + str(10) + ' --open_files=' + str(int(20 * K))
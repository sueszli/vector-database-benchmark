import sys
import pytest
from textwrap import dedent
from chalice import analyzer
from chalice.analyzer import Boto3ModuleType, Boto3CreateClientType
from chalice.analyzer import Boto3ClientType, Boto3ClientMethodType
from chalice.analyzer import Boto3ClientMethodCallType
from chalice.analyzer import FunctionType

def aws_calls(source_code):
    if False:
        return 10
    real_source_code = dedent(source_code)
    calls = analyzer.get_client_calls(real_source_code)
    return calls

def chalice_aws_calls(source_code):
    if False:
        i = 10
        return i + 15
    real_source_code = dedent(source_code)
    calls = analyzer.get_client_calls_for_app(real_source_code)
    return calls

def known_types_for_module(source_code):
    if False:
        return 10
    real_source_code = dedent(source_code)
    compiled = analyzer.parse_code(real_source_code)
    t = analyzer.SymbolTableTypeInfer(compiled)
    t.bind_types()
    known = t.known_types()
    return known

def known_types_for_function(source_code, name):
    if False:
        while True:
            i = 10
    real_source_code = dedent(source_code)
    compiled = analyzer.parse_code(real_source_code)
    t = analyzer.SymbolTableTypeInfer(compiled)
    t.bind_types()
    known = t.known_types(scope_name=name)
    return known

def test_can_analyze_chalice_app():
    if False:
        while True:
            i = 10
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        app = Chalice(app_name='james1')\n        ec2 = boto3.client('ec2')\n\n\n        @app.route('/')\n        def index():\n            ec2.describe_instances()\n            return {}\n    ") == {'ec2': set(['describe_instances'])}

def test_inferred_module_type():
    if False:
        print('Hello World!')
    assert known_types_for_module('        import boto3\n        import os\n        a = 1\n    ') == {'boto3': Boto3ModuleType()}

def test_recursive_function_none():
    if False:
        return 10
    assert aws_calls('        def recursive_function():\n            recursive_function()\n        recursive_function()\n    ') == {}

def test_recursive_comprehension_none():
    if False:
        print('Hello World!')
    assert aws_calls('        xs = []\n        def recursive_function():\n            [recursive_function() for x in xs]\n        recursive_function()\n    ') == {}

def test_recursive_function_client_calls():
    if False:
        print('Hello World!')
    assert aws_calls("        import boto3\n        def recursive_function():\n            recursive_function()\n            boto3.client('ec2').describe_instances()\n        recursive_function()\n    ") == {'ec2': set(['describe_instances'])}

def test_mutual_recursion():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3\n        ec2 = boto3.client('ec2')\n\n        def a():\n            b()\n            ec2.run_instances()\n\n\n        def b():\n            ec2.describe_instances()\n            a()\n        a()\n    ") == {'ec2': set(['describe_instances', 'run_instances'])}

def test_inferred_module_type_tracks_assignment():
    if False:
        for i in range(10):
            print('nop')
    assert known_types_for_module('        import boto3\n        a = boto3\n    ') == {'boto3': Boto3ModuleType(), 'a': Boto3ModuleType()}

def test_inferred_module_type_tracks_multi_assignment():
    if False:
        for i in range(10):
            print('nop')
    assert known_types_for_module('        import boto3\n        a = b = c = boto3\n    ') == {'boto3': Boto3ModuleType(), 'a': Boto3ModuleType(), 'b': Boto3ModuleType(), 'c': Boto3ModuleType()}

def test_inferred_client_create_type():
    if False:
        return 10
    assert known_types_for_module('        import boto3\n        a = boto3.client\n    ') == {'boto3': Boto3ModuleType(), 'a': Boto3CreateClientType()}

def test_inferred_client_type():
    if False:
        print('Hello World!')
    assert known_types_for_module("        import boto3\n        a = boto3.client('ec2')\n    ") == {'boto3': Boto3ModuleType(), 'a': Boto3ClientType('ec2')}

def test_inferred_client_type_each_part():
    if False:
        return 10
    assert known_types_for_module("        import boto3\n        a = boto3.client\n        b = a('ec2')\n    ") == {'boto3': Boto3ModuleType(), 'a': Boto3CreateClientType(), 'b': Boto3ClientType('ec2')}

def test_infer_client_method():
    if False:
        return 10
    assert known_types_for_module("        import boto3\n        a = boto3.client('ec2').describe_instances\n    ") == {'boto3': Boto3ModuleType(), 'a': Boto3ClientMethodType('ec2', 'describe_instances')}

def test_infer_client_method_called():
    if False:
        while True:
            i = 10
    assert known_types_for_module("        import boto3\n        a = boto3.client('ec2').describe_instances()\n    ") == {'boto3': Boto3ModuleType(), 'a': Boto3ClientMethodCallType('ec2', 'describe_instances')}

def test_infer_type_on_function_scope():
    if False:
        i = 10
        return i + 15
    assert known_types_for_function("        import boto3\n        def foo():\n            d = boto3.client('dynamodb')\n            e = d.list_tables()\n        foo()\n    ", name='foo') == {'d': Boto3ClientType('dynamodb'), 'e': Boto3ClientMethodCallType('dynamodb', 'list_tables')}

def test_can_understand_return_types():
    if False:
        i = 10
        return i + 15
    assert known_types_for_module("        import boto3\n        def create_client():\n            d = boto3.client('dynamodb')\n            return d\n        e = create_client()\n    ") == {'boto3': Boto3ModuleType(), 'create_client': FunctionType(Boto3ClientType('dynamodb')), 'e': Boto3ClientType('dynamodb')}

def test_type_equality():
    if False:
        while True:
            i = 10
    assert Boto3ModuleType() == Boto3ModuleType()
    assert Boto3CreateClientType() == Boto3CreateClientType()
    assert Boto3ModuleType() != Boto3CreateClientType()
    assert Boto3ClientType('s3') == Boto3ClientType('s3')
    assert Boto3ClientType('s3') != Boto3ClientType('ec2')
    assert Boto3ClientType('s3') == Boto3ClientType('s3')
    assert Boto3ClientMethodType('s3', 'list_objects') == Boto3ClientMethodType('s3', 'list_objects')
    assert Boto3ClientMethodType('ec2', 'describe_instances') != Boto3ClientMethodType('s3', 'list_object')
    assert Boto3ClientMethodType('ec2', 'describe_instances') != Boto3CreateClientType()

def test_single_call():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        d.list_tables()\n    ") == {'dynamodb': set(['list_tables'])}

def test_multiple_calls():
    if False:
        print('Hello World!')
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        d.list_tables()\n        d.create_table(TableName='foobar')\n    ") == {'dynamodb': set(['list_tables', 'create_table'])}

def test_multiple_services():
    if False:
        i = 10
        return i + 15
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        asdf = boto3.client('s3')\n        d.list_tables()\n        asdf.get_object(Bucket='foo', Key='bar')\n        d.create_table(TableName='foobar')\n    ") == {'dynamodb': set(['list_tables', 'create_table']), 's3': set(['get_object'])}

def test_basic_aliasing():
    if False:
        print('Hello World!')
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        alias = d\n        alias.list_tables()\n    ") == {'dynamodb': set(['list_tables'])}

def test_multiple_aliasing():
    if False:
        i = 10
        return i + 15
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        alias = d\n        alias2 = alias\n        alias3 = alias2\n        alias3.list_tables()\n    ") == {'dynamodb': set(['list_tables'])}

def test_multiple_aliasing_non_chained():
    if False:
        print('Hello World!')
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        alias = d\n        alias2 = alias\n        alias3 = alias\n        alias3.list_tables()\n    ") == {'dynamodb': set(['list_tables'])}

def test_no_calls_found():
    if False:
        while True:
            i = 10
    assert aws_calls('        import boto3\n    ') == {}

def test_original_name_replaced():
    if False:
        return 10
    assert aws_calls("        import boto3\n        import some_other_thing\n        d = boto3.client('dynamodb')\n        d.list_tables()\n        d = some_other_thing\n        d.create_table()\n    ") == {'dynamodb': set(['list_tables'])}

def test_multiple_targets():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3\n        a = b = boto3.client('dynamodb')\n        b.list_tables()\n        a.create_table()\n    ") == {'dynamodb': set(['create_table', 'list_tables'])}

def test_in_function():
    if False:
        while True:
            i = 10
    assert aws_calls("        import boto3\n        def foo():\n            d = boto3.client('dynamodb')\n            d.list_tables()\n        foo()\n    ") == {'dynamodb': set(['list_tables'])}

def test_ignores_built_in_scope():
    if False:
        return 10
    assert aws_calls("        import boto3\n        a = boto3.client('dynamodb')\n        def foo():\n            if a is not None:\n                try:\n                    a.list_tables()\n                except Exception as e:\n                    a.create_table()\n        foo()\n    ") == {'dynamodb': set(['create_table', 'list_tables'])}

def test_understands_scopes():
    if False:
        print('Hello World!')
    assert aws_calls("        import boto3, mock\n        d = mock.Mock()\n        def foo():\n            d = boto3.client('dynamodb')\n        d.list_tables()\n    ") == {}

def test_function_return_types():
    if False:
        print('Hello World!')
    assert aws_calls("        import boto3\n        def create_client():\n            return boto3.client('dynamodb')\n        create_client().list_tables()\n    ") == {'dynamodb': set(['list_tables'])}

def test_propagates_return_types():
    if False:
        return 10
    assert aws_calls("        import boto3\n        def create_client1():\n            return create_client2()\n        def create_client2():\n            return create_client3()\n        def create_client3():\n            return boto3.client('dynamodb')\n        create_client1().list_tables()\n    ") == {'dynamodb': set(['list_tables'])}

def test_decorator_list_is_ignored():
    if False:
        return 10
    assert known_types_for_function("        import boto3\n        import decorators\n\n        @decorators.retry(10)\n        def foo():\n            d = boto3.client('dynamodb')\n            e = d.list_tables()\n        foo()\n    ", name='foo') == {'d': Boto3ClientType('dynamodb'), 'e': Boto3ClientMethodCallType('dynamodb', 'list_tables')}

def test_can_map_function_params():
    if False:
        while True:
            i = 10
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        def make_call(client):\n            a = 1\n            return client.list_tables()\n        make_call(d)\n    ") == {'dynamodb': set(['list_tables'])}

def test_can_understand_shadowed_vars_from_func_arg():
    if False:
        while True:
            i = 10
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        def make_call(d):\n            return d.list_tables()\n        make_call('foo')\n    ") == {}

def test_can_understand_shadowed_vars_from_local_scope():
    if False:
        return 10
    assert aws_calls("        import boto3, mock\n        d = boto3.client('dynamodb')\n        def make_call(e):\n            d = mock.Mock()\n            return d.list_tables()\n        make_call(d)\n    ") == {}

def test_can_map_function_with_multiple_args():
    if False:
        return 10
    assert aws_calls("        import boto3, mock\n        m = mock.Mock()\n        d = boto3.client('dynamodb')\n        def make_call(other, client):\n            a = 1\n            other.create_table()\n            return client.list_tables()\n        make_call(m, d)\n    ") == {'dynamodb': set(['list_tables'])}

def test_multiple_function_calls():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3, mock\n        m = mock.Mock()\n        d = boto3.client('dynamodb')\n        def make_call(other, client):\n            a = 1\n            other.create_table()\n            return other_call(a, 2, 3, client)\n        def other_call(a, b, c, client):\n            return client.list_tables()\n        make_call(m, d)\n    ") == {'dynamodb': set(['list_tables'])}

def test_can_lookup_var_names_to_functions():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3\n        service_name = 'dynamodb'\n        d = boto3.client(service_name)\n        d.list_tables()\n    ") == {'dynamodb': set(['list_tables'])}

def test_map_string_literals_across_scopes():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3\n        service_name = 'dynamodb'\n        def foo():\n            service_name = 's3'\n            d = boto3.client(service_name)\n            d.list_buckets()\n        d = boto3.client(service_name)\n        d.list_tables()\n        foo()\n    ") == {'s3': set(['list_buckets']), 'dynamodb': set(['list_tables'])}

def test_can_handle_lambda_keyword():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls('        def foo(a):\n            return sorted(bar.values(),\n                          key=lambda x: x.baz[a - 1],\n                          reverse=True)\n        bar = {}\n        foo(12)\n    ') == {}

def test_dict_comp_with_no_client_calls():
    if False:
        while True:
            i = 10
    assert aws_calls('        import boto3\n        foo = {i: i for i in range(10)}\n    ') == {}

def test_can_handle_gen_expr():
    if False:
        print('Hello World!')
    assert aws_calls("        import boto3\n        ('a' for y in [1,2,3])\n    ") == {}

def test_can_detect_calls_in_gen_expr():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3\n        service_name = 'dynamodb'\n        d = boto3.client('dynamodb')\n        (d.list_tables() for i in [1,2,3])\n    ") == {'dynamodb': set(['list_tables'])}

def test_can_handle_gen_from_call():
    if False:
        return 10
    assert aws_calls("        import boto3\n        service_name = 'dynamodb'\n        d = boto3.client('dynamodb')\n        (i for i in d.list_tables())\n    ") == {'dynamodb': set(['list_tables'])}

def test_can_detect_calls_in_multiple_gen_exprs():
    if False:
        i = 10
        return i + 15
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        (d for i in [1,2,3])\n        (d.list_tables() for j in [1,2,3])\n    ") == {'dynamodb': set(['list_tables'])}

def test_multiple_gen_exprs():
    if False:
        while True:
            i = 10
    assert aws_calls('        (i for i in [1,2,3])\n        (j for j in [1,2,3])\n    ') == {}

def test_can_handle_list_expr_with_api_calls():
    if False:
        return 10
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        [d.list_tables() for y in [1,2,3]]\n    ") == {'dynamodb': set(['list_tables'])}

def test_can_handle_multiple_listcomps():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        bar_key = 'bar'\n        baz_key = 'baz'\n        items = [{'foo': 'sun', 'bar': 'moon', 'baz': 'stars'}]\n        foos = [i['foo'] for i in items]\n        bars = [j[bar_key] for j in items]\n        bazs = [k[baz_key] for k in items]\n    ") == {}

def test_can_analyze_lambda_function():
    if False:
        i = 10
        return i + 15
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n        app = Chalice(app_name='james1')\n        ec2 = boto3.client('ec2')\n        @app.lambda_function(name='lambda1')\n        def index():\n            ec2.describe_instances()\n            return {}\n    ") == {'ec2': set(['describe_instances'])}

def test_can_analyze_schedule():
    if False:
        for i in range(10):
            print('nop')
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n        app = Chalice(app_name='james1')\n        s3cli = boto3.client('s3')\n        @app.schedule('rate(1 hour)')\n        def index():\n            s3cli.list_buckets()\n            return {}\n    ") == {'s3': set(['list_buckets'])}

def test_can_analyze_combination():
    if False:
        for i in range(10):
            print('nop')
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n        app = Chalice(app_name='james1')\n        s3 = boto3.client('s3')\n        ec = boto3.client('ec2')\n        @app.route('/')\n        def index():\n            ec2.describe_instances()\n            return {}\n        @app.schedule('rate(1 hour)')\n        def index_sc():\n            s3.list_buckets()\n            return {}\n\n        @app.lambda_function(name='lambda1')\n        def index_lm():\n            ec.describe_instances()\n            return {}\n\n        @random\n        def foo():\n            return {}\n\n    ") == {'s3': set(['list_buckets']), 'ec2': set(['describe_instances'])}

def test_can_handle_dict_comp():
    if False:
        i = 10
        return i + 15
    assert aws_calls("        import boto3\n        ddb = boto3.client('dynamodb')\n        tables = {t: t for t in ddb.list_tables()}\n    ") == {'dynamodb': set(['list_tables'])}

def test_can_handle_dict_comp_if():
    if False:
        return 10
    assert aws_calls("        import boto3\n        ddb = boto3.client('dynamodb')\n        tables = {t: t for t in [1] if ddb.list_tables()}\n    ") == {'dynamodb': set(['list_tables'])}

def test_can_handle_comp_ifs():
    if False:
        i = 10
        return i + 15
    assert aws_calls('        [(x,y) for x in [1,2,3,4] for y in [1,2,3,4] if x % 2 == 0]\n    ') == {}

def test_can_handle_dict_comp_ifs():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3\n        d = boto3.client('dynamodb')\n        {x: y for x in d.create_table()         for y in d.update_table()         if d.list_tables()}\n        {x: y for x in d.create_table()         for y in d.update_table()         if d.list_tables()}\n    ") == {'dynamodb': set(['list_tables', 'create_table', 'update_table'])}

@pytest.mark.skipif(sys.version[0] == '2', reason='Async await syntax is not in Python 2')
def test_can_handle_async_await():
    if False:
        for i in range(10):
            print('nop')
    assert aws_calls("        import boto3\n        import asyncio\n        async def test():\n            d = boto3.client('dynamodb')\n            d.list_tables()\n            await asyncio.sleep(1)\n        test()\n    ") == {'dynamodb': set(['list_tables'])}

def test_can_analyze_custom_auth():
    if False:
        for i in range(10):
            print('nop')
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        ec2 = boto3.client('ec2')\n        app = Chalice(app_name='custom-auth')\n\n        @app.authorizer()\n        def index(auth_request):\n            ec2.describe_instances()\n            return {}\n    ") == {'ec2': set(['describe_instances'])}

def test_can_analyze_s3_events():
    if False:
        while True:
            i = 10
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        s3 = boto3.client('s3')\n        app = Chalice(app_name='s3-event')\n\n        @app.on_s3_event(bucket='mybucket')\n        def index(event):\n            s3.list_buckets()\n            return {}\n    ") == {'s3': set(['list_buckets'])}

def test_can_analyze_sns_events():
    if False:
        return 10
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        s3 = boto3.client('s3')\n        app = Chalice(app_name='sns-event')\n\n        @app.on_sns_message(topic='mytopic')\n        def index(event):\n            s3.list_buckets()\n            return {}\n    ") == {'s3': set(['list_buckets'])}

def test_can_analyze_sqs_events():
    if False:
        i = 10
        return i + 15
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        s3 = boto3.client('s3')\n        app = Chalice(app_name='sqs-event')\n\n        @app.on_sqs_message(queue='myqueue')\n        def index(event):\n            s3.list_buckets()\n            return {}\n    ") == {'s3': set(['list_buckets'])}

def test_can_analyze_transfer_manager_methods():
    if False:
        while True:
            i = 10
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        s3 = boto3.client('s3')\n        app = Chalice(app_name='sqs-event')\n\n        @app.on_s3_event(bucket='mybucket')\n        def index(event):\n            s3.download_file(event.bucket, event.key, 'foo')\n            return {}\n    ") == {'s3': set(['download_file'])}

def test_can_handle_replacing_function_name():
    if False:
        return 10
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        app = Chalice(app_name='sqs-event')\n\n        def index():\n            pass\n\n        @app.on_sqs_message(queue='myqueue')\n        def index(event):\n            foo = boto3.client('s3').list_buckets()\n\n    ") == {'s3': set(['list_buckets'])}

def test_can_handle_multiple_shadowing():
    if False:
        for i in range(10):
            print('nop')
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        app = Chalice(app_name='sqs-event')\n\n        def index():\n            pass\n\n        @app.on_sqs_message(queue='myqueue')\n        def index(event):\n            foo = boto3.client('s3').list_buckets()\n\n        @app.on_s3_event(bucket='mybucket')\n        def index(event):\n            bar = boto3.client('s3').head_bucket(Bucket='foo')\n\n    ") == {'s3': set(['list_buckets', 'head_bucket'])}

def test_can_handle_forward_declaration():
    if False:
        i = 10
        return i + 15
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        app = Chalice(app_name='forward-declaration')\n\n        def get_regions():\n            return boto3.client('s3').list_buckets()\n\n        @app.route('/')\n        def index():\n            return get_regions()\n\n    ") == {'s3': set(['list_buckets'])}

def test_can_handle_post_declaration():
    if False:
        print('Hello World!')
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        app = Chalice(app_name='post-declaration')\n\n        @app.route('/')\n        def index():\n            return get_regions()\n\n        def get_regions():\n            return boto3.client('s3').list_buckets()\n\n    ") == {'s3': set(['list_buckets'])}

def test_can_handle_shadowed_declaration():
    if False:
        return 10
    assert chalice_aws_calls("        from chalice import Chalice\n        import boto3\n\n        app = Chalice(app_name='shadowed-declaration')\n\n        def get_regions():\n            return boto3.client('s3').list_buckets()\n\n        @app.route('/')\n        def index():\n            return get_regions()\n\n        def get_regions():\n            return boto3.client('s3').head_bucket(Bucket='foo')\n\n    ") == {'s3': set(['head_bucket'])}
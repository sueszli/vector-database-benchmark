from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pytest
from kombu import Exchange, Queue
from celery import uuid
from celery.app.amqp import Queues, utf8dict
from celery.utils.time import to_utc

class test_TaskConsumer:

    def test_accept_content(self, app):
        if False:
            while True:
                i = 10
        with app.pool.acquire(block=True) as con:
            app.conf.accept_content = ['application/json']
            assert app.amqp.TaskConsumer(con).accept == {'application/json'}
            assert app.amqp.TaskConsumer(con, accept=['json']).accept == {'application/json'}

class test_ProducerPool:

    def test_setup_nolimit(self, app):
        if False:
            return 10
        app.conf.broker_pool_limit = None
        try:
            delattr(app, '_pool')
        except AttributeError:
            pass
        app.amqp._producer_pool = None
        pool = app.amqp.producer_pool
        assert pool.limit == app.pool.limit
        assert not pool._resource.queue
        r1 = pool.acquire()
        r2 = pool.acquire()
        r1.release()
        r2.release()
        r1 = pool.acquire()
        r2 = pool.acquire()

    def test_setup(self, app):
        if False:
            while True:
                i = 10
        app.conf.broker_pool_limit = 2
        try:
            delattr(app, '_pool')
        except AttributeError:
            pass
        app.amqp._producer_pool = None
        pool = app.amqp.producer_pool
        assert pool.limit == app.pool.limit
        assert pool._resource.queue
        p1 = r1 = pool.acquire()
        p2 = r2 = pool.acquire()
        r1.release()
        r2.release()
        r1 = pool.acquire()
        r2 = pool.acquire()
        assert p2 is r1
        assert p1 is r2
        r1.release()
        r2.release()

class test_Queues:

    def test_queues_format(self):
        if False:
            return 10
        self.app.amqp.queues._consume_from = {}
        assert self.app.amqp.queues.format() == ''

    def test_with_defaults(self):
        if False:
            print('Hello World!')
        assert Queues(None) == {}

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        q = Queues()
        q.add('foo', exchange='ex', routing_key='rk')
        assert 'foo' in q
        assert isinstance(q['foo'], Queue)
        assert q['foo'].routing_key == 'rk'

    def test_setitem_adds_default_exchange(self):
        if False:
            return 10
        q = Queues(default_exchange=Exchange('bar'))
        assert q.default_exchange
        queue = Queue('foo', exchange=None)
        queue.exchange = None
        q['foo'] = queue
        assert q['foo'].exchange == q.default_exchange

    def test_select_add(self):
        if False:
            print('Hello World!')
        q = Queues()
        q.select(['foo', 'bar'])
        q.select_add('baz')
        assert sorted(q._consume_from.keys()) == ['bar', 'baz', 'foo']

    def test_deselect(self):
        if False:
            i = 10
            return i + 15
        q = Queues()
        q.select(['foo', 'bar'])
        q.deselect('bar')
        assert sorted(q._consume_from.keys()) == ['foo']

    def test_add_default_exchange(self):
        if False:
            return 10
        ex = Exchange('fff', 'fanout')
        q = Queues(default_exchange=ex)
        q.add(Queue('foo'))
        assert q['foo'].exchange.name == 'fff'

    def test_alias(self):
        if False:
            for i in range(10):
                print('nop')
        q = Queues()
        q.add(Queue('foo', alias='barfoo'))
        assert q['barfoo'] is q['foo']

    @pytest.mark.parametrize('queues_kwargs,qname,q,expected', [({'max_priority': 10}, 'foo', 'foo', {'x-max-priority': 10}), ({'max_priority': 10}, 'xyz', Queue('xyz', queue_arguments={'x-max-priority': 3}), {'x-max-priority': 3}), ({'max_priority': 10}, 'moo', Queue('moo', queue_arguments=None), {'x-max-priority': 10}), ({'max_priority': None}, 'foo2', 'foo2', None), ({'max_priority': None}, 'xyx3', Queue('xyx3', queue_arguments={'x-max-priority': 7}), {'x-max-priority': 7})])
    def test_with_max_priority(self, queues_kwargs, qname, q, expected):
        if False:
            return 10
        queues = Queues(**queues_kwargs)
        queues.add(q)
        assert queues[qname].queue_arguments == expected

class test_default_queues:

    @pytest.mark.parametrize('name,exchange,rkey', [('default', None, None), ('default', 'exchange', None), ('default', 'exchange', 'routing_key'), ('default', None, 'routing_key')])
    def test_setting_default_queue(self, name, exchange, rkey):
        if False:
            print('Hello World!')
        self.app.conf.task_queues = {}
        self.app.conf.task_default_exchange = exchange
        self.app.conf.task_default_routing_key = rkey
        self.app.conf.task_default_queue = name
        assert self.app.amqp.queues.default_exchange.name == exchange or name
        queues = dict(self.app.amqp.queues)
        assert len(queues) == 1
        queue = queues[name]
        assert queue.exchange.name == exchange or name
        assert queue.exchange.type == 'direct'
        assert queue.routing_key == rkey or name

class test_default_exchange:

    @pytest.mark.parametrize('name,exchange,rkey', [('default', 'foo', None), ('default', 'foo', 'routing_key')])
    def test_setting_default_exchange(self, name, exchange, rkey):
        if False:
            for i in range(10):
                print('nop')
        q = Queue(name, routing_key=rkey)
        self.app.conf.task_queues = {q}
        self.app.conf.task_default_exchange = exchange
        queues = dict(self.app.amqp.queues)
        queue = queues[name]
        assert queue.exchange.name == exchange

    @pytest.mark.parametrize('name,extype,rkey', [('default', 'direct', None), ('default', 'direct', 'routing_key'), ('default', 'topic', None), ('default', 'topic', 'routing_key')])
    def test_setting_default_exchange_type(self, name, extype, rkey):
        if False:
            return 10
        q = Queue(name, routing_key=rkey)
        self.app.conf.task_queues = {q}
        self.app.conf.task_default_exchange_type = extype
        queues = dict(self.app.amqp.queues)
        queue = queues[name]
        assert queue.exchange.type == extype

class test_AMQP_proto1:

    def test_kwargs_must_be_mapping(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError):
            self.app.amqp.as_task_v1(uuid(), 'foo', kwargs=[1, 2])

    def test_args_must_be_list(self):
        if False:
            print('Hello World!')
        with pytest.raises(TypeError):
            self.app.amqp.as_task_v1(uuid(), 'foo', args='abc')

    def test_countdown_negative(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            self.app.amqp.as_task_v1(uuid(), 'foo', countdown=-1232132323123)

    def test_as_task_message_without_utc(self):
        if False:
            print('Hello World!')
        self.app.amqp.utc = False
        self.app.amqp.as_task_v1(uuid(), 'foo', countdown=30, expires=40)

class test_AMQP_Base:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.simple_message = self.app.amqp.as_task_v2(uuid(), 'foo', create_sent_event=True)
        self.simple_message_no_sent_event = self.app.amqp.as_task_v2(uuid(), 'foo', create_sent_event=False)

class test_AMQP(test_AMQP_Base):

    def test_kwargs_must_be_mapping(self):
        if False:
            print('Hello World!')
        with pytest.raises(TypeError):
            self.app.amqp.as_task_v2(uuid(), 'foo', kwargs=[1, 2])

    def test_args_must_be_list(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError):
            self.app.amqp.as_task_v2(uuid(), 'foo', args='abc')

    def test_countdown_negative(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError):
            self.app.amqp.as_task_v2(uuid(), 'foo', countdown=-1232132323123)

    def test_Queues__with_max_priority(self):
        if False:
            print('Hello World!')
        x = self.app.amqp.Queues({}, max_priority=23)
        assert x.max_priority == 23

    def test_send_task_message__no_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.app.amqp.send_task_message(Mock(), 'foo', self.simple_message)

    def test_send_task_message__properties(self):
        if False:
            print('Hello World!')
        prod = Mock(name='producer')
        self.app.amqp.send_task_message(prod, 'foo', self.simple_message_no_sent_event, foo=1, retry=False)
        assert prod.publish.call_args[1]['foo'] == 1

    def test_send_task_message__headers(self):
        if False:
            while True:
                i = 10
        prod = Mock(name='producer')
        self.app.amqp.send_task_message(prod, 'foo', self.simple_message_no_sent_event, headers={'x1x': 'y2x'}, retry=False)
        assert prod.publish.call_args[1]['headers']['x1x'] == 'y2x'

    def test_send_task_message__queue_string(self):
        if False:
            while True:
                i = 10
        prod = Mock(name='producer')
        self.app.amqp.send_task_message(prod, 'foo', self.simple_message_no_sent_event, queue='foo', retry=False)
        kwargs = prod.publish.call_args[1]
        assert kwargs['routing_key'] == 'foo'
        assert kwargs['exchange'] == ''

    def test_send_task_message__broadcast_without_exchange(self):
        if False:
            i = 10
            return i + 15
        from kombu.common import Broadcast
        evd = Mock(name='evd')
        self.app.amqp.send_task_message(Mock(), 'foo', self.simple_message, retry=False, routing_key='xyz', queue=Broadcast('abc'), event_dispatcher=evd)
        evd.publish.assert_called()
        event = evd.publish.call_args[0][1]
        assert event['routing_key'] == 'xyz'
        assert event['exchange'] == 'abc'

    def test_send_event_exchange_direct_with_exchange(self):
        if False:
            return 10
        prod = Mock(name='prod')
        self.app.amqp.send_task_message(prod, 'foo', self.simple_message_no_sent_event, queue='bar', retry=False, exchange_type='direct', exchange='xyz')
        prod.publish.assert_called()
        pub = prod.publish.call_args[1]
        assert pub['routing_key'] == 'bar'
        assert pub['exchange'] == ''

    def test_send_event_exchange_direct_with_routing_key(self):
        if False:
            print('Hello World!')
        prod = Mock(name='prod')
        self.app.amqp.send_task_message(prod, 'foo', self.simple_message_no_sent_event, queue='bar', retry=False, exchange_type='direct', routing_key='xyb')
        prod.publish.assert_called()
        pub = prod.publish.call_args[1]
        assert pub['routing_key'] == 'bar'
        assert pub['exchange'] == ''

    def test_send_event_exchange_string(self):
        if False:
            print('Hello World!')
        evd = Mock(name='evd')
        self.app.amqp.send_task_message(Mock(), 'foo', self.simple_message, retry=False, exchange='xyz', routing_key='xyb', event_dispatcher=evd)
        evd.publish.assert_called()
        event = evd.publish.call_args[0][1]
        assert event['routing_key'] == 'xyb'
        assert event['exchange'] == 'xyz'

    def test_send_task_message__with_delivery_mode(self):
        if False:
            while True:
                i = 10
        prod = Mock(name='producer')
        self.app.amqp.send_task_message(prod, 'foo', self.simple_message_no_sent_event, delivery_mode=33, retry=False)
        assert prod.publish.call_args[1]['delivery_mode'] == 33

    def test_send_task_message__with_receivers(self):
        if False:
            return 10
        mocked_receiver = ((Mock(), Mock()), Mock())
        with patch('celery.signals.task_sent.receivers', [mocked_receiver]):
            self.app.amqp.send_task_message(Mock(), 'foo', self.simple_message)

    def test_routes(self):
        if False:
            while True:
                i = 10
        r1 = self.app.amqp.routes
        r2 = self.app.amqp.routes
        assert r1 is r2

    def update_conf_runtime_for_tasks_queues(self):
        if False:
            for i in range(10):
                print('nop')
        self.app.conf.update(task_routes={'task.create_pr': 'queue.qwerty'})
        self.app.send_task('task.create_pr')
        router_was = self.app.amqp.router
        self.app.conf.update(task_routes={'task.create_pr': 'queue.asdfgh'})
        self.app.send_task('task.create_pr')
        router = self.app.amqp.router
        assert router != router_was

class test_as_task_v2(test_AMQP_Base):

    def test_raises_if_args_is_not_tuple(self):
        if False:
            print('Hello World!')
        with pytest.raises(TypeError):
            self.app.amqp.as_task_v2(uuid(), 'foo', args='123')

    def test_raises_if_kwargs_is_not_mapping(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError):
            self.app.amqp.as_task_v2(uuid(), 'foo', kwargs=(1, 2, 3))

    def test_countdown_to_eta(self):
        if False:
            i = 10
            return i + 15
        now = to_utc(datetime.utcnow()).astimezone(self.app.timezone)
        m = self.app.amqp.as_task_v2(uuid(), 'foo', countdown=10, now=now)
        assert m.headers['eta'] == (now + timedelta(seconds=10)).isoformat()

    def test_expires_to_datetime(self):
        if False:
            while True:
                i = 10
        now = to_utc(datetime.utcnow()).astimezone(self.app.timezone)
        m = self.app.amqp.as_task_v2(uuid(), 'foo', expires=30, now=now)
        assert m.headers['expires'] == (now + timedelta(seconds=30)).isoformat()

    def test_eta_to_datetime(self):
        if False:
            return 10
        eta = datetime.utcnow()
        m = self.app.amqp.as_task_v2(uuid(), 'foo', eta=eta)
        assert m.headers['eta'] == eta.isoformat()

    def test_compression(self):
        if False:
            while True:
                i = 10
        self.app.conf.task_compression = 'gzip'
        prod = Mock(name='producer')
        self.app.amqp.send_task_message(prod, 'foo', self.simple_message_no_sent_event, compression=None)
        assert prod.publish.call_args[1]['compression'] == 'gzip'

    def test_compression_override(self):
        if False:
            while True:
                i = 10
        self.app.conf.task_compression = 'gzip'
        prod = Mock(name='producer')
        self.app.amqp.send_task_message(prod, 'foo', self.simple_message_no_sent_event, compression='bz2')
        assert prod.publish.call_args[1]['compression'] == 'bz2'

    def test_callbacks_errbacks_chord(self):
        if False:
            print('Hello World!')

        @self.app.task
        def t(i):
            if False:
                print('Hello World!')
            pass
        m = self.app.amqp.as_task_v2(uuid(), 'foo', callbacks=[t.s(1), t.s(2)], errbacks=[t.s(3), t.s(4)], chord=t.s(5))
        (_, _, embed) = m.body
        assert embed['callbacks'] == [utf8dict(t.s(1)), utf8dict(t.s(2))]
        assert embed['errbacks'] == [utf8dict(t.s(3)), utf8dict(t.s(4))]
        assert embed['chord'] == utf8dict(t.s(5))
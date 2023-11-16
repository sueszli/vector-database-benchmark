"""

Tags: Load test, stress test.

A utility script that injects trigger instances into st2 system.

This tool is designed with stress/load testing in mind. Trigger
instances need appropriate rules to be setup so there is some
meaningful work.

"""
from __future__ import absolute_import
import os
import random
import eventlet
from oslo_config import cfg
import yaml
from st2common import config
from st2common.util.monkey_patch import monkey_patch
from st2common.util import date as date_utils
from st2common.transport.reactor import TriggerDispatcher

def do_register_cli_opts(opts, ignore_errors=False):
    if False:
        for i in range(10):
            print('nop')
    for opt in opts:
        try:
            cfg.CONF.register_cli_opt(opt)
        except:
            if not ignore_errors:
                raise

def _inject_instances(trigger, rate_per_trigger, duration, payload=None, max_throughput=False):
    if False:
        i = 10
        return i + 15
    payload = payload or {}
    start = date_utils.get_datetime_utc_now()
    elapsed = 0.0
    count = 0
    dispatcher = TriggerDispatcher()
    while elapsed < duration:
        dispatcher.dispatch(trigger, payload)
        if rate_per_trigger:
            delta = random.expovariate(rate_per_trigger)
            eventlet.sleep(delta * 0.56)
        elapsed = (date_utils.get_datetime_utc_now() - start).seconds
        count += 1
    actual_rate = int(count / elapsed)
    print('%s: Emitted %d triggers in %d seconds (actual rate=%s triggers / second)' % (trigger, count, elapsed, actual_rate))
    if rate_per_trigger and actual_rate < rate_per_trigger * 0.9:
        print('')
        print('Warning, requested rate was %s triggers / second, but only achieved %s triggers / second' % (rate_per_trigger, actual_rate))
        print('Too increase the throuput you will likely need to run multiple instances of this script in parallel.')

def main():
    if False:
        return 10
    monkey_patch()
    cli_opts = [cfg.IntOpt('rate', default=100, help='Rate of trigger injection measured in instances in per sec.' + ' Assumes a default exponential distribution in time so arrival is poisson.'), cfg.ListOpt('triggers', required=False, help='List of triggers for which instances should be fired.' + ' Uniform distribution will be followed if there is more than one' + 'trigger.'), cfg.StrOpt('schema_file', default=None, help='Path to schema file defining trigger and payload.'), cfg.IntOpt('duration', default=60, help='Duration of stress test in seconds.'), cfg.BoolOpt('max-throughput', default=False, help='If True, "rate" argument will be ignored and this script will try to saturize the CPU and achieve max utilization.')]
    do_register_cli_opts(cli_opts)
    config.parse_args()
    triggers = cfg.CONF.triggers
    trigger_payload_schema = {}
    if not triggers:
        if cfg.CONF.schema_file is None or cfg.CONF.schema_file == '' or (not os.path.exists(cfg.CONF.schema_file)):
            print('Either "triggers" need to be provided or a schema file containing' + ' triggers should be provided.')
            return
        with open(cfg.CONF.schema_file) as fd:
            trigger_payload_schema = yaml.safe_load(fd)
            triggers = list(trigger_payload_schema.keys())
            print('Triggers=%s' % triggers)
    rate = cfg.CONF.rate
    rate_per_trigger = int(rate / len(triggers))
    duration = cfg.CONF.duration
    max_throughput = cfg.CONF.max_throughput
    if max_throughput:
        rate = 0
        rate_per_trigger = 0
    dispatcher_pool = eventlet.GreenPool(len(triggers))
    for trigger in triggers:
        payload = trigger_payload_schema.get(trigger, {})
        dispatcher_pool.spawn(_inject_instances, trigger, rate_per_trigger, duration, payload=payload, max_throughput=max_throughput)
        eventlet.sleep(random.uniform(0, 1))
    dispatcher_pool.waitall()
if __name__ == '__main__':
    main()
from copy import copy
from logbook import Logger, Processor
from zipline.finance.order import ORDER_STATUS
from zipline.protocol import BarData
from zipline.utils.api_support import ZiplineAPI
from zipline.utils.compat import ExitStack
from six import viewkeys
from zipline.gens.sim_engine import BAR, SESSION_START, SESSION_END, MINUTE_END, BEFORE_TRADING_START_BAR
log = Logger('Trade Simulation')

class AlgorithmSimulator(object):
    EMISSION_TO_PERF_KEY_MAP = {'minute': 'minute_perf', 'daily': 'daily_perf'}

    def __init__(self, algo, sim_params, data_portal, clock, benchmark_source, restrictions, universe_func):
        if False:
            while True:
                i = 10
        self.sim_params = sim_params
        self.data_portal = data_portal
        self.restrictions = restrictions
        self.algo = algo
        self.current_data = self._create_bar_data(universe_func)
        self.simulation_dt = None
        self.clock = clock
        self.benchmark_source = benchmark_source

        def inject_algo_dt(record):
            if False:
                return 10
            if 'algo_dt' not in record.extra:
                record.extra['algo_dt'] = self.simulation_dt
        self.processor = Processor(inject_algo_dt)

    def get_simulation_dt(self):
        if False:
            while True:
                i = 10
        return self.simulation_dt

    def _create_bar_data(self, universe_func):
        if False:
            print('Hello World!')
        return BarData(data_portal=self.data_portal, simulation_dt_func=self.get_simulation_dt, data_frequency=self.sim_params.data_frequency, trading_calendar=self.algo.trading_calendar, restrictions=self.restrictions, universe_func=universe_func)

    def transform(self):
        if False:
            while True:
                i = 10
        '\n        Main generator work loop.\n        '
        algo = self.algo
        metrics_tracker = algo.metrics_tracker
        emission_rate = metrics_tracker.emission_rate

        def every_bar(dt_to_use, current_data=self.current_data, handle_data=algo.event_manager.handle_data):
            if False:
                for i in range(10):
                    print('nop')
            for capital_change in calculate_minute_capital_changes(dt_to_use):
                yield capital_change
            self.simulation_dt = dt_to_use
            algo.on_dt_changed(dt_to_use)
            blotter = algo.blotter
            (new_transactions, new_commissions, closed_orders) = blotter.get_transactions(current_data)
            blotter.prune_orders(closed_orders)
            for transaction in new_transactions:
                metrics_tracker.process_transaction(transaction)
                order = blotter.orders[transaction.order_id]
                metrics_tracker.process_order(order)
            for commission in new_commissions:
                metrics_tracker.process_commission(commission)
            handle_data(algo, current_data, dt_to_use)
            new_orders = blotter.new_orders
            blotter.new_orders = []
            for new_order in new_orders:
                metrics_tracker.process_order(new_order)

        def once_a_day(midnight_dt, current_data=self.current_data, data_portal=self.data_portal):
            if False:
                print('Hello World!')
            for capital_change in algo.calculate_capital_changes(midnight_dt, emission_rate=emission_rate, is_interday=True):
                yield capital_change
            self.simulation_dt = midnight_dt
            algo.on_dt_changed(midnight_dt)
            metrics_tracker.handle_market_open(midnight_dt, algo.data_portal)
            assets_we_care_about = viewkeys(metrics_tracker.positions) | viewkeys(algo.blotter.open_orders)
            if assets_we_care_about:
                splits = data_portal.get_splits(assets_we_care_about, midnight_dt)
                if splits:
                    algo.blotter.process_splits(splits)
                    metrics_tracker.handle_splits(splits)

        def on_exit():
            if False:
                i = 10
                return i + 15
            self.algo = None
            self.benchmark_source = self.current_data = self.data_portal = None
        with ExitStack() as stack:
            stack.callback(on_exit)
            stack.enter_context(self.processor)
            stack.enter_context(ZiplineAPI(self.algo))
            if algo.data_frequency == 'minute':

                def execute_order_cancellation_policy():
                    if False:
                        return 10
                    algo.blotter.execute_cancel_policy(SESSION_END)

                def calculate_minute_capital_changes(dt):
                    if False:
                        while True:
                            i = 10
                    return algo.calculate_capital_changes(dt, emission_rate=emission_rate, is_interday=False)
            else:

                def execute_order_cancellation_policy():
                    if False:
                        while True:
                            i = 10
                    pass

                def calculate_minute_capital_changes(dt):
                    if False:
                        print('Hello World!')
                    return []
            for (dt, action) in self.clock:
                if action == BAR:
                    for capital_change_packet in every_bar(dt):
                        yield capital_change_packet
                elif action == SESSION_START:
                    for capital_change_packet in once_a_day(dt):
                        yield capital_change_packet
                elif action == SESSION_END:
                    positions = metrics_tracker.positions
                    position_assets = algo.asset_finder.retrieve_all(positions)
                    self._cleanup_expired_assets(dt, position_assets)
                    execute_order_cancellation_policy()
                    algo.validate_account_controls()
                    yield self._get_daily_message(dt, algo, metrics_tracker)
                elif action == BEFORE_TRADING_START_BAR:
                    self.simulation_dt = dt
                    algo.on_dt_changed(dt)
                    algo.before_trading_start(self.current_data)
                elif action == MINUTE_END:
                    minute_msg = self._get_minute_message(dt, algo, metrics_tracker)
                    yield minute_msg
            risk_message = metrics_tracker.handle_simulation_end(self.data_portal)
            yield risk_message

    def _cleanup_expired_assets(self, dt, position_assets):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear out any assets that have expired before starting a new sim day.\n\n        Performs two functions:\n\n        1. Finds all assets for which we have open orders and clears any\n           orders whose assets are on or after their auto_close_date.\n\n        2. Finds all assets for which we have positions and generates\n           close_position events for any assets that have reached their\n           auto_close_date.\n        '
        algo = self.algo

        def past_auto_close_date(asset):
            if False:
                i = 10
                return i + 15
            acd = asset.auto_close_date
            return acd is not None and acd <= dt
        assets_to_clear = [asset for asset in position_assets if past_auto_close_date(asset)]
        metrics_tracker = algo.metrics_tracker
        data_portal = self.data_portal
        for asset in assets_to_clear:
            metrics_tracker.process_close_position(asset, dt, data_portal)
        blotter = algo.blotter
        assets_to_cancel = [asset for asset in blotter.open_orders if past_auto_close_date(asset)]
        for asset in assets_to_cancel:
            blotter.cancel_all_orders_for_asset(asset)
        for order in copy(blotter.new_orders):
            if order.status == ORDER_STATUS.CANCELLED:
                metrics_tracker.process_order(order)
                blotter.new_orders.remove(order)

    def _get_daily_message(self, dt, algo, metrics_tracker):
        if False:
            print('Hello World!')
        '\n        Get a perf message for the given datetime.\n        '
        perf_message = metrics_tracker.handle_market_close(dt, self.data_portal)
        perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
        return perf_message

    def _get_minute_message(self, dt, algo, metrics_tracker):
        if False:
            print('Hello World!')
        '\n        Get a perf message for the given datetime.\n        '
        rvars = algo.recorded_vars
        minute_message = metrics_tracker.handle_minute_close(dt, self.data_portal)
        minute_message['minute_perf']['recorded_vars'] = rvars
        return minute_message
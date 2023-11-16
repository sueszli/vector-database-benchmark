from bs4 import BeautifulSoup

class DashPageMixin:

    def _get_dash_dom_by_attribute(self, attr):
        if False:
            i = 10
            return i + 15
        return BeautifulSoup(self.find_element(self.dash_entry_locator).get_attribute(attr), 'lxml')

    @property
    def devtools_error_count_locator(self):
        if False:
            print('Hello World!')
        return '.test-devtools-error-count'

    @property
    def dash_entry_locator(self):
        if False:
            i = 10
            return i + 15
        return '#react-entry-point'

    @property
    def dash_outerhtml_dom(self):
        if False:
            i = 10
            return i + 15
        return self._get_dash_dom_by_attribute('outerHTML')

    @property
    def dash_innerhtml_dom(self):
        if False:
            while True:
                i = 10
        return self._get_dash_dom_by_attribute('innerHTML')

    @property
    def redux_state_paths(self):
        if False:
            return 10
        return self.driver.execute_script('\n            var p = window.store.getState().paths;\n            return {strs: p.strs, objs: p.objs}\n            ')

    @property
    def redux_state_rqs(self):
        if False:
            i = 10
            return i + 15
        return self.driver.execute_script("\n\n            // Check for legacy `pendingCallbacks` store prop (compatibility for Dash matrix testing)\n            var pendingCallbacks = window.store.getState().pendingCallbacks;\n            if (pendingCallbacks) {\n                return pendingCallbacks.map(function(cb) {\n                    var out = {};\n                    for (var key in cb) {\n                        if (typeof cb[key] !== 'function') { out[key] = cb[key]; }\n                    }\n                    return out;\n                });\n            }\n\n            // Otherwise, use the new `callbacks` store prop\n            var callbacksState =  Object.assign({}, window.store.getState().callbacks);\n            delete callbacksState.stored;\n            delete callbacksState.completed;\n\n            return Array.prototype.concat.apply([], Object.values(callbacksState));\n            ")

    @property
    def redux_state_is_loading(self):
        if False:
            while True:
                i = 10
        return self.driver.execute_script('\n            return window.store.getState().isLoading;\n            ')

    @property
    def window_store(self):
        if False:
            return 10
        return self.driver.execute_script('return window.store')

    def _wait_for_callbacks(self):
        if False:
            for i in range(10):
                print('nop')
        return not self.window_store or self.redux_state_rqs == []

    def get_local_storage(self, store_id='local'):
        if False:
            for i in range(10):
                print('nop')
        return self.driver.execute_script(f"return JSON.parse(window.localStorage.getItem('{store_id}'));")

    def get_session_storage(self, session_id='session'):
        if False:
            while True:
                i = 10
        return self.driver.execute_script(f"return JSON.parse(window.sessionStorage.getItem('{session_id}'));")

    def clear_local_storage(self):
        if False:
            i = 10
            return i + 15
        self.driver.execute_script('window.localStorage.clear()')

    def clear_session_storage(self):
        if False:
            print('Hello World!')
        self.driver.execute_script('window.sessionStorage.clear()')

    def clear_storage(self):
        if False:
            while True:
                i = 10
        self.clear_local_storage()
        self.clear_session_storage()
from locust import HttpUser, task, events

@events.init_command_line_parser.add_listener
def _(parser):
    if False:
        while True:
            i = 10
    parser.add_argument('--my-argument', type=str, env_var='LOCUST_MY_ARGUMENT', default='', help="It's working")
    parser.add_argument('--env', choices=['dev', 'staging', 'prod'], default='dev', help='Environment')
    parser.add_argument('--my-ui-invisible-argument', include_in_web_ui=False, default='I am invisible')
    parser.add_argument('--my-ui-password-argument', is_secret=True, default='I am a secret')

@events.test_start.add_listener
def _(environment, **kw):
    if False:
        for i in range(10):
            print('nop')
    print(f'Custom argument supplied: {environment.parsed_options.my_argument}')

class WebsiteUser(HttpUser):

    @task
    def my_task(self):
        if False:
            for i in range(10):
                print('nop')
        print(f'my_argument={self.environment.parsed_options.my_argument}')
        print(f'my_ui_invisible_argument={self.environment.parsed_options.my_ui_invisible_argument}')
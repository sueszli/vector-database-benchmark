"""Selenium Plugin for SeleniumBase tests that run with pynose / nosetests"""
import sys
from nose.plugins import Plugin
from seleniumbase import config as sb_config
from seleniumbase.config import settings
from seleniumbase.core import proxy_helper
from seleniumbase.fixtures import constants
from seleniumbase.fixtures import shared_utils

class SeleniumBrowser(Plugin):
    """This plugin adds the following command-line options to pynose:
    --browser=BROWSER  (The web browser to use. Default: "chrome".)
    --chrome  (Shortcut for "--browser=chrome". Default.)
    --edge  (Shortcut for "--browser=edge".)
    --firefox  (Shortcut for "--browser=firefox".)
    --safari  (Shortcut for "--browser=safari".)
    --user-data-dir=DIR  (Set the Chrome user data directory to use.)
    --protocol=PROTOCOL  (The Selenium Grid protocol: http|https.)
    --server=SERVER  (The Selenium Grid server/IP used for tests.)
    --port=PORT  (The Selenium Grid port used by the test server.)
    --cap-file=FILE  (The web browser's desired capabilities to use.)
    --cap-string=STRING  (The web browser's desired capabilities to use.)
    --proxy=SERVER:PORT  (Connect to a proxy server:port as tests are running)
    --proxy=USERNAME:PASSWORD@SERVER:PORT  (Use an authenticated proxy server)
    --proxy-bypass-list=STRING (";"-separated hosts to bypass, Eg "*.foo.com")
    --proxy-pac-url=URL  (Connect to a proxy server using a PAC_URL.pac file.)
    --proxy-pac-url=USERNAME:PASSWORD@URL  (Authenticated proxy with PAC URL.)
    --proxy-driver  (If a driver download is needed, will use: --proxy=PROXY.)
    --multi-proxy  (Allow multiple authenticated proxies when multi-threaded.)
    --agent=STRING  (Modify the web browser's User-Agent string.)
    --mobile  (Use the mobile device emulator while running tests.)
    --metrics=STRING  (Set mobile metrics: "CSSWidth,CSSHeight,PixelRatio".)
    --chromium-arg="ARG=N,ARG2" (Set Chromium args, ","-separated, no spaces.)
    --firefox-arg="ARG=N,ARG2" (Set Firefox args, comma-separated, no spaces.)
    --firefox-pref=SET  (Set a Firefox preference:value set, comma-separated.)
    --extension-zip=ZIP  (Load a Chrome Extension .zip|.crx, comma-separated.)
    --extension-dir=DIR  (Load a Chrome Extension directory, comma-separated.)
    --binary-location=PATH  (Set path of the Chromium browser binary to use.)
    --driver-version=VER  (Set the chromedriver or uc_driver version to use.)
    --sjw  (Skip JS Waits for readyState to be "complete" or Angular to load.)
    --pls=PLS  (Set pageLoadStrategy on Chrome: "normal", "eager", or "none".)
    --headless  (Run tests in headless mode. The default arg on Linux OS.)
    --headless2  (Use the new headless mode, which supports extensions.)
    --headed  (Run tests in headed/GUI mode on Linux OS, where not default.)
    --xvfb  (Run tests using the Xvfb virtual display server on Linux OS.)
    --locale=LOCALE_CODE  (Set the Language Locale Code for the web browser.)
    --interval=SECONDS  (The autoplay interval for presentations & tour steps)
    --start-page=URL  (The starting URL for the web browser when tests begin.)
    --time-limit=SECONDS  (Safely fail any test that exceeds the time limit.)
    --slow  (Slow down the automation. Faster than using Demo Mode.)
    --demo  (Slow down and visually see test actions as they occur.)
    --demo-sleep=SECONDS  (Set the wait time after Slow & Demo Mode actions.)
    --highlights=NUM  (Number of highlight animations for Demo Mode actions.)
    --message-duration=SECONDS  (The time length for Messenger alerts.)
    --check-js  (Check for JavaScript errors after page loads.)
    --ad-block  (Block some types of display ads from loading.)
    --host-resolver-rules=RULES  (Set host-resolver-rules, comma-separated.)
    --block-images  (Block images from loading during tests.)
    --do-not-track  (Indicate to websites that you don't want to be tracked.)
    --verify-delay=SECONDS  (The delay before MasterQA verification checks.)
    --recorder  (Enables the Recorder for turning browser actions into code.)
    --rec-behave  (Same as Recorder Mode, but also generates behave-gherkin.)
    --rec-sleep  (If the Recorder is enabled, also records self.sleep calls.)
    --rec-print  (If the Recorder is enabled, prints output after tests end.)
    --disable-js  (Disable JavaScript on websites. Pages might break!)
    --disable-csp  (Disable the Content Security Policy of websites.)
    --disable-ws  (Disable Web Security on Chromium-based browsers.)
    --enable-ws  (Enable Web Security on Chromium-based browsers.)
    --enable-sync  (Enable "Chrome Sync" on websites.)
    --uc | --undetected  (Use undetected-chromedriver to evade bot-detection.)
    --uc-cdp-events  (Capture CDP events when running in "--undetected" mode.)
    --log-cdp  ("goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"})
    --remote-debug  (Sync to Chrome Remote Debugger chrome://inspect/#devices)
    --enable-3d-apis  (Enables WebGL and 3D APIs.)
    --swiftshader  (Chrome "--use-gl=angle" / "--use-angle=swiftshader-webgl")
    --incognito  (Enable Chrome's Incognito mode.)
    --guest  (Enable Chrome's Guest mode.)
    --dark  (Enable Chrome's Dark mode.)
    --devtools  (Open Chrome's DevTools when the browser opens.)
    --disable-beforeunload  (Disable the "beforeunload" event on Chrome.)
    --window-size=WIDTH,HEIGHT  (Set the browser's starting window size.)
    --maximize  (Start tests with the browser window maximized.)
    --screenshot  (Save a screenshot at the end of each test.)
    --visual-baseline  (Set the visual baseline for Visual/Layout tests.)
    --wire  (Use selenium-wire's webdriver for replacing selenium webdriver.)
    --external-pdf (Set Chromium "plugins.always_open_pdf_externally": True.)
    --timeout-multiplier=MULTIPLIER  (Multiplies the default timeout values.)
    """
    name = 'selenium'

    def options(self, parser, env):
        if False:
            for i in range(10):
                print('nop')
        super().options(parser, env=env)
        parser.addoption = parser.add_option
        parser.addoption('--browser', action='store', dest='browser', choices=constants.ValidBrowsers.valid_browsers, default=constants.Browser.GOOGLE_CHROME, help='Specifies the web browser to use. Default: Chrome.\n                    Examples: (--browser=edge OR --browser=firefox)')
        parser.addoption('--chrome', action='store_true', dest='use_chrome', default=False, help='Shortcut for --browser=chrome (Default)')
        parser.addoption('--edge', action='store_true', dest='use_edge', default=False, help='Shortcut for --browser=edge')
        parser.addoption('--firefox', action='store_true', dest='use_firefox', default=False, help='Shortcut for --browser=firefox')
        parser.addoption('--ie', action='store_true', dest='use_ie', default=False, help='Shortcut for --browser=ie')
        parser.addoption('--safari', action='store_true', dest='use_safari', default=False, help='Shortcut for --browser=safari')
        parser.addoption('--cap_file', '--cap-file', action='store', dest='cap_file', default=None, help='The file that stores browser desired capabilities\n                    for BrowserStack, Sauce Labs, or other grids.')
        parser.addoption('--cap_string', '--cap-string', dest='cap_string', default=None, help='The string that stores browser desired capabilities\n                    for BrowserStack, Sauce Labs, or other grids.\n                    Enclose cap-string in single quotes.\n                    Enclose parameter keys in double quotes.\n                    Example: --cap-string=\'{"name":"test1","v":"42"}\'')
        parser.addoption('--user_data_dir', '--user-data-dir', action='store', dest='user_data_dir', default=None, help="The Chrome User Data Directory to use. (Chrome Profile)\n                    If the directory doesn't exist, it'll be created.")
        parser.addoption('--sjw', '--skip_js_waits', '--skip-js-waits', action='store_true', dest='skip_js_waits', default=False, help='Skip all calls to wait_for_ready_state_complete()\n                    and wait_for_angularjs(), which are part of many\n                    SeleniumBase methods for improving reliability.')
        parser.addoption('--protocol', action='store', dest='protocol', choices=(constants.Protocol.HTTP, constants.Protocol.HTTPS), default=constants.Protocol.HTTP, help='Designates the Selenium Grid protocol to use.\n                    Default: http.')
        parser.addoption('--server', action='store', dest='servername', default='localhost', help='Designates the Selenium Grid server to use.\n                    Use "127.0.0.1" to connect to a localhost Grid.\n                    If unset or set to "localhost", Grid isn\'t used.\n                    Default: "localhost".')
        parser.addoption('--port', action='store', dest='port', default='4444', help='Designates the Selenium Grid port to use.\n                    Default: 4444. (If 443, protocol becomes "https")')
        parser.addoption('--proxy', '--proxy-server', '--proxy-string', action='store', dest='proxy_string', default=None, help='Designates the proxy server:port to use.\n                    Format: servername:port.  OR\n                            username:password@servername:port  OR\n                            A dict key from proxy_list.PROXY_LIST\n                    Default: None.')
        parser.addoption('--proxy-bypass-list', '--proxy_bypass_list', action='store', dest='proxy_bypass_list', default=None, help='Designates the hosts, domains, and/or IP addresses\n                    to bypass when using a proxy server with "--proxy".\n                    Format: A ";"-separated string.\n                    Example usage:\n                        pytest\n                            --proxy="username:password@servername:port"\n                            --proxy-bypass-list="*.foo.com;github.com"\n                        pytest\n                            --proxy="servername:port"\n                            --proxy-bypass-list="127.0.0.1:8080"\n                    Default: None.')
        parser.addoption('--proxy-pac-url', '--pac-url', action='store', dest='proxy_pac_url', default=None, help='Designates the proxy PAC URL to use.\n                    Format: A URL string  OR\n                            A username:password@URL string\n                    Default: None.')
        parser.addoption('--proxy-driver', '--proxy_driver', action='store_true', dest='proxy_driver', default=False, help='If a driver download is needed for tests,\n                    uses proxy settings set via --proxy=PROXY.')
        parser.addoption('--multi-proxy', '--multi_proxy', action='store_true', dest='multi_proxy', default=False, help='If you need to run multi-threaded tests with\n                    multiple proxies that require authentication,\n                    set this to allow multiple configurations.')
        parser.addoption('--agent', '--user-agent', '--user_agent', action='store', dest='user_agent', default=None, help='Designates the User-Agent for the browser to use.\n                    Format: A string.\n                    Default: None.')
        parser.addoption('--mobile', '--mobile-emulator', '--mobile_emulator', action='store_true', dest='mobile_emulator', default=False, help='If this option is enabled, the mobile emulator\n                    will be used while running tests.')
        parser.addoption('--metrics', '--device-metrics', '--device_metrics', action='store', dest='device_metrics', default=None, help='Designates the three device metrics of the mobile\n                    emulator: CSS Width, CSS Height, and Pixel-Ratio.\n                    Format: A comma-separated string with the 3 values.\n                    Examples: "375,734,5" or "411,731,3" or "390,715,3"\n                    Default: None. (Will use default values if None)')
        parser.addoption('--chromium_arg', '--chromium-arg', action='store', dest='chromium_arg', default=None, help='Add a Chromium argument for Chrome/Edge browsers.\n                    Format: A comma-separated list of Chromium args.\n                    If an arg doesn\'t start with "--", that will be\n                    added to the beginning of the arg automatically.\n                    Default: None.')
        parser.addoption('--firefox_arg', '--firefox-arg', action='store', dest='firefox_arg', default=None, help='Add a Firefox argument for Firefox browser runs.\n                    Format: A comma-separated list of Firefox args.\n                    If an arg doesn\'t start with "--", that will be\n                    added to the beginning of the arg automatically.\n                    Default: None.')
        parser.addoption('--firefox_pref', '--firefox-pref', action='store', dest='firefox_pref', default=None, help='Set a Firefox preference:value combination.\n                    Format: A comma-separated list of pref:value items.\n                    Example usage:\n                        --firefox-pref="browser.formfill.enable:True"\n                        --firefox-pref="pdfjs.disabled:False"\n                        --firefox-pref="abc.def.xyz:42,hello.world:text"\n                    Boolean and integer values to the right of the ":"\n                    will be automatically converted into proper format.\n                    If there\'s no ":" in the string, then True is used.\n                    Default: None.')
        parser.addoption('--extension_zip', '--extension-zip', '--crx', action='store', dest='extension_zip', default=None, help='Designates the Chrome Extension ZIP file to load.\n                    Format: A comma-separated list of .zip or .crx files\n                    containing the Chrome extensions to load.\n                    Default: None.')
        parser.addoption('--extension_dir', '--extension-dir', action='store', dest='extension_dir', default=None, help='Designates the Chrome Extension folder to load.\n                    Format: A directory containing the Chrome extension.\n                    (Can also be a comma-separated list of directories.)\n                    Default: None.')
        parser.addoption('--binary_location', '--binary-location', action='store', dest='binary_location', default=None, help='Sets the path of the Chromium browser binary to use.\n                    Uses the default location if not os.path.exists(PATH)')
        parser.addoption('--driver_version', '--driver-version', action='store', dest='driver_version', default=None, help='Setting this overrides the default driver version,\n                    which is set to match the detected browser version.\n                    Major version only. Example: "--driver-version=114"\n                    (Only chromedriver and uc_driver are affected.)')
        parser.addoption('--pls', '--page_load_strategy', '--page-load-strategy', action='store', dest='page_load_strategy', choices=(constants.PageLoadStrategy.NORMAL, constants.PageLoadStrategy.EAGER, constants.PageLoadStrategy.NONE), default=None, help='This option sets Chrome\'s pageLoadStrategy.\n                    List of choices: "normal", "eager", "none".')
        parser.addoption('--headless', action='store_true', dest='headless', default=False, help='Using this option activates headless mode,\n                which is required on headless machines\n                UNLESS using a virtual display with Xvfb.\n                Default: False on Mac/Windows. True on Linux.')
        parser.addoption('--headless2', action='store_true', dest='headless2', default=False, help='This option activates the new headless mode,\n                    which supports Chromium extensions, and more,\n                    but is slower than the standard headless mode.')
        parser.addoption('--headed', '--gui', action='store_true', dest='headed', default=False, help='Using this makes Webdriver run web browsers with\n                    a GUI when running tests on Linux machines.\n                    (The default setting on Linux is headless.)\n                    (The default setting on Mac or Windows is headed.)')
        parser.addoption('--xvfb', action='store_true', dest='xvfb', default=False, help='Using this makes tests run headlessly using Xvfb\n                    instead of the browser\'s built-in headless mode.\n                    When using "--xvfb", the "--headless" option\n                    will no longer be enabled by default on Linux.\n                    Default: False. (Linux-ONLY!)')
        parser.addoption('--locale_code', '--locale-code', '--locale', action='store', dest='locale_code', default=None, help="Designates the Locale Code for the web browser.\n                    A Locale is a specific version of a spoken Language.\n                    The Locale alters visible text on supported websites.\n                    See: https://seleniumbase.io/help_docs/locale_codes/\n                    Default: None. (The web browser's default mode.)")
        parser.addoption('--interval', action='store', dest='interval', default=None, help='This globally overrides the default interval,\n                    (in seconds), of features that include autoplay\n                    functionality, such as tours and presentations.\n                    Overrides from methods take priority over this.\n                    (Headless Mode skips tours and presentations.)')
        parser.addoption('--start_page', '--start-page', '--url', action='store', dest='start_page', default=None, help='Designates the starting URL for the web browser\n                    when each test begins.\n                    Default: None.')
        parser.addoption('--time_limit', '--time-limit', '--timelimit', action='store', dest='time_limit', default=None, help='Use this to set a time limit per test, in seconds.\n                    If a test runs beyond the limit, it fails.')
        parser.addoption('--slow_mode', '--slow-mode', '--slowmo', '--slow', action='store_true', dest='slow_mode', default=False, help='Using this slows down the automation.')
        parser.addoption('--demo_mode', '--demo-mode', '--demo', action='store_true', dest='demo_mode', default=False, help='Using this slows down the automation and lets you\n                    visually see what the tests are actually doing.')
        parser.addoption('--demo_sleep', '--demo-sleep', action='store', dest='demo_sleep', default=None, help='Setting this overrides the Demo Mode sleep\n                    time that happens after browser actions.')
        parser.addoption('--highlights', action='store', dest='highlights', default=None, help='Setting this overrides the default number of\n                    highlight animation loops to have per call.')
        parser.addoption('--message_duration', '--message-duration', action='store', dest='message_duration', default=None, help='Setting this overrides the default time that\n                    messenger notifications remain visible when\n                    reaching assert statements during Demo Mode.')
        parser.addoption('--check_js', '--check-js', action='store_true', dest='js_checking_on', default=False, help='The option to check for JavaScript errors after\n                    every page load.')
        parser.addoption('--adblock', '--ad_block', '--ad-block', '--block_ads', '--block-ads', action='store_true', dest='ad_block_on', default=False, help='Using this makes WebDriver block display ads\n                    that are defined in ad_block_list.AD_BLOCK_LIST.')
        parser.addoption('--host_resolver_rules', '--host-resolver-rules', action='store', dest='host_resolver_rules', default=None, help='Use this option to set "host-resolver-rules".\n                    This lets you re-map traffic from any domain.\n                    Eg. "MAP www.google-analytics.com 0.0.0.0".\n                    Eg. "MAP * ~NOTFOUND , EXCLUDE myproxy".\n                    Eg. "MAP * 0.0.0.0 , EXCLUDE 127.0.0.1".\n                    Eg. "MAP *.google.com myproxy".\n                    Find more examples on these pages:\n                    (https://www.electronjs.org/docs/\n                     latest/api/command-line-switches)\n                    (https://www.chromium.org/developers/\n                     design-documents/network-stack/socks-proxy/)\n                    Use comma-separation for multiple host rules.')
        parser.addoption('--block_images', '--block-images', action='store_true', dest='block_images', default=False, help='Using this makes WebDriver block images from\n                    loading on web pages during tests.')
        parser.addoption('--do_not_track', '--do-not-track', action='store_true', dest='do_not_track', default=False, help="Indicate to websites that you don't want to be\n                    tracked. The browser will send an extra HTTP\n                    header each time it requests a web page.\n                    https://support.google.com/chrome/answer/2790761")
        parser.addoption('--verify_delay', '--verify-delay', action='store', dest='verify_delay', default=None, help='Setting this overrides the default wait time\n                    before each MasterQA verification pop-up.')
        parser.addoption('--recorder', '--record', '--rec', '--codegen', action='store_true', dest='recorder_mode', default=False, help='Using this enables the SeleniumBase Recorder,\n                    which records browser actions for converting\n                    into SeleniumBase scripts.')
        parser.addoption('--rec-behave', '--rec-gherkin', action='store_true', dest='rec_behave', default=False, help='Not only enables the SeleniumBase Recorder,\n                    but also saves recorded actions into the\n                    behave-gerkin format, which includes a\n                    feature file, an imported steps file,\n                    and the environment.py file.')
        parser.addoption('--rec-sleep', '--record-sleep', action='store_true', dest='record_sleep', default=False, help='If Recorder Mode is enabled,\n                    records sleep(seconds) calls.')
        parser.addoption('--rec-print', action='store_true', dest='rec_print', default=False, help='If Recorder Mode is enabled,\n                    prints output after tests end.')
        parser.addoption('--disable_js', '--disable-js', action='store_true', dest='disable_js', default=False, help='The option to disable JavaScript on web pages.\n                    Warning: Most web pages will stop working!')
        parser.addoption('--disable_csp', '--disable-csp', '--no_csp', '--no-csp', '--dcsp', action='store_true', dest='disable_csp', default=False, help='Using this disables the Content Security Policy of\n                    websites, which may interfere with some features of\n                    SeleniumBase, such as loading custom JavaScript\n                    libraries for various testing actions.\n                    Setting this to True (--disable-csp) overrides the\n                    value set in seleniumbase/config/settings.py')
        parser.addoption('--disable_ws', '--disable-ws', '--disable-web-security', action='store_true', dest='disable_ws', default=False, help='Using this disables the "Web Security" feature of\n                    Chrome and Chromium-based browsers such as Edge.')
        parser.addoption('--enable_ws', '--enable-ws', '--enable-web-security', action='store_true', dest='enable_ws', default=False, help='Using this enables the "Web Security" feature of\n                    Chrome and Chromium-based browsers such as Edge.')
        parser.addoption('--enable_sync', '--enable-sync', action='store_true', dest='enable_sync', default=False, help='Using this enables the "Chrome Sync" feature.')
        parser.addoption('--use_auto_ext', '--use-auto-ext', '--auto-ext', action='store_true', dest='use_auto_ext', default=False, help="(DEPRECATED) - Enable the automation extension.\n                    It's not required, but some commands & advanced\n                    features may need it.")
        parser.addoption('--undetected', '--undetectable', '--uc', action='store_true', dest='undetectable', default=False, help='Using this option makes chromedriver undetectable\n                    to websites that use anti-bot services to block\n                    automation tools from navigating them freely.')
        parser.addoption('--uc_cdp_events', '--uc-cdp-events', '--uc-cdp', action='store_true', dest='uc_cdp_events', default=None, help='Captures CDP events during Undetectable Mode runs.\n                    Then you can add a listener to perform actions on\n                    received data, such as printing it to the console:\n                        from pprint import pformat\n                        self.driver.add_cdp_listener(\n                            "*", lambda data: print(pformat(data))\n                        )\n                        self.open(URL)')
        parser.addoption('--uc_subprocess', '--uc-subprocess', '--uc-sub', action='store_true', dest='uc_subprocess', default=None, help='(DEPRECATED) - (UC Mode always uses this now.)\n                    Use undetectable-chromedriver as a subprocess,\n                    which can help avoid issues that might result.')
        parser.addoption('--no_sandbox', '--no-sandbox', action='store_true', dest='no_sandbox', default=False, help='(DEPRECATED) - "--no-sandbox" is always used now.\n                    Using this enables the "No Sandbox" feature.\n                    (This setting is now always enabled by default.)')
        parser.addoption('--disable_gpu', '--disable-gpu', action='store_true', dest='disable_gpu', default=False, help='(DEPRECATED) - GPU is disabled if no swiftshader.\n                    Using this enables the "Disable GPU" feature.\n                    (GPU is disabled by default if swiftshader off.)')
        parser.addoption('--log_cdp', '--log-cdp', '--log_cdp_events', '--log-cdp-events', action='store_true', dest='log_cdp_events', default=None, help='Capture CDP events. Then you can print them.\n                    Eg. print(driver.get_log("performance"))')
        parser.addoption('--remote_debug', '--remote-debug', '--remote-debugger', '--remote_debugger', action='store_true', dest='remote_debug', default=False, help="This syncs the browser to Chromium's remote debugger.\n                    To access the remote debugging interface, go to:\n                    chrome://inspect/#devices while tests are running.\n                    The previous URL was at: http://localhost:9222/\n                    Info: chromedevtools.github.io/devtools-protocol/")
        parser.addoption('--enable_3d_apis', '--enable-3d-apis', action='store_true', dest='enable_3d_apis', default=False, help='Using this enables WebGL and 3D APIs.')
        parser.addoption('--swiftshader', action='store_true', dest='swiftshader', default=False, help='Using this enables the "--use-gl=swiftshader"\n                    feature when running tests on Chrome.')
        parser.addoption('--incognito', '--incognito_mode', '--incognito-mode', action='store_true', dest='incognito', default=False, help="Using this enables Chrome's Incognito mode.")
        parser.addoption('--guest', '--guest_mode', '--guest-mode', action='store_true', dest='guest_mode', default=False, help="Using this enables Chrome's Guest mode.")
        parser.addoption('--dark', '--dark_mode', '--dark-mode', action='store_true', dest='dark_mode', default=False, help="Using this enables Chrome's Dark mode.")
        parser.addoption('--devtools', '--open_devtools', '--open-devtools', action='store_true', dest='devtools', default=False, help="Using this opens Chrome's DevTools.")
        parser.addoption('--disable-beforeunload', '--disable_beforeunload', action='store_true', dest='_disable_beforeunload', default=False, help='The option to disable the "beforeunload" event\n                    on Chromium browsers (Chrome or Edge).\n                    This is already the default Firefox option.')
        parser.addoption('--window-size', '--window_size', action='store', dest='window_size', default=None, help='The option to set the default window "width,height".\n                    Format: A comma-separated string with the 2 values.\n                    Example: "1200,800"\n                    Default: None. (Will use default values if None)')
        parser.addoption('--maximize_window', '--maximize-window', '--maximize', '--fullscreen', action='store_true', dest='maximize_option', default=False, help='The option to start with a maximized browser window.\n                    (Overrides the "window-size" option if used.)')
        parser.addoption('--screenshot', '--save_screenshot', '--save-screenshot', '--ss', action='store_true', dest='save_screenshot', default=False, help='Save a screenshot at the end of every test.\n                    By default, this is only done for failures.\n                    Will be saved in the "latest_logs/" folder.')
        parser.addoption('--no-screenshot', '--no_screenshot', '--ns', action='store_true', dest='no_screenshot', default=False, help='No screenshots saved unless tests directly ask it.\n                    This changes default behavior where screenshots are\n                    saved for test failures and pytest-html reports.')
        parser.addoption('--visual_baseline', '--visual-baseline', action='store_true', dest='visual_baseline', default=False, help='Setting this resets the visual baseline for\n                    Automated Visual Testing with SeleniumBase.\n                    When a test calls self.check_window(), it will\n                    rebuild its files in the visual_baseline folder.')
        parser.addoption('--wire', action='store_true', dest='use_wire', default=False, help="Use selenium-wire's webdriver for selenium webdriver.")
        parser.addoption('--external_pdf', '--external-pdf', action='store_true', dest='external_pdf', default=False, help='This option sets the following on Chromium:\n                    "plugins.always_open_pdf_externally": True,\n                    which causes opened PDF URLs to download immediately,\n                    instead of being displayed in the browser window.')
        parser.addoption('--timeout_multiplier', '--timeout-multiplier', action='store', dest='timeout_multiplier', default=None, help='Setting this overrides the default timeout\n                    by the multiplier when waiting for page elements.\n                    Unused when tests override the default value.')

    def configure(self, options, conf):
        if False:
            while True:
                i = 10
        super().configure(options, conf)
        self.enabled = True
        self.options = options
        self.headless_active = False
        sb_config.headless_active = False
        sb_config.is_nosetest = True
        proxy_helper.remove_proxy_zip_if_present()

    def beforeTest(self, test):
        if False:
            print('Hello World!')
        browser = self.options.browser
        test.test.browser = browser
        test.test.headless = None
        test.test.headless2 = None
        sb_config._browser_shortcut = None
        sys_argv = sys.argv
        browser_changes = 0
        browser_set = None
        browser_text = None
        browser_list = []
        if '--browser=chrome' in sys_argv or '--browser chrome' in sys_argv:
            browser_changes += 1
            browser_set = 'chrome'
            browser_list.append('--browser=chrome')
        if '--browser=edge' in sys_argv or '--browser edge' in sys_argv:
            browser_changes += 1
            browser_set = 'edge'
            browser_list.append('--browser=edge')
        if '--browser=firefox' in sys_argv or '--browser firefox' in sys_argv:
            browser_changes += 1
            browser_set = 'firefox'
            browser_list.append('--browser=firefox')
        if '--browser=safari' in sys_argv or '--browser safari' in sys_argv:
            browser_changes += 1
            browser_set = 'safari'
            browser_list.append('--browser=safari')
        if '--browser=ie' in sys_argv or '--browser ie' in sys_argv:
            browser_changes += 1
            browser_set = 'ie'
            browser_list.append('--browser=ie')
        if '--browser=remote' in sys_argv or '--browser remote' in sys_argv:
            browser_changes += 1
            browser_set = 'remote'
            browser_list.append('--browser=remote')
        browser_text = browser_set
        if '--chrome' in sys_argv and (not browser_set == 'chrome'):
            browser_changes += 1
            browser_text = 'chrome'
            sb_config._browser_shortcut = 'chrome'
            browser_list.append('--chrome')
        if '--edge' in sys_argv and (not browser_set == 'edge'):
            browser_changes += 1
            browser_text = 'edge'
            sb_config._browser_shortcut = 'edge'
            browser_list.append('--edge')
        if '--firefox' in sys_argv and (not browser_set == 'firefox'):
            browser_changes += 1
            browser_text = 'firefox'
            sb_config._browser_shortcut = 'firefox'
            browser_list.append('--firefox')
        if '--ie' in sys_argv and (not browser_set == 'ie'):
            browser_changes += 1
            browser_text = 'ie'
            sb_config._browser_shortcut = 'ie'
            browser_list.append('--ie')
        if '--safari' in sys_argv and (not browser_set == 'safari'):
            browser_changes += 1
            browser_text = 'safari'
            sb_config._browser_shortcut = 'safari'
            browser_list.append('--safari')
        if browser_changes > 1:
            message = '\n\n  TOO MANY browser types were entered!'
            message += '\n  There were %s found:\n  >  %s' % (browser_changes, ', '.join(browser_list))
            message += '\n  ONLY ONE default browser is allowed!'
            message += '\n  Select a single browser & try again!\n'
            raise Exception(message)
        if browser_text:
            browser = browser_text
        if self.options.recorder_mode and browser not in ['chrome', 'edge']:
            message = '\n\n  Recorder Mode ONLY supports Chrome and Edge!\n  (Your browser choice was: "%s")\n' % browser
            raise Exception(message)
        window_size = self.options.window_size
        if window_size:
            if window_size.count(',') != 1:
                message = '\n\n  window_size expects a "width,height" string!\n  (Your input was: "%s")\n' % window_size
                raise Exception(message)
            window_size = window_size.replace(' ', '')
            width = None
            height = None
            try:
                width = int(window_size.split(',')[0])
                height = int(window_size.split(',')[1])
            except Exception:
                message = '\n\n  Expecting integer values for "width,height"!\n  (window_size input was: "%s")\n' % window_size
                raise Exception(message)
            settings.CHROME_START_WIDTH = width
            settings.CHROME_START_HEIGHT = height
            settings.HEADLESS_START_WIDTH = width
            settings.HEADLESS_START_HEIGHT = height
        test.test.is_nosetest = True
        test.test.is_behave = False
        test.test.is_pytest = False
        test.test.is_context_manager = False
        sb_config.is_nosetest = True
        sb_config.is_behave = False
        sb_config.is_pytest = False
        sb_config.is_context_manager = False
        test.test.browser = self.options.browser
        if sb_config._browser_shortcut:
            self.options.browser = sb_config._browser_shortcut
            test.test.browser = sb_config._browser_shortcut
        test.test.cap_file = self.options.cap_file
        test.test.cap_string = self.options.cap_string
        test.test.headless = self.options.headless
        test.test.headless2 = self.options.headless2
        if test.test.headless and test.test.browser == 'safari':
            test.test.headless = False
        if test.test.headless2 and test.test.browser == 'firefox':
            test.test.headless2 = False
            test.test.headless = True
            self.options.headless2 = False
            self.options.headless = True
        elif test.test.browser not in ['chrome', 'edge']:
            test.test.headless2 = False
            self.options.headless2 = False
        test.test.headed = self.options.headed
        test.test.xvfb = self.options.xvfb
        test.test.locale_code = self.options.locale_code
        test.test.interval = self.options.interval
        test.test.start_page = self.options.start_page
        if self.options.skip_js_waits:
            settings.SKIP_JS_WAITS = True
        test.test.protocol = self.options.protocol
        test.test.servername = self.options.servername
        test.test.port = self.options.port
        test.test.user_data_dir = self.options.user_data_dir
        test.test.extension_zip = self.options.extension_zip
        test.test.extension_dir = self.options.extension_dir
        test.test.binary_location = self.options.binary_location
        test.test.driver_version = self.options.driver_version
        test.test.page_load_strategy = self.options.page_load_strategy
        test.test.chromium_arg = self.options.chromium_arg
        test.test.firefox_arg = self.options.firefox_arg
        test.test.firefox_pref = self.options.firefox_pref
        test.test.proxy_string = self.options.proxy_string
        test.test.proxy_bypass_list = self.options.proxy_bypass_list
        test.test.proxy_pac_url = self.options.proxy_pac_url
        test.test.multi_proxy = self.options.multi_proxy
        test.test.user_agent = self.options.user_agent
        test.test.mobile_emulator = self.options.mobile_emulator
        test.test.device_metrics = self.options.device_metrics
        test.test.time_limit = self.options.time_limit
        test.test.slow_mode = self.options.slow_mode
        test.test.demo_mode = self.options.demo_mode
        test.test.demo_sleep = self.options.demo_sleep
        test.test.highlights = self.options.highlights
        test.test.message_duration = self.options.message_duration
        test.test.js_checking_on = self.options.js_checking_on
        test.test.ad_block_on = self.options.ad_block_on
        test.test.host_resolver_rules = self.options.host_resolver_rules
        test.test.block_images = self.options.block_images
        test.test.do_not_track = self.options.do_not_track
        test.test.verify_delay = self.options.verify_delay
        test.test.recorder_mode = self.options.recorder_mode
        test.test.recorder_ext = self.options.recorder_mode
        test.test.rec_behave = self.options.rec_behave
        test.test.rec_print = self.options.rec_print
        test.test.record_sleep = self.options.record_sleep
        if self.options.rec_print:
            test.test.recorder_mode = True
            test.test.recorder_ext = True
        elif self.options.rec_behave:
            test.test.recorder_mode = True
            test.test.recorder_ext = True
        elif self.options.record_sleep:
            test.test.recorder_mode = True
            test.test.recorder_ext = True
        test.test.disable_js = self.options.disable_js
        test.test.disable_csp = self.options.disable_csp
        test.test.disable_ws = self.options.disable_ws
        test.test.enable_ws = self.options.enable_ws
        if not self.options.disable_ws:
            test.test.enable_ws = True
        test.test.enable_sync = self.options.enable_sync
        test.test.use_auto_ext = self.options.use_auto_ext
        test.test.undetectable = self.options.undetectable
        test.test.uc_cdp_events = self.options.uc_cdp_events
        test.test.log_cdp_events = self.options.log_cdp_events
        if test.test.uc_cdp_events and (not test.test.undetectable):
            test.test.undetectable = True
        test.test.uc_subprocess = self.options.uc_subprocess
        if test.test.uc_subprocess and (not test.test.undetectable):
            test.test.undetectable = True
        test.test.no_sandbox = self.options.no_sandbox
        test.test.disable_gpu = self.options.disable_gpu
        test.test.remote_debug = self.options.remote_debug
        test.test.enable_3d_apis = self.options.enable_3d_apis
        test.test._swiftshader = self.options.swiftshader
        test.test.incognito = self.options.incognito
        test.test.guest_mode = self.options.guest_mode
        test.test.dark_mode = self.options.dark_mode
        test.test.devtools = self.options.devtools
        test.test._disable_beforeunload = self.options._disable_beforeunload
        test.test.window_size = self.options.window_size
        test.test.maximize_option = self.options.maximize_option
        if self.options.save_screenshot and self.options.no_screenshot:
            self.options.save_screenshot = False
        test.test.save_screenshot_after_test = self.options.save_screenshot
        test.test.no_screenshot_after_test = self.options.no_screenshot
        test.test.visual_baseline = self.options.visual_baseline
        test.test.use_wire = self.options.use_wire
        test.test.external_pdf = self.options.external_pdf
        test.test.timeout_multiplier = self.options.timeout_multiplier
        test.test.dashboard = False
        test.test._multithreaded = False
        test.test._reuse_session = False
        sb_config.no_screenshot = test.test.no_screenshot_after_test
        if test.test.servername != 'localhost':
            if str(self.options.port) == '443':
                test.test.protocol = 'https'
        if shared_utils.is_linux() and (not self.options.headed) and (not self.options.headless) and (not self.options.headless2) and (not self.options.xvfb):
            print('(Linux uses --headless by default. To override, use --headed / --gui. For Xvfb mode instead, use --xvfb. Or you can hide this info by using --headless / --headless2.)')
            self.options.headless = True
            test.test.headless = True
        if self.options.use_wire and self.options.undetectable:
            print("\nSeleniumBase doesn't support mixing --uc with --wire mode.\nIf you need both, override get_new_driver() from BaseCase:\nhttps://seleniumbase.io/help_docs/syntax_formats/#sb_sf_09\n(Only UC Mode without Wire Mode will be used for this run)\n")
            self.options.use_wire = False
            test.test.use_wire = False
        if self.options.recorder_mode and self.options.headless:
            self.options.headless = False
            self.options.headless2 = True
            test.test.headless = False
            test.test.headless2 = True
        if not self.options.headless and (not self.options.headless2):
            self.options.headed = True
            test.test.headed = True
        sb_config._virtual_display = None
        sb_config.headless_active = False
        self.headless_active = False
        if shared_utils.is_linux() and (not self.options.headed or self.options.xvfb):
            width = settings.HEADLESS_START_WIDTH
            height = settings.HEADLESS_START_HEIGHT
            try:
                from sbvirtualdisplay import Display
                self._xvfb_display = Display(visible=0, size=(width, height))
                self._xvfb_display.start()
                sb_config._virtual_display = self._xvfb_display
                self.headless_active = True
                sb_config.headless_active = True
            except Exception:
                pass
        sb_config._is_timeout_changed = False
        sb_config._SMALL_TIMEOUT = settings.SMALL_TIMEOUT
        sb_config._LARGE_TIMEOUT = settings.LARGE_TIMEOUT
        sb_config._context_of_runner = False
        sb_config.mobile_emulator = self.options.mobile_emulator
        sb_config.proxy_driver = self.options.proxy_driver
        sb_config.multi_proxy = self.options.multi_proxy
        self.driver = None
        test.test.driver = self.driver

    def finalize(self, result):
        if False:
            return 10
        'This runs after all tests have completed with nosetests.'
        if hasattr(sb_config, 'multi_proxy') and (not sb_config.multi_proxy) or not hasattr(sb_config, 'multi_proxy'):
            proxy_helper.remove_proxy_zip_if_present()

    def afterTest(self, test):
        if False:
            print('Hello World!')
        try:
            if not shared_utils.is_windows() or test.test.browser == 'ie' or self.driver.service.process:
                self.driver.quit()
        except AttributeError:
            pass
        except Exception:
            pass
        try:
            if hasattr(self, '_xvfb_display') and self._xvfb_display and hasattr(self._xvfb_display, 'stop'):
                self.headless_active = False
                sb_config.headless_active = False
                self._xvfb_display.stop()
                self._xvfb_display = None
            if hasattr(sb_config, '_virtual_display') and sb_config._virtual_display and hasattr(sb_config._virtual_display, 'stop'):
                sb_config._virtual_display.stop()
                sb_config._virtual_display = None
        except Exception:
            pass
from behave import step

@step('Open the RealWorld Login Page')
def go_to_realworld(context):
    if False:
        for i in range(10):
            print('nop')
    sb = context.sb
    context.sb.open('https://seleniumbase.io/realworld/login')
    sb.clear_session_storage()

@step('Login to the RealWorld App')
def login_to_realworld(context):
    if False:
        print('Hello World!')
    sb = context.sb
    sb.type('#username', 'demo_user')
    sb.type('#password', 'secret_pass')
    sb.enter_mfa_code('#totpcode', 'GAXG2MTEOR3DMMDG')

@step('Highlight element {selector}')
def highlight(context, selector):
    if False:
        for i in range(10):
            print('nop')
    if selector.startswith('"') or selector.startswith("'"):
        selector = selector[1:]
    if selector.endswith('"') or selector.endswith("'"):
        selector = selector[:-1]
    sb = context.sb
    sb.highlight(selector)

@step('Click element {selector}')
def click(context, selector):
    if False:
        print('Hello World!')
    if selector.startswith('"') or selector.startswith("'"):
        selector = selector[1:]
    if selector.endswith('"') or selector.endswith("'"):
        selector = selector[:-1]
    sb = context.sb
    sb.click(selector)

@step('Click link {link}')
def click_link(context, link):
    if False:
        for i in range(10):
            print('nop')
    if link.startswith('"') or link.startswith("'"):
        link = link[1:]
    if link.endswith('"') or link.endswith("'"):
        link = link[:-1]
    sb = context.sb
    sb.click_link(link)

@step('Save a screenshot to the logs')
def save_screenshot_to_logs(context):
    if False:
        while True:
            i = 10
    sb = context.sb
    sb.save_screenshot_to_logs()

@step('Assert element {selector}')
def assert_element(context, selector):
    if False:
        i = 10
        return i + 15
    if selector.startswith('"') or selector.startswith("'"):
        selector = selector[1:]
    if selector.endswith('"') or selector.endswith("'"):
        selector = selector[:-1]
    sb = context.sb
    sb.assert_element(selector)

@step('Assert text {text} in {selector}')
def assert_text_in_selector(context, text, selector):
    if False:
        for i in range(10):
            print('nop')
    if text.startswith('"') or text.startswith("'"):
        text = text[1:]
    if text.endswith('"') or text.endswith("'"):
        text = text[:-1]
    if selector.startswith('"') or selector.startswith("'"):
        selector = selector[1:]
    if selector.endswith('"') or selector.endswith("'"):
        selector = selector[:-1]
    sb = context.sb
    sb.assert_text(text, selector)

@step('Assert text {text}')
def assert_text(context, text):
    if False:
        return 10
    if text.startswith('"') or text.startswith("'"):
        text = text[1:]
    if text.endswith('"') or text.endswith("'"):
        text = text[:-1]
    sb = context.sb
    sb.assert_text(text)

@step('Assert exact text {text} in {selector}')
def assert_exact_text(context, text, selector):
    if False:
        i = 10
        return i + 15
    if text.startswith('"') or text.startswith("'"):
        text = text[1:]
    if text.endswith('"') or text.endswith("'"):
        text = text[:-1]
    if selector.startswith('"') or selector.startswith("'"):
        selector = selector[1:]
    if selector.endswith('"') or selector.endswith("'"):
        selector = selector[:-1]
    sb = context.sb
    sb.assert_exact_text(text, selector)
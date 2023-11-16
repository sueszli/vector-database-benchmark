from behave import step

@step('Open the Fail Page')
def go_to_error_page(context):
    if False:
        while True:
            i = 10
    context.sb.open('https://seleniumbase.io/error_page/')

@step('Fail test on purpose')
def fail_on_purpose(context):
    if False:
        print('Hello World!')
    context.sb.fail('This test fails on purpose!')
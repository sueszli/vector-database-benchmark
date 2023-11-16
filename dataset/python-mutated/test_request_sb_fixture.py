def test_request_sb_fixture(request):
    if False:
        i = 10
        return i + 15
    sb = request.getfixturevalue('sb')
    sb.open('https://seleniumbase.io/demo_page')
    sb.assert_text('SeleniumBase', '#myForm h2')
    sb.assert_element('input#myTextInput')
    sb.type('#myTextarea', 'This is me')
    sb.click('#myButton')
    sb.tearDown()

class Test_Request_Fixture:

    def test_request_sb_fixture_in_class(self, request):
        if False:
            while True:
                i = 10
        sb = request.getfixturevalue('sb')
        sb.open('https://seleniumbase.io/demo_page')
        sb.assert_element('input#myTextInput')
        sb.type('#myTextarea', 'Automated')
        sb.assert_text('This Text is Green', '#pText')
        sb.click('#myButton')
        sb.assert_text('This Text is Purple', '#pText')
        sb.tearDown()
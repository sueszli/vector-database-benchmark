def test_sb_fixture_with_no_class(sb):
    if False:
        for i in range(10):
            print('nop')
    sb.open('seleniumbase.io/simple/login')
    sb.type('#username', 'demo_user')
    sb.type('#password', 'secret_pass')
    sb.click('a:contains("Sign in")')
    sb.assert_exact_text('Welcome!', 'h1')
    sb.assert_element('img#image1')
    sb.highlight('#image1')
    sb.click_link('Sign out')
    sb.assert_text('signed out', '#top_message')

class Test_SB_Fixture:

    def test_sb_fixture_inside_class(self, sb):
        if False:
            i = 10
            return i + 15
        sb.open('seleniumbase.io/simple/login')
        sb.type('#username', 'demo_user')
        sb.type('#password', 'secret_pass')
        sb.click('a:contains("Sign in")')
        sb.assert_exact_text('Welcome!', 'h1')
        sb.assert_element('img#image1')
        sb.highlight('#image1')
        sb.click_link('Sign out')
        sb.assert_text('signed out', '#top_message')
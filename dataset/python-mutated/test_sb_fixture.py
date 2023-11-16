def test_sb_fixture_with_no_class(sb):
    if False:
        return 10
    sb.open('seleniumbase.io/help_docs/install/')
    sb.type('input[aria-label="Search"]', 'GUI Commander')
    sb.click('mark:contains("Commander")')
    sb.assert_title_contains('GUI / Commander')

class Test_SB_Fixture:

    def test_sb_fixture_inside_class(self, sb):
        if False:
            for i in range(10):
                print('nop')
        sb.open('seleniumbase.io/help_docs/install/')
        sb.type('input[aria-label="Search"]', 'GUI Commander')
        sb.click('mark:contains("Commander")')
        sb.assert_title_contains('GUI / Commander')
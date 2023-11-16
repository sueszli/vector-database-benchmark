from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class AngularMaterialPaginatorTests(BaseCase):

    def test_pagination(self):
        if False:
            return 10
        self.open('https://material.angular.io/components/paginator/examples')
        self.click('mat-select > div')
        self.click('#mat-option-0')
        self.click('button[aria-label="Next page"]')
        self.assert_exact_text('6 – 10 of 50', '.mat-mdc-paginator-range-label')
        self.click('button[aria-label="Previous page"]')
        self.assert_exact_text('1 – 5 of 50', '.mat-mdc-paginator-range-label')
        self.click('mat-select > div')
        self.click('#mat-option-1')
        self.assert_exact_text('1 – 10 of 50', '.mat-mdc-paginator-range-label')
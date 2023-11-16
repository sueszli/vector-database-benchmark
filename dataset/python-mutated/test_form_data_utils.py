from django.test import SimpleTestCase
from wagtail.test.utils.form_data import querydict_from_html

class TestQueryDictFromHTML(SimpleTestCase):
    html = '\n    <form id="personal-details">\n        <input type="hidden" name="csrfmiddlewaretoken" value="Z783HTL5Bc2J54WhAtEeR3eefM1FBkq0EbTfNnYnepFGuJSOfvosFvwjeKYtMwFr">\n        <input type="hidden" name="no_value_input">\n        <input type="hidden" value="no name input">\n        <div class="mt-8 max-w-md">\n            <div class="grid grid-cols-1 gap-6">\n                <label class="block">\n                    <span class="text-gray-700">Full name</span>\n                    <input type="text" name="name" value="Jane Doe" class="mt-1 block w-full" placeholder="">\n                </label>\n                <label class="block">\n                    <span class="text-gray-700">Email address</span>\n                    <input type="email" name="email" class="mt-1 block w-full" value="jane@example.com" placeholder="name@example.com">\n                </label>\n            </div>\n        </div>\n    </form>\n    <form id="event-details">\n        <div class="mt-8 max-w-md">\n            <div class="grid grid-cols-1 gap-6">\n                <label class="block">\n                    <span class="text-gray-700">When is your event?</span>\n                    <input type="date" name="date" class="mt-1 block w-full" value="2023-01-01">\n                </label>\n                <label class="block">\n                    <span class="text-gray-700">What type of event is it?</span>\n                    <select name="event_type" class="block w-full mt-1">\n                        <option value="corporate">Corporate event</option>\n                        <option value="wedding">Wedding</option>\n                        <option value="birthday">Birthday</option>\n                        <option value="other" selected>Other</option>\n                    </select>\n                </label>\n                <label class="block">\n                    <span class="text-gray-700">What age groups is it suitable for?</span>\n                    <select name="ages" class="block w-full mt-1" multiple>\n                        <option>Infants</option>\n                        <option>Children</option>\n                        <option>Teenagers</option>\n                        <option selected>18-30</option>\n                        <option selected>30-50</option>\n                        <option>50-70</option>\n                        <option>70+</option>\n                    </select>\n                </label>\n            </div>\n        </div>\n    </form>\n    <form id="market-research">\n        <div class="mt-8 max-w-md">\n            <div class="grid grid-cols-1 gap-6">\n                <fieldset class="block">\n                    <legend>How many pets do you have?</legend>\n                    <div class="radio-list">\n                        <div class="radio">\n                            <label>\n                                <input type="radio" name="pets" value="0" />\n                                None\n                            </label>\n                        </div>\n                        <div class="radio">\n                            <label>\n                                <input type="radio" name="pets" value="1" />\n                                One\n                            </label>\n                        </div>\n                        <div class="radio">\n                            <label>\n                                <input type="radio" name="pets" value="2" checked />\n                                Two\n                            </label>\n                        </div>\n                        <div class="radio">\n                            <label>\n                                <input type="radio" name="pets" value="3+" />\n                                Three or more\n                            </label>\n                        </div>\n                    </div>\n                </fieldset>\n                <fieldset class="block">\n                    <legend>Which two colours do you like best?</legend>\n                    <div class="checkbox-list">\n                        <div class="checkbox">\n                            <label>\n                                <input type="checkbox" name="colours" value="cyan">\n                                Cyan\n                            </label>\n                        </div>\n                        <div class="checkbox">\n                            <label>\n                                <input type="checkbox" name="colours" value="magenta" checked />\n                                Magenta\n                            </label>\n                        </div>\n                        <div class="checkbox">\n                            <label>\n                                <input type="checkbox" name="colours" value="yellow" />\n                                Yellow\n                            </label>\n                        </div>\n                        <div class="checkbox">\n                            <label>\n                                <input type="checkbox" name="colours" value="black" checked />\n                                Black\n                            </label>\n                        </div>\n                        <div class="checkbox">\n                            <label>\n                                <input type="checkbox" name="colours" value="white" />\n                                White\n                            </label>\n                        </div>\n                    </div>\n                </fieldset>\n                <label class="block">\n                    <span class="text-gray-700">Tell us what you love</span>\n                    <textarea name="love" class="mt-1 block w-full" rows="3">Comic books</textarea>\n                </label>\n            </div>\n        </div>\n    </form>\n    '
    personal_details = [('no_value_input', ['']), ('name', ['Jane Doe']), ('email', ['jane@example.com'])]
    event_details = [('date', ['2023-01-01']), ('event_type', ['other']), ('ages', ['18-30', '30-50'])]
    market_research = [('pets', ['2']), ('colours', ['magenta', 'black']), ('love', ['Comic books'])]

    def test_html_only(self):
        if False:
            for i in range(10):
                print('nop')
        result = querydict_from_html(self.html)
        self.assertEqual(list(result.lists()), self.personal_details)

    def test_include_csrf(self):
        if False:
            for i in range(10):
                print('nop')
        result = querydict_from_html(self.html, exclude_csrf=False)
        expected_result = [('csrfmiddlewaretoken', ['Z783HTL5Bc2J54WhAtEeR3eefM1FBkq0EbTfNnYnepFGuJSOfvosFvwjeKYtMwFr'])] + self.personal_details
        self.assertEqual(list(result.lists()), expected_result)

    def test_form_index(self):
        if False:
            print('Hello World!')
        for (index, expected_data) in ((0, self.personal_details), ('2', self.market_research), (1, self.event_details)):
            result = querydict_from_html(self.html, form_index=index)
            self.assertEqual(list(result.lists()), expected_data)

    def test_form_id(self):
        if False:
            while True:
                i = 10
        for (id, expected_data) in (('event-details', self.event_details), ('personal-details', self.personal_details), ('market-research', self.market_research)):
            result = querydict_from_html(self.html, form_id=id)
            self.assertEqual(list(result.lists()), expected_data)

    def test_invalid_form_id(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            querydict_from_html(self.html, form_id='invalid-id')

    def test_invalid_index(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            querydict_from_html(self.html, form_index=5)
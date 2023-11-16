import psycopg2
from odoo.models import BaseModel
from odoo.tests.common import TransactionCase
from odoo.tools import mute_logger
import odoo.osv.expression as expression

class TestExpression(TransactionCase):

    def test_00_in_not_in_m2m(self):
        if False:
            while True:
                i = 10
        categories = self.env['res.partner.category']
        cat_a = categories.create({'name': 'test_expression_category_A'})
        cat_b = categories.create({'name': 'test_expression_category_B'})
        partners = self.env['res.partner']
        a = partners.create({'name': 'test_expression_partner_A', 'category_id': [(6, 0, [cat_a.id])]})
        b = partners.create({'name': 'test_expression_partner_B', 'category_id': [(6, 0, [cat_b.id])]})
        ab = partners.create({'name': 'test_expression_partner_AB', 'category_id': [(6, 0, [cat_a.id, cat_b.id])]})
        c = partners.create({'name': 'test_expression_partner_C'})
        with_a = partners.search([('category_id', 'in', [cat_a.id])])
        self.assertEqual(a + ab, with_a, 'Search for category_id in cat_a failed.')
        with_b = partners.search([('category_id', 'in', [cat_b.id])])
        self.assertEqual(b + ab, with_b, 'Search for category_id in cat_b failed.')
        with_a_or_b = partners.search([('category_id', 'in', [cat_a.id, cat_b.id])])
        self.assertEqual(a + b + ab, with_a_or_b, 'Search for category_id contains cat_a or cat_b failed.')
        with_a_or_with_b = partners.search(['|', ('category_id', 'in', [cat_a.id]), ('category_id', 'in', [cat_b.id])])
        self.assertEqual(a + b + ab, with_a_or_with_b, 'Search for category_id contains cat_a or contains cat_b failed.')
        with_a_and_b = partners.search([('category_id', 'in', [cat_a.id]), ('category_id', 'in', [cat_b.id])])
        self.assertEqual(ab, with_a_and_b, 'Search for category_id contains cat_a and cat_b failed.')
        without_a_or_b = partners.search([('category_id', 'not in', [cat_a.id, cat_b.id])])
        self.assertFalse(without_a_or_b & a + b + ab, "Search for category_id doesn't contain cat_a or cat_b failed (1).")
        self.assertTrue(c in without_a_or_b, "Search for category_id doesn't contain cat_a or cat_b failed (2).")
        without_a_and_without_b = partners.search([('category_id', 'not in', [cat_a.id]), ('category_id', 'not in', [cat_b.id])])
        self.assertFalse(without_a_and_without_b & a + b + ab, "Search for category_id doesn't contain cat_a and cat_b failed (1).")
        self.assertTrue(c in without_a_and_without_b, "Search for category_id doesn't contain cat_a and cat_b failed (2).")
        without_a = partners.search([('category_id', 'not in', [cat_a.id])])
        self.assertTrue(a not in without_a, "Search for category_id doesn't contain cat_a failed (1).")
        self.assertTrue(ab not in without_a, "Search for category_id doesn't contain cat_a failed (2).")
        self.assertLessEqual(b + c, without_a, "Search for category_id doesn't contain cat_a failed (3).")
        without_b = partners.search([('category_id', 'not in', [cat_b.id])])
        self.assertTrue(b not in without_b, "Search for category_id doesn't contain cat_b failed (1).")
        self.assertTrue(ab not in without_b, "Search for category_id doesn't contain cat_b failed (2).")
        self.assertLessEqual(a + c, without_b, "Search for category_id doesn't contain cat_b failed (3).")

    def test_05_not_str_m2m(self):
        if False:
            i = 10
            return i + 15
        partners = self.env['res.partner']
        categories = self.env['res.partner.category']
        cids = {}
        for name in 'A B AB'.split():
            cids[name] = categories.create({'name': name}).id
        partners_config = {'0': [], 'a': [cids['A']], 'b': [cids['B']], 'ab': [cids['AB']], 'a b': [cids['A'], cids['B']], 'b ab': [cids['B'], cids['AB']]}
        pids = {}
        for (name, cat_ids) in partners_config.iteritems():
            pids[name] = partners.create({'name': name, 'category_id': [(6, 0, cat_ids)]}).id
        base_domain = [('id', 'in', pids.values())]

        def test(op, value, expected):
            if False:
                print('Hello World!')
            found_ids = partners.search(base_domain + [('category_id', op, value)]).ids
            expected_ids = [pids[name] for name in expected]
            self.assertItemsEqual(found_ids, expected_ids, '%s %r should return %r' % (op, value, expected))
        test('=', 'A', ['a', 'a b'])
        test('!=', 'B', ['0', 'a', 'ab'])
        test('like', 'A', ['a', 'ab', 'a b', 'b ab'])
        test('not ilike', 'B', ['0', 'a'])
        test('not like', 'AB', ['0', 'a', 'b', 'a b'])

    def test_10_hierarchy_in_m2m(self):
        if False:
            i = 10
            return i + 15
        Partner = self.env['res.partner']
        Category = self.env['res.partner.category']
        partners = Partner.search([('category_id', 'child_of', self.ref('base.res_partner_category_0'))])
        self.assertTrue(partners)
        categ_root = Category.create({'name': 'Root category'})
        categ_0 = Category.create({'name': 'Parent category', 'parent_id': categ_root.id})
        categ_1 = Category.create({'name': 'Child1', 'parent_id': categ_0.id})
        cats = Category.search([('id', 'child_of', categ_root.ids)])
        self.assertEqual(len(cats), 3)
        cats = Category.search([('id', 'child_of', categ_root.id)])
        self.assertEqual(len(cats), 3)
        cats = Category.search([('id', 'child_of', (categ_0 + categ_1).ids)])
        self.assertEqual(len(cats), 2)
        cats = Category.search([('id', 'child_of', categ_0.ids)])
        self.assertEqual(len(cats), 2)
        cats = Category.search([('id', 'child_of', categ_1.ids)])
        self.assertEqual(len(cats), 1)
        cats = Category.search([('id', 'parent_of', categ_1.ids)])
        self.assertEqual(len(cats), 3)
        cats = Category.search([('id', 'parent_of', categ_1.id)])
        self.assertEqual(len(cats), 3)
        cats = Category.search([('id', 'parent_of', (categ_root + categ_0).ids)])
        self.assertEqual(len(cats), 2)
        cats = Category.search([('id', 'parent_of', categ_0.ids)])
        self.assertEqual(len(cats), 2)
        cats = Category.search([('id', 'parent_of', categ_root.ids)])
        self.assertEqual(len(cats), 1)

    def test_10_equivalent_id(self):
        if False:
            while True:
                i = 10
        Currency = self.env['res.currency']
        non_currency_id = max(Currency.search([]).ids) + 1003
        res_0 = Currency.search([])
        res_1 = Currency.search([('name', 'not like', 'probably_unexisting_name')])
        self.assertEqual(res_0, res_1)
        res_2 = Currency.search([('id', 'not in', [non_currency_id])])
        self.assertEqual(res_0, res_2)
        res_3 = Currency.search([('id', 'not in', [])])
        self.assertEqual(res_0, res_3)
        res_4 = Currency.search([('id', '!=', False)])
        self.assertEqual(res_0, res_4)
        Partner = self.env['res.partner']
        all_partners = Partner.search([])
        self.assertTrue(len(all_partners) > 1)
        one = all_partners[0]
        others = all_partners[1:]
        res_1 = Partner.search([('id', '=', one.id)])
        self.assertEqual(one, res_1)
        res_2 = Partner.search([('id', 'not in', others.ids)])
        self.assertEqual(one, res_2)
        res_3 = Partner.search(['!', ('id', '!=', one.id)])
        self.assertEqual(one, res_3)
        res_4 = Partner.search(['!', ('id', 'in', others.ids)])
        self.assertEqual(one, res_4)
        res_6 = Partner.search([('id', 'in', [one.id])])
        self.assertEqual(one, res_6)
        res_7 = Partner.search([('name', '=', one.name)])
        self.assertEqual(one, res_7)
        res_8 = Partner.search([('name', 'in', [one.name])])

    def test_15_m2o(self):
        if False:
            i = 10
            return i + 15
        Partner = self.env['res.partner']
        partners = Partner.search([('parent_id', '=', 'Agrolait')])
        self.assertTrue(partners)
        partners = Partner.search([('parent_id', 'in', 'Agrolait')])
        self.assertTrue(partners)
        partners = Partner.search([('parent_id', 'in', ['Agrolait', 'ASUStek'])])
        self.assertTrue(partners)
        partners = Partner.search([('company_id', 'in', [])])
        self.assertFalse(partners)
        company2 = self.env['res.company'].create({'name': 'Acme 2'})
        for i in xrange(4):
            Partner.create({'name': 'P of Acme %s' % i, 'company_id': company2.id})
        for i in xrange(4):
            Partner.create({'name': 'P of All %s' % i, 'company_id': False})
        all_partners = Partner.search([])
        res_partners = Partner.search(['|', ('company_id', 'not in', []), ('company_id', '=', False)])
        self.assertEqual(all_partners, res_partners, 'not in [] fails')
        partners = Partner.search([('company_id', 'in', [False])])
        self.assertTrue(len(partners) >= 4, 'We should have at least 4 partners with no company')
        partners = Partner.search([('company_id', 'not in', [1])])
        self.assertTrue(len(partners) >= 4, 'We should have at least 4 partners not related to company #1')
        partners = Partner.search(['|', ('company_id', 'not in', [1]), ('company_id', '=', False)])
        self.assertTrue(len(partners) >= 8, 'We should have at least 8 partners not related to company #1')
        partners = Partner.search([('company_id.partner_id', 'in', [])])
        self.assertFalse(partners)
        partners = Partner.search([('create_uid.active', '=', True)])
        all_partners = Partner.search([('company_id', '!=', False)])
        res_partners = Partner.search([('company_id.partner_id', 'not in', [])])
        self.assertEqual(all_partners, res_partners, 'not in [] fails')
        all_partners = Partner.search([])
        non_partner_id = max(all_partners.ids) + 1
        with_parent = all_partners.filtered(lambda p: p.parent_id)
        without_parent = all_partners.filtered(lambda p: not p.parent_id)
        with_website = all_partners.filtered(lambda p: p.website)
        res_0 = Partner.search([('parent_id', 'not like', 'probably_unexisting_name')])
        self.assertEqual(res_0, all_partners)
        res_1 = Partner.search([('parent_id', 'not in', [non_partner_id])])
        self.assertEqual(res_1, all_partners)
        res_2 = Partner.search([('parent_id', '!=', False)])
        self.assertEqual(res_2, with_parent)
        res_3 = Partner.search([('parent_id', 'not in', [])])
        self.assertEqual(res_3, all_partners)
        res_4 = Partner.search([('parent_id', 'not in', [False])])
        self.assertEqual(res_4, with_parent)
        res_4b = Partner.search([('parent_id', 'not ilike', '')])
        self.assertEqual(res_4b, without_parent)
        res_5 = Partner.search([('parent_id', 'like', 'probably_unexisting_name')])
        self.assertFalse(res_5)
        res_6 = Partner.search([('parent_id', 'in', [non_partner_id])])
        self.assertFalse(res_6)
        res_7 = Partner.search([('parent_id', '=', False)])
        self.assertEqual(res_7, without_parent)
        res_8 = Partner.search([('parent_id', 'in', [])])
        self.assertFalse(res_8)
        res_9 = Partner.search([('parent_id', 'in', [False])])
        self.assertEqual(res_9, without_parent)
        res_9b = Partner.search([('parent_id', 'ilike', '')])
        self.assertEqual(res_9b, with_parent)
        res_10 = Partner.search(['!', ('parent_id', 'like', 'probably_unexisting_name')])
        self.assertEqual(res_0, res_10)
        res_11 = Partner.search(['!', ('parent_id', 'in', [non_partner_id])])
        self.assertEqual(res_1, res_11)
        res_12 = Partner.search(['!', ('parent_id', '=', False)])
        self.assertEqual(res_2, res_12)
        res_13 = Partner.search(['!', ('parent_id', 'in', [])])
        self.assertEqual(res_3, res_13)
        res_14 = Partner.search(['!', ('parent_id', 'in', [False])])
        self.assertEqual(res_4, res_14)
        res_15 = Partner.search([('website', 'in', [])])
        self.assertFalse(res_15)
        res_16 = Partner.search([('website', 'not in', [])])
        self.assertEqual(res_16, all_partners)
        res_17 = Partner.search([('website', '!=', False)])
        self.assertEqual(res_17, with_website)
        companies = self.env['res.company'].search([])
        res_101 = companies.search([('currency_id', 'not ilike', '')])
        self.assertFalse(res_101)
        res_102 = companies.search([('currency_id', 'ilike', '')])
        self.assertEqual(res_102, companies)

    def test_in_operator(self):
        if False:
            for i in range(10):
                print('nop')
        " check that we can use the 'in' operator for plain fields "
        menus = self.env['ir.ui.menu'].search([('sequence', 'in', [1, 2, 10, 20])])
        self.assertTrue(menus)

    def test_15_o2m(self):
        if False:
            print('Hello World!')
        Partner = self.env['res.partner']
        partners = Partner.search([('child_ids', 'in', [])])
        self.assertFalse(partners)
        partners = Partner.search([('child_ids', '=', False)])
        for partner in partners:
            self.assertFalse(partner.child_ids)
        categories = self.env['res.partner.category'].search([])
        parents = categories.search([('child_ids', '!=', False)])
        self.assertEqual(parents, categories.filtered(lambda c: c.child_ids))
        leafs = categories.search([('child_ids', '=', False)])
        self.assertEqual(leafs, categories.filtered(lambda c: not c.child_ids))
        partners = Partner.search([('category_id', 'in', [])])
        self.assertFalse(partners)
        partners = Partner.search([('category_id', '=', False)])
        for partner in partners:
            self.assertFalse(partner.category_id)
        partners = Partner.search([('child_ids.city', '=', 'foo')])
        self.assertFalse(partners)

    def test_15_equivalent_one2many_1(self):
        if False:
            i = 10
            return i + 15
        Company = self.env['res.company']
        company3 = Company.create({'name': 'Acme 3'})
        company4 = Company.create({'name': 'Acme 4', 'parent_id': company3.id})
        res_1 = Company.search([('child_ids', 'in', company3.child_ids.ids)])
        self.assertEqual(res_1, company3)
        res_2 = Company.search([('child_ids', 'in', company3.child_ids[0].ids)])
        self.assertEqual(res_2, company3)
        expected = company3 + company4
        res_1 = Company.search([('id', 'child_of', [company3.id])])
        self.assertEqual(res_1, expected)
        res_2 = Company.search([('id', 'child_of', company3.id)])
        self.assertEqual(res_2, expected)
        res_3 = Company.search([('id', 'child_of', [company3.name])])
        self.assertEqual(res_3, expected)
        res_4 = Company.search([('id', 'child_of', company3.name)])
        self.assertEqual(res_4, expected)
        expected = company3 + company4
        res_1 = Company.search([('id', 'parent_of', [company4.id])])
        self.assertEqual(res_1, expected)
        res_2 = Company.search([('id', 'parent_of', company4.id)])
        self.assertEqual(res_2, expected)
        res_3 = Company.search([('id', 'parent_of', [company4.name])])
        self.assertEqual(res_3, expected)
        res_4 = Company.search([('id', 'parent_of', company4.name)])
        self.assertEqual(res_4, expected)
        Partner = self.env['res.partner']
        Users = self.env['res.users']
        (p1, _) = Partner.name_create('Dédé Boitaclou')
        (p2, _) = Partner.name_create("Raoulette Pizza O'poil")
        u1a = Users.create({'login': 'dbo', 'partner_id': p1}).id
        u1b = Users.create({'login': 'dbo2', 'partner_id': p1}).id
        u2 = Users.create({'login': 'rpo', 'partner_id': p2}).id
        self.assertEqual([p1], Partner.search([('user_ids', 'in', u1a)]).ids, 'o2m IN accept single int on right side')
        self.assertEqual([p1], Partner.search([('user_ids', '=', 'Dédé Boitaclou')]).ids, 'o2m NOT IN matches none on the right side')
        self.assertEqual([], Partner.search([('user_ids', 'in', [10000])]).ids, 'o2m NOT IN matches none on the right side')
        self.assertEqual([p1, p2], Partner.search([('user_ids', 'in', [u1a, u2])]).ids, 'o2m IN matches any on the right side')
        all_ids = Partner.search([]).ids
        self.assertEqual(set(all_ids) - set([p1]), set(Partner.search([('user_ids', 'not in', u1a)]).ids), 'o2m NOT IN matches none on the right side')
        self.assertEqual(set(all_ids) - set([p1]), set(Partner.search([('user_ids', '!=', 'Dédé Boitaclou')]).ids), 'o2m NOT IN matches none on the right side')
        self.assertEqual(set(all_ids) - set([p1, p2]), set(Partner.search([('user_ids', 'not in', [u1b, u2])]).ids), 'o2m NOT IN matches none on the right side')

    def test_15_equivalent_one2many_2(self):
        if False:
            return 10
        Currency = self.env['res.currency']
        CurrencyRate = self.env['res.currency.rate']
        currency = Currency.create({'name': 'ZZZ', 'symbol': 'ZZZ', 'rounding': 1.0})
        currency_rate = CurrencyRate.create({'name': '2010-01-01', 'currency_id': currency.id, 'rate': 1.0})
        non_currency_id = currency_rate.id + 1000
        default_currency = Currency.browse(1)
        currency_rate1 = CurrencyRate.search([('name', 'not like', 'probably_unexisting_name')])
        currency_rate2 = CurrencyRate.search([('id', 'not in', [non_currency_id])])
        self.assertEqual(currency_rate1, currency_rate2)
        currency_rate3 = CurrencyRate.search([('id', 'not in', [])])
        self.assertEqual(currency_rate1, currency_rate3)
        res_3 = Currency.search([('rate_ids', 'in', default_currency.rate_ids.ids)])
        self.assertEqual(res_3, default_currency)
        res_4 = Currency.search([('rate_ids', 'in', default_currency.rate_ids[0].ids)])
        self.assertEqual(res_4, default_currency)
        res_5 = Currency.search([('rate_ids', 'in', default_currency.rate_ids[0].id)])
        self.assertEqual(res_5, default_currency)
        res_9 = Currency.search([('rate_ids', 'like', 'probably_unexisting_name')])
        self.assertFalse(res_9)
        res_10 = Currency.search([('rate_ids', 'not like', 'probably_unexisting_name')])
        res_11 = Currency.search([('rate_ids', 'not in', [non_currency_id])])
        self.assertEqual(res_10, res_11)
        res_12 = Currency.search([('rate_ids', '!=', False)])
        self.assertEqual(res_10, res_12)
        res_13 = Currency.search([('rate_ids', 'not in', [])])
        self.assertEqual(res_10, res_13)

    def test_20_expression_parse(self):
        if False:
            i = 10
            return i + 15
        Users = self.env['res.users']
        a = Users.create({'name': 'test_A', 'login': 'test_A'})
        b1 = Users.create({'name': 'test_B', 'login': 'test_B'})
        b2 = Users.create({'name': 'test_B2', 'login': 'test_B2', 'parent_id': b1.partner_id.id})
        users = Users.search([('name', 'like', 'test')])
        self.assertEqual(users, a + b1 + b2, 'searching through inheritance failed')
        users = Users.search([('name', '=', 'test_B')])
        self.assertEqual(users, b1, 'searching through inheritance failed')
        users = Users.search([('child_ids.name', 'like', 'test_B')])
        self.assertEqual(users, b1, 'searching through inheritance failed')
        users = Users.search([('name', 'like', 'test'), ('parent_id', '=?', False)])
        self.assertEqual(users, a + b1 + b2, '(x =? False) failed')
        users = Users.search([('name', 'like', 'test'), ('parent_id', '=?', b1.partner_id.id)])
        self.assertEqual(users, b2, '(x =? id) failed')

    def test_30_normalize_domain(self):
        if False:
            return 10
        norm_domain = domain = ['&', (1, '=', 1), ('a', '=', 'b')]
        self.assertEqual(norm_domain, expression.normalize_domain(domain), 'Normalized domains should be left untouched')
        domain = [('x', 'in', ['y', 'z']), ('a.v', '=', 'e'), '|', '|', ('a', '=', 'b'), '!', ('c', '>', 'd'), ('e', '!=', 'f'), ('g', '=', 'h')]
        norm_domain = ['&', '&', '&'] + domain
        self.assertEqual(norm_domain, expression.normalize_domain(domain), 'Non-normalized domains should be properly normalized')

    def test_40_negating_long_expression(self):
        if False:
            while True:
                i = 10
        source = ['!', '&', ('user_id', '=', 4), ('partner_id', 'in', [1, 2])]
        expect = ['|', ('user_id', '!=', 4), ('partner_id', 'not in', [1, 2])]
        self.assertEqual(expression.distribute_not(source), expect, 'distribute_not on expression applied wrongly')
        pos_leaves = [[('a', 'in', [])], [('d', '!=', 3)]]
        neg_leaves = [[('a', 'not in', [])], [('d', '=', 3)]]
        source = expression.OR([expression.AND(pos_leaves)] * 1000)
        expect = source
        self.assertEqual(expression.distribute_not(source), expect, 'distribute_not on long expression without negation operator should not alter it')
        source = ['!'] + source
        expect = expression.AND([expression.OR(neg_leaves)] * 1000)
        self.assertEqual(expression.distribute_not(source), expect, 'distribute_not on long expression applied wrongly')

    def test_accent(self):
        if False:
            print('Hello World!')
        if not self.registry.has_unaccent:
            return
        Company = self.env['res.company']
        helene = Company.create({'name': u'Hélène'})
        self.assertEqual(helene, Company.search([('name', 'ilike', 'Helene')]))
        self.assertEqual(helene, Company.search([('name', 'ilike', 'hélène')]))
        self.assertNotIn(helene, Company.search([('name', 'not ilike', 'Helene')]))
        self.assertNotIn(helene, Company.search([('name', 'not ilike', 'hélène')]))

    def test_like_wildcards(self):
        if False:
            for i in range(10):
                print('nop')
        Partner = self.env['res.partner']
        partners = Partner.search([('name', '=like', 'A_U_TeK')])
        self.assertTrue(len(partners) == 1, 'Must match one partner (ASUSTeK)')
        partners = Partner.search([('name', '=ilike', 'c%')])
        self.assertTrue(len(partners) >= 1, 'Must match one partner (China Export)')
        Country = self.env['res.country']
        countries = Country.search([('name', '=like', 'Ind__')])
        self.assertTrue(len(countries) == 1, 'Must match India only')
        countries = Country.search([('name', '=ilike', 'z%')])
        self.assertTrue(len(countries) == 3, 'Must match only countries with names starting with Z (currently 3)')

    def test_translate_search(self):
        if False:
            print('Hello World!')
        Country = self.env['res.country']
        belgium = self.env.ref('base.be')
        domains = [[('name', '=', 'Belgium')], [('name', 'ilike', 'Belgi')], [('name', 'in', ['Belgium', 'Care Bears'])]]
        for domain in domains:
            countries = Country.search(domain)
            self.assertEqual(countries, belgium)

    def test_long_table_alias(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch_order('res.users', 'partner_id')
        self.patch_order('res.partner', 'commercial_partner_id,company_id,name')
        self.patch_order('res.company', 'parent_id')
        self.env['res.users'].search([('name', '=', 'test')])

    @mute_logger('odoo.sql_db')
    def test_invalid(self):
        if False:
            print('Hello World!')
        ' verify that invalid expressions are refused, even for magic fields '
        Country = self.env['res.country']
        with self.assertRaises(ValueError):
            Country.search([('does_not_exist', '=', 'foo')])
        with self.assertRaises(ValueError):
            Country.search([('create_date', '>>', 'foo')])
        with self.assertRaises(psycopg2.DataError):
            Country.search([('create_date', '=', "1970-01-01'); --")])

    def test_active(self):
        if False:
            for i in range(10):
                print('nop')
        Partner = self.env['res.partner']
        vals = {'name': 'OpenERP Test', 'active': False, 'category_id': [(6, 0, [self.ref('base.res_partner_category_1')])], 'child_ids': [(0, 0, {'name': 'address of OpenERP Test', 'country_id': self.ref('base.be')})]}
        Partner.create(vals)
        partner = Partner.search([('category_id', 'ilike', 'vendor'), ('active', '=', False)])
        self.assertTrue(partner, 'Record not Found with category vendor and active False.')
        partner = Partner.search([('child_ids.country_id', '=', 'Belgium'), ('active', '=', False)])
        self.assertTrue(partner, 'Record not Found with country Belgium and active False.')

    def test_lp1071710(self):
        if False:
            for i in range(10):
                print('nop')
        ' Check that we can exclude translated fields (bug lp:1071710) '
        self.env['ir.translation'].load_module_terms(['base'], ['fr_FR'])
        Country = self.env['res.country']
        be = self.env.ref('base.be')
        not_be = Country.with_context(lang='fr_FR').search([('name', '!=', 'Belgique')])
        self.assertNotIn(be, not_be)
        Partner = self.env['res.partner']
        agrolait = Partner.search([('name', '=', 'Agrolait')])
        not_be = Partner.search([('country_id', '!=', 'Belgium')])
        self.assertNotIn(agrolait, not_be)
        not_be = Partner.with_context(lang='fr_FR').search([('country_id', '!=', 'Belgique')])
        self.assertNotIn(agrolait, not_be)

class TestAutoJoin(TransactionCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestAutoJoin, self).setUp()
        self._reinit_mock()
        BaseModel_where_calc = BaseModel._where_calc

        def _where_calc(model, *args, **kwargs):
            if False:
                print('Hello World!')
            ' Mock `_where_calc` to be able to test its results. Store them\n                into some internal variable for latter processing. '
            query = BaseModel_where_calc(model, *args, **kwargs)
            self.query_list.append(query)
            return query
        self.patch(BaseModel, '_where_calc', _where_calc)

    def _reinit_mock(self):
        if False:
            i = 10
            return i + 15
        self.query_list = []

    def test_auto_join(self):
        if False:
            while True:
                i = 10
        unaccent = expression.get_unaccent_wrapper(self.cr)
        partner_obj = self.env['res.partner']
        state_obj = self.env['res.country.state']
        bank_obj = self.env['res.partner.bank']

        def patch_auto_join(model, fname, value):
            if False:
                for i in range(10):
                    print('nop')
            self.patch(model._fields[fname], 'auto_join', value)

        def patch_domain(model, fname, value):
            if False:
                return 10
            self.patch(model._fields[fname], 'domain', value)
        country_us = self.env['res.country'].search([('code', 'like', 'US')], limit=1)
        states = self.env['res.country.state'].search([('country_id', '=', country_us.id)], limit=2)
        p_a = partner_obj.create({'name': 'test__A', 'state_id': states[0].id})
        p_b = partner_obj.create({'name': 'test__B', 'state_id': states[1].id})
        p_aa = partner_obj.create({'name': 'test__AA', 'parent_id': p_a.id, 'state_id': states[0].id})
        p_ab = partner_obj.create({'name': 'test__AB', 'parent_id': p_a.id, 'state_id': states[1].id})
        p_ba = partner_obj.create({'name': 'test__BA', 'parent_id': p_b.id, 'state_id': states[0].id})
        b_aa = bank_obj.create({'acc_number': '123', 'acc_type': 'bank', 'partner_id': p_aa.id})
        b_ab = bank_obj.create({'acc_number': '456', 'acc_type': 'bank', 'partner_id': p_ab.id})
        b_ba = bank_obj.create({'acc_number': '789', 'acc_type': 'bank', 'partner_id': p_ba.id})
        patch_auto_join(partner_obj, 'category_id', True)
        with self.assertRaises(NotImplementedError):
            partner_obj.search([('category_id.name', '=', 'foo')])
        name_test = '12'
        self._reinit_mock()
        partners = partner_obj.search([('bank_ids.sanitized_acc_number', 'like', name_test)])
        self.assertEqual(partners, p_aa, "_auto_join off: ('bank_ids.sanitized_acc_number', 'like', '..'): incorrect result")
        self.assertEqual(len(self.query_list), 2, "_auto_join off: ('bank_ids.sanitized_acc_number', 'like', '..') should produce 2 queries (1 in res_partner_bank, 1 on res_partner)")
        sql_query = self.query_list[0].get_sql()
        self.assertIn('res_partner_bank', sql_query[0], "_auto_join off: ('bank_ids.sanitized_acc_number', 'like', '..') first query incorrect main table")
        expected = '%s::text like %s' % (unaccent('"res_partner_bank"."sanitized_acc_number"'), unaccent('%s'))
        self.assertIn(expected, sql_query[1], "_auto_join off: ('bank_ids.sanitized_acc_number', 'like', '..') first query incorrect where condition")
        self.assertEqual(['%' + name_test + '%'], sql_query[2], "_auto_join off: ('bank_ids.sanitized_acc_number', 'like', '..') first query incorrect parameter")
        sql_query = self.query_list[1].get_sql()
        self.assertIn('res_partner', sql_query[0], "_auto_join off: ('bank_ids.sanitized_acc_number', 'like', '..') second query incorrect main table")
        self.assertIn('"res_partner"."id" in (%s)', sql_query[1], "_auto_join off: ('bank_ids.sanitized_acc_number', 'like', '..') second query incorrect where condition")
        self.assertIn(p_aa.id, sql_query[2], "_auto_join off: ('bank_ids.sanitized_acc_number', 'like', '..') second query incorrect parameter")
        self._reinit_mock()
        partners = partner_obj.search([('child_ids.bank_ids.id', 'in', [b_aa.id, b_ba.id])])
        self.assertEqual(partners, p_a + p_b, "_auto_join off: ('child_ids.bank_ids.id', 'in', [..]): incorrect result")
        self.assertEqual(len(self.query_list), 3, "_auto_join off: ('child_ids.bank_ids.id', 'in', [..]) should produce 3 queries (1 in res_partner_bank, 2 on res_partner)")
        patch_auto_join(partner_obj, 'bank_ids', True)
        self._reinit_mock()
        partners = partner_obj.search([('bank_ids.sanitized_acc_number', 'like', name_test)])
        self.assertEqual(partners, p_aa, "_auto_join on: ('bank_ids.sanitized_acc_number', 'like', '..') incorrect result")
        self.assertEqual(len(self.query_list), 1, "_auto_join on: ('bank_ids.sanitized_acc_number', 'like', '..') should produce 1 query")
        sql_query = self.query_list[0].get_sql()
        self.assertIn('"res_partner"', sql_query[0], "_auto_join on: ('bank_ids.sanitized_acc_number', 'like', '..') query incorrect main table")
        self.assertIn('"res_partner_bank" as "res_partner__bank_ids"', sql_query[0], "_auto_join on: ('bank_ids.sanitized_acc_number', 'like', '..') query incorrect join")
        expected = '%s::text like %s' % (unaccent('"res_partner__bank_ids"."sanitized_acc_number"'), unaccent('%s'))
        self.assertIn(expected, sql_query[1], "_auto_join on: ('bank_ids.sanitized_acc_number', 'like', '..') query incorrect where condition")
        self.assertIn('"res_partner"."id"="res_partner__bank_ids"."partner_id"', sql_query[1], "_auto_join on: ('bank_ids.sanitized_acc_number', 'like', '..') query incorrect join condition")
        self.assertIn('%' + name_test + '%', sql_query[2], "_auto_join on: ('bank_ids.sanitized_acc_number', 'like', '..') query incorrect parameter")
        self._reinit_mock()
        bank_ids = [b_aa.id, b_ab.id]
        partners = partner_obj.search([('bank_ids.id', 'in', bank_ids)])
        self.assertEqual(partners, p_aa + p_ab, "_auto_join on: ('bank_ids.id', 'in', [..]) incorrect result")
        self.assertEqual(len(self.query_list), 1, "_auto_join on: ('bank_ids.id', 'in', [..]) should produce 1 query")
        sql_query = self.query_list[0].get_sql()
        self.assertIn('"res_partner"', sql_query[0], "_auto_join on: ('bank_ids.id', 'in', [..]) query incorrect main table")
        self.assertIn('"res_partner__bank_ids"."id" in (%s,%s)', sql_query[1], "_auto_join on: ('bank_ids.id', 'in', [..]) query incorrect where condition")
        self.assertLessEqual(set(bank_ids), set(sql_query[2]), "_auto_join on: ('bank_ids.id', 'in', [..]) query incorrect parameter")
        patch_auto_join(partner_obj, 'child_ids', True)
        self._reinit_mock()
        bank_ids = [b_aa.id, b_ba.id]
        partners = partner_obj.search([('child_ids.bank_ids.id', 'in', bank_ids)])
        self.assertEqual(partners, p_a + p_b, "_auto_join on: ('child_ids.bank_ids.id', 'not in', [..]): incorrect result")
        self.assertEqual(len(self.query_list), 1, "_auto_join on: ('child_ids.bank_ids.id', 'in', [..]) should produce 1 query")
        sql_query = self.query_list[0].get_sql()
        self.assertIn('"res_partner"', sql_query[0], "_auto_join on: ('child_ids.bank_ids.id', 'in', [..]) incorrect main table")
        self.assertIn('"res_partner" as "res_partner__child_ids"', sql_query[0], "_auto_join on: ('child_ids.bank_ids.id', 'in', [..]) query incorrect join")
        self.assertIn('"res_partner_bank" as "res_partner__child_ids__bank_ids"', sql_query[0], "_auto_join on: ('child_ids.bank_ids.id', 'in', [..]) query incorrect join")
        self.assertIn('"res_partner__child_ids__bank_ids"."id" in (%s,%s)', sql_query[1], "_auto_join on: ('child_ids.bank_ids.id', 'in', [..]) query incorrect where condition")
        self.assertIn('"res_partner"."id"="res_partner__child_ids"."parent_id"', sql_query[1], "_auto_join on: ('child_ids.bank_ids.id', 'in', [..]) query incorrect join condition")
        self.assertIn('"res_partner__child_ids"."id"="res_partner__child_ids__bank_ids"."partner_id"', sql_query[1], "_auto_join on: ('child_ids.bank_ids.id', 'in', [..]) query incorrect join condition")
        self.assertLessEqual(set(bank_ids), set(sql_query[2][-2:]), "_auto_join on: ('child_ids.bank_ids.id', 'in', [..]) query incorrect parameter")
        name_test = 'US'
        self._reinit_mock()
        partners = partner_obj.search([('state_id.country_id.code', 'like', name_test)])
        self.assertLessEqual(p_a + p_b + p_aa + p_ab + p_ba, partners, "_auto_join off: ('state_id.country_id.code', 'like', '..') incorrect result")
        self.assertEqual(len(self.query_list), 3, "_auto_join off: ('state_id.country_id.code', 'like', '..') should produce 3 queries (1 on res_country, 1 on res_country_state, 1 on res_partner)")
        patch_auto_join(partner_obj, 'state_id', True)
        self._reinit_mock()
        partners = partner_obj.search([('state_id.country_id.code', 'like', name_test)])
        self.assertLessEqual(p_a + p_b + p_aa + p_ab + p_ba, partners, "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') incorrect result")
        self.assertEqual(len(self.query_list), 2, "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') should produce 2 query")
        sql_query = self.query_list[0].get_sql()
        self.assertIn('"res_country"', sql_query[0], "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') query 1 incorrect main table")
        expected = '%s::text like %s' % (unaccent('"res_country"."code"'), unaccent('%s'))
        self.assertIn(expected, sql_query[1], "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') query 1 incorrect where condition")
        self.assertEqual(['%' + name_test + '%'], sql_query[2], "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') query 1 incorrect parameter")
        sql_query = self.query_list[1].get_sql()
        self.assertIn('"res_partner"', sql_query[0], "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') query 2 incorrect main table")
        self.assertIn('"res_country_state" as "res_partner__state_id"', sql_query[0], "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') query 2 incorrect join")
        self.assertIn('"res_partner__state_id"."country_id" in (%s)', sql_query[1], "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') query 2 incorrect where condition")
        self.assertIn('"res_partner"."state_id"="res_partner__state_id"."id"', sql_query[1], "_auto_join on for state_id: ('state_id.country_id.code', 'like', '..') query 2 incorrect join condition")
        patch_auto_join(partner_obj, 'state_id', False)
        patch_auto_join(state_obj, 'country_id', True)
        self._reinit_mock()
        partners = partner_obj.search([('state_id.country_id.code', 'like', name_test)])
        self.assertLessEqual(p_a + p_b + p_aa + p_ab + p_ba, partners, "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') incorrect result")
        self.assertEqual(len(self.query_list), 2, "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') should produce 2 query")
        sql_query = self.query_list[0].get_sql()
        self.assertIn('"res_country_state"', sql_query[0], "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') query 1 incorrect main table")
        self.assertIn('"res_country" as "res_country_state__country_id"', sql_query[0], "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') query 1 incorrect join")
        expected = '%s::text like %s' % (unaccent('"res_country_state__country_id"."code"'), unaccent('%s'))
        self.assertIn(expected, sql_query[1], "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') query 1 incorrect where condition")
        self.assertIn('"res_country_state"."country_id"="res_country_state__country_id"."id"', sql_query[1], "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') query 1 incorrect join condition")
        self.assertEqual(['%' + name_test + '%'], sql_query[2], "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') query 1 incorrect parameter")
        sql_query = self.query_list[1].get_sql()
        self.assertIn('"res_partner"', sql_query[0], "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') query 2 incorrect main table")
        self.assertIn('"res_partner"."state_id" in', sql_query[1], "_auto_join on for country_id: ('state_id.country_id.code', 'like', '..') query 2 incorrect where condition")
        patch_auto_join(partner_obj, 'state_id', True)
        patch_auto_join(state_obj, 'country_id', True)
        self._reinit_mock()
        partners = partner_obj.search([('state_id.country_id.code', 'like', name_test)])
        self.assertLessEqual(p_a + p_b + p_aa + p_ab + p_ba, partners, "_auto_join on: ('state_id.country_id.code', 'like', '..') incorrect result")
        self.assertEqual(len(self.query_list), 1, "_auto_join on: ('state_id.country_id.code', 'like', '..') should produce 1 query")
        sql_query = self.query_list[0].get_sql()
        self.assertIn('"res_partner"', sql_query[0], "_auto_join on: ('state_id.country_id.code', 'like', '..') query incorrect main table")
        self.assertIn('"res_country_state" as "res_partner__state_id"', sql_query[0], "_auto_join on: ('state_id.country_id.code', 'like', '..') query incorrect join")
        self.assertIn('"res_country" as "res_partner__state_id__country_id"', sql_query[0], "_auto_join on: ('state_id.country_id.code', 'like', '..') query incorrect join")
        expected = '%s::text like %s' % (unaccent('"res_partner__state_id__country_id"."code"'), unaccent('%s'))
        self.assertIn(expected, sql_query[1], "_auto_join on: ('state_id.country_id.code', 'like', '..') query incorrect where condition")
        self.assertIn('"res_partner"."state_id"="res_partner__state_id"."id"', sql_query[1], "_auto_join on: ('state_id.country_id.code', 'like', '..') query incorrect join condition")
        self.assertIn('"res_partner__state_id"."country_id"="res_partner__state_id__country_id"."id"', sql_query[1], "_auto_join on: ('state_id.country_id.code', 'like', '..') query incorrect join condition")
        self.assertIn('%' + name_test + '%', sql_query[2], "_auto_join on: ('state_id.country_id.code', 'like', '..') query incorrect parameter")
        patch_auto_join(partner_obj, 'child_ids', True)
        patch_auto_join(partner_obj, 'bank_ids', True)
        patch_domain(partner_obj, 'child_ids', lambda self: ['!', ('name', '=', self._name)])
        patch_domain(partner_obj, 'bank_ids', [('sanitized_acc_number', 'like', '2')])
        self._reinit_mock()
        partners = partner_obj.search(['&', (1, '=', 1), ('child_ids.bank_ids.id', 'in', [b_aa.id, b_ba.id])])
        self.assertLessEqual(p_a, partners, '_auto_join on one2many with domains incorrect result')
        self.assertFalse(p_ab + p_ba & partners, '_auto_join on one2many with domains incorrect result')
        sql_query = self.query_list[0].get_sql()
        expected = '%s::text like %s' % (unaccent('"res_partner__child_ids__bank_ids"."sanitized_acc_number"'), unaccent('%s'))
        self.assertIn(expected, sql_query[1], '_auto_join on one2many with domains incorrect result')
        self.assertIn('"res_partner__child_ids"."name" = %s', sql_query[1], '_auto_join on one2many with domains incorrect result')
        patch_domain(partner_obj, 'child_ids', lambda self: [('name', '=', '__%s' % self._name)])
        self._reinit_mock()
        partners = partner_obj.search(['&', (1, '=', 1), ('child_ids.bank_ids.id', 'in', [b_aa.id, b_ba.id])])
        self.assertFalse(partners, '_auto_join on one2many with domains incorrect result')
        patch_auto_join(partner_obj, 'bank_ids', False)
        patch_auto_join(partner_obj, 'child_ids', False)
        patch_auto_join(partner_obj, 'state_id', False)
        patch_auto_join(partner_obj, 'parent_id', False)
        patch_auto_join(state_obj, 'country_id', False)
        patch_domain(partner_obj, 'child_ids', [])
        patch_domain(partner_obj, 'bank_ids', [])
        self._reinit_mock()
        partners = partner_obj.search([('child_ids.state_id.country_id.code', 'like', name_test)])
        self.assertLessEqual(p_a + p_b, partners, "_auto_join off: ('child_ids.state_id.country_id.code', 'like', '..') incorrect result")
        self.assertEqual(len(self.query_list), 4, "_auto_join off: ('child_ids.state_id.country_id.code', 'like', '..') number of queries incorrect")
        patch_auto_join(partner_obj, 'child_ids', True)
        patch_auto_join(partner_obj, 'state_id', True)
        patch_auto_join(state_obj, 'country_id', True)
        self._reinit_mock()
        partners = partner_obj.search([('child_ids.state_id.country_id.code', 'like', name_test)])
        self.assertLessEqual(p_a + p_b, partners, "_auto_join on: ('child_ids.state_id.country_id.code', 'like', '..') incorrect result")
        self.assertEqual(len(self.query_list), 1, "_auto_join on: ('child_ids.state_id.country_id.code', 'like', '..') number of queries incorrect")
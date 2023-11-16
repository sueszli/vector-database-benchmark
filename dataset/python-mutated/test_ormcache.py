from odoo.tests.common import TransactionCase
from odoo.tools import get_cache_key_counter

class TestOrmcache(TransactionCase):

    def test_ormcache(self):
        if False:
            print('Hello World!')
        ' Test the effectiveness of the ormcache() decorator. '
        IMD = self.env['ir.model.data']
        XMLID = 'base.group_no_one'
        (cache, key, counter) = get_cache_key_counter(IMD.xmlid_lookup, XMLID)
        hit = counter.hit
        miss = counter.miss
        IMD.xmlid_lookup.clear_cache(IMD)
        self.assertNotIn(key, cache)
        self.env.ref(XMLID)
        self.assertEqual(counter.hit, hit)
        self.assertEqual(counter.miss, miss + 1)
        self.assertIn(key, cache)
        self.env.ref(XMLID)
        self.assertEqual(counter.hit, hit + 1)
        self.assertEqual(counter.miss, miss + 1)
        self.assertIn(key, cache)
        self.env.ref(XMLID)
        self.assertEqual(counter.hit, hit + 2)
        self.assertEqual(counter.miss, miss + 1)
        self.assertIn(key, cache)
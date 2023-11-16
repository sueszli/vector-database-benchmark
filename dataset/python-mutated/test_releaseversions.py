from test.picardtestcase import PicardTestCase, load_test_json
from picard.releasegroup import ReleaseGroup
settings = {'standardize_tracks': False, 'standardize_artists': False, 'standardize_releases': False, 'translate_artist_names': False}

class ReleaseTest(PicardTestCase):

    def test_1(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_config_values(settings)
        rlist = load_test_json('release_group_2.json')
        r = ReleaseGroup(1)
        r._parse_versions(rlist)
        self.assertEqual(r.versions[0]['name'], '5 / 2009 / GB / CD / label A / cat 123 / Jewel Case / special')
        self.assertEqual(r.versions[1]['name'], '5 / 2009 / GB / CD / label A / cat 123 / Digipak / special')
        self.assertEqual(r.versions[2]['name'], '5 / 2009 / GB / CD / label A / cat 123 / Digipak / specialx')

    def test_2(self):
        if False:
            i = 10
            return i + 15
        self.set_config_values(settings)
        rlist = load_test_json('release_group_3.json')
        r = ReleaseGroup(1)
        r._parse_versions(rlist)
        self.assertEqual(r.versions[0]['name'], '5 / 2011 / FR / CD / label A / cat 123 / special A')
        self.assertEqual(r.versions[1]['name'], '5 / 2011 / FR / CD / label A / cat 123')

    def test_3(self):
        if False:
            return 10
        self.set_config_values(settings)
        rlist = load_test_json('release_group_4.json')
        r = ReleaseGroup(1)
        r._parse_versions(rlist)
        self.assertEqual(r.versions[0]['name'], '5 / 2009 / FR / CD / label A / cat 123 / 0123456789')
        self.assertEqual(r.versions[1]['name'], '5 / 2009 / FR / CD / label A / cat 123 / [no barcode]')
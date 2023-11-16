import json
import io
import pytest
from Crypt import CryptBitcoin
from Content.ContentManager import VerifyError, SignError

@pytest.mark.usefixtures('resetSettings')
class TestContentUser:

    def testSigners(self, site):
        if False:
            return 10
        file_info = site.content_manager.getFileInfo('data/users/notexist/data.json')
        assert file_info['content_inner_path'] == 'data/users/notexist/content.json'
        file_info = site.content_manager.getFileInfo('data/users/notexist/a/b/data.json')
        assert file_info['content_inner_path'] == 'data/users/notexist/content.json'
        valid_signers = site.content_manager.getValidSigners('data/users/notexist/content.json')
        assert valid_signers == ['14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet', 'notexist', '1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT']
        valid_signers = site.content_manager.getValidSigners('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json')
        assert '1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT' in valid_signers
        assert '14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet' in valid_signers
        assert '1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C' in valid_signers
        assert len(valid_signers) == 3
        user_content = site.storage.loadJson('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json')
        user_content['cert_user_id'] = 'bad@zeroid.bit'
        valid_signers = site.content_manager.getValidSigners('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_content)
        assert '1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT' in valid_signers
        assert '14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet' in valid_signers
        assert '1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C' not in valid_signers

    def testRules(self, site):
        if False:
            return 10
        user_content = site.storage.loadJson('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json')
        user_content['cert_auth_type'] = 'web'
        user_content['cert_user_id'] = 'nofish@zeroid.bit'
        rules = site.content_manager.getRules('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_content)
        assert rules['max_size'] == 100000
        assert '1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C' in rules['signers']
        user_content['cert_auth_type'] = 'web'
        user_content['cert_user_id'] = 'noone@zeroid.bit'
        rules = site.content_manager.getRules('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_content)
        assert rules['max_size'] == 10000
        assert '1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C' in rules['signers']
        user_content['cert_auth_type'] = 'bitmsg'
        user_content['cert_user_id'] = 'noone@zeroid.bit'
        rules = site.content_manager.getRules('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_content)
        assert rules['max_size'] == 15000
        assert '1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C' in rules['signers']
        user_content['cert_auth_type'] = 'web'
        user_content['cert_user_id'] = 'bad@zeroid.bit'
        rules = site.content_manager.getRules('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_content)
        assert '1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C' not in rules['signers']

    def testRulesAddress(self, site):
        if False:
            print('Hello World!')
        user_inner_path = 'data/users/1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9/content.json'
        user_content = site.storage.loadJson(user_inner_path)
        rules = site.content_manager.getRules(user_inner_path, user_content)
        assert rules['max_size'] == 10000
        assert '1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9' in rules['signers']
        users_content = site.content_manager.contents['data/users/content.json']
        users_content['user_contents']['permissions']['1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9'] = False
        rules = site.content_manager.getRules(user_inner_path, user_content)
        assert '1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9' not in rules['signers']
        users_content['user_contents']['permissions']['1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9'] = {'max_size': 20000}
        rules = site.content_manager.getRules(user_inner_path, user_content)
        assert rules['max_size'] == 20000

    def testVerifyAddress(self, site):
        if False:
            i = 10
            return i + 15
        privatekey = '5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv'
        user_inner_path = 'data/users/1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9/content.json'
        data_dict = site.storage.loadJson(user_inner_path)
        users_content = site.content_manager.contents['data/users/content.json']
        data = io.BytesIO(json.dumps(data_dict).encode())
        assert site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        data_dict['files']['data.json']['size'] = 1024 * 15
        del data_dict['signs']
        data_dict['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), privatekey)}
        data = io.BytesIO(json.dumps(data_dict).encode())
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        assert 'Include too large' in str(err.value)
        users_content['user_contents']['permissions']['1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9'] = {'max_size': 20000}
        del data_dict['signs']
        data_dict['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), privatekey)}
        data = io.BytesIO(json.dumps(data_dict).encode())
        assert site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)

    def testVerify(self, site):
        if False:
            for i in range(10):
                print('nop')
        privatekey = '5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv'
        user_inner_path = 'data/users/1CjfbrbwtP8Y2QjPy12vpTATkUT7oSiPQ9/content.json'
        data_dict = site.storage.loadJson(user_inner_path)
        users_content = site.content_manager.contents['data/users/content.json']
        data = io.BytesIO(json.dumps(data_dict).encode())
        assert site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        rules = site.content_manager.getRules(user_inner_path, data_dict)
        assert rules['max_size'] == 10000
        assert users_content['user_contents']['permission_rules']['.*']['max_size'] == 10000
        users_content['user_contents']['permission_rules']['.*']['max_size'] = 0
        rules = site.content_manager.getRules(user_inner_path, data_dict)
        assert rules['max_size'] == 0
        data = io.BytesIO(json.dumps(data_dict).encode())
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        assert 'Include too large' in str(err.value)
        users_content['user_contents']['permission_rules']['.*']['max_size'] = 10000
        data_dict['files_optional']['peanut-butter-jelly-time.gif']['size'] = 1024 * 1024
        del data_dict['signs']
        data_dict['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), privatekey)}
        data = io.BytesIO(json.dumps(data_dict).encode())
        assert site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        data_dict['files_optional']['peanut-butter-jelly-time.gif']['size'] = 100 * 1024 * 1024
        del data_dict['signs']
        data_dict['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), privatekey)}
        data = io.BytesIO(json.dumps(data_dict).encode())
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        assert 'Include optional files too large' in str(err.value)
        data_dict['files_optional']['peanut-butter-jelly-time.gif']['size'] = 1024 * 1024
        data_dict['files_optional']['hello.exe'] = data_dict['files_optional']['peanut-butter-jelly-time.gif']
        del data_dict['signs']
        data_dict['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), privatekey)}
        data = io.BytesIO(json.dumps(data_dict).encode())
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        assert 'Optional file not allowed' in str(err.value)
        del data_dict['files_optional']['hello.exe']
        data_dict['includes'] = {'other.json': {}}
        del data_dict['signs']
        data_dict['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(data_dict, sort_keys=True), privatekey)}
        data = io.BytesIO(json.dumps(data_dict).encode())
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile(user_inner_path, data, ignore_same=False)
        assert 'Includes not allowed' in str(err.value)

    def testCert(self, site):
        if False:
            i = 10
            return i + 15
        user_priv = '5Kk7FSA63FC2ViKmKLuBxk9gQkaQ5713hKq8LmFAf4cVeXh6K6A'
        cert_priv = '5JusJDSjHaMHwUjDT3o6eQ54pA6poo8La5fAgn1wNc3iK59jxjA'
        assert 'data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json' in site.content_manager.contents
        user_content = site.content_manager.contents['data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json']
        rules_content = site.content_manager.contents['data/users/content.json']
        rules_content['user_contents']['cert_signers']['zeroid.bit'] = ['14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet', '1iD5ZQJMNXu43w1qLB8sfdHVKppVMduGz']
        rules = site.content_manager.getRules('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_content)
        assert rules['cert_signers'] == {'zeroid.bit': ['14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet', '1iD5ZQJMNXu43w1qLB8sfdHVKppVMduGz']}
        user_content['cert_sign'] = CryptBitcoin.sign('1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C#%s/%s' % (user_content['cert_auth_type'], user_content['cert_user_id'].split('@')[0]), cert_priv)
        assert site.content_manager.verifyCert('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_content)
        assert not site.content_manager.verifyCert('data/users/badaddress/content.json', user_content)
        signed_content = site.content_manager.sign('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_priv, filewrite=False)
        assert site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        cert_user_id = user_content['cert_user_id']
        site.content_manager.contents['data/users/content.json']['user_contents']['permissions'][cert_user_id] = False
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        assert 'Valid signs: 0/1' in str(err.value)
        del site.content_manager.contents['data/users/content.json']['user_contents']['permissions'][cert_user_id]
        user_content['cert_sign'] = CryptBitcoin.sign('badaddress#%s/%s' % (user_content['cert_auth_type'], user_content['cert_user_id']), cert_priv)
        signed_content = site.content_manager.sign('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_priv, filewrite=False)
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        assert 'Invalid cert' in str(err.value)
        user_content['cert_sign'] = CryptBitcoin.sign('1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C#%s/%s' % (user_content['cert_auth_type'], user_content['cert_user_id'].split('@')[0]), cert_priv)
        cert_user_id = user_content['cert_user_id']
        site.content_manager.contents['data/users/content.json']['user_contents']['permissions'][cert_user_id] = False
        site_privatekey = '5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv'
        del user_content['signs']
        user_content['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(user_content, sort_keys=True), site_privatekey)}
        assert site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(user_content).encode()), ignore_same=False)

    def testMissingCert(self, site):
        if False:
            print('Hello World!')
        user_priv = '5Kk7FSA63FC2ViKmKLuBxk9gQkaQ5713hKq8LmFAf4cVeXh6K6A'
        cert_priv = '5JusJDSjHaMHwUjDT3o6eQ54pA6poo8La5fAgn1wNc3iK59jxjA'
        user_content = site.content_manager.contents['data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json']
        rules_content = site.content_manager.contents['data/users/content.json']
        rules_content['user_contents']['cert_signers']['zeroid.bit'] = ['14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet', '1iD5ZQJMNXu43w1qLB8sfdHVKppVMduGz']
        user_content['cert_sign'] = CryptBitcoin.sign('1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C#%s/%s' % (user_content['cert_auth_type'], user_content['cert_user_id'].split('@')[0]), cert_priv)
        signed_content = site.content_manager.sign('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_priv, filewrite=False)
        assert site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        user_content['cert_user_id'] = 'nodomain'
        user_content['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(user_content, sort_keys=True), user_priv)}
        signed_content = site.content_manager.sign('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_priv, filewrite=False)
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        assert 'Invalid domain in cert_user_id' in str(err.value)
        del user_content['cert_user_id']
        del user_content['cert_auth_type']
        del user_content['signs']
        user_content['signs'] = {'1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT': CryptBitcoin.sign(json.dumps(user_content, sort_keys=True), user_priv)}
        signed_content = site.content_manager.sign('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_priv, filewrite=False)
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        assert 'Missing cert_user_id' in str(err.value)

    def testCertSignersPattern(self, site):
        if False:
            i = 10
            return i + 15
        user_priv = '5Kk7FSA63FC2ViKmKLuBxk9gQkaQ5713hKq8LmFAf4cVeXh6K6A'
        cert_priv = '5JusJDSjHaMHwUjDT3o6eQ54pA6poo8La5fAgn1wNc3iK59jxjA'
        user_content = site.content_manager.contents['data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json']
        rules_content = site.content_manager.contents['data/users/content.json']
        rules_content['user_contents']['cert_signers_pattern'] = '14wgQ[0-9][A-Z]'
        user_content['cert_user_id'] = 'certuser@14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet'
        user_content['cert_sign'] = CryptBitcoin.sign('1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C#%s/%s' % (user_content['cert_auth_type'], 'certuser'), cert_priv)
        signed_content = site.content_manager.sign('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', user_priv, filewrite=False)
        assert site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        rules_content['user_contents']['cert_signers_pattern'] = '14wgX[0-9][A-Z]'
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        assert 'Invalid cert signer: 14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet' in str(err.value)
        del rules_content['user_contents']['cert_signers_pattern']
        with pytest.raises(VerifyError) as err:
            site.content_manager.verifyFile('data/users/1J6UrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json', io.BytesIO(json.dumps(signed_content).encode()), ignore_same=False)
        assert 'Invalid cert signer: 14wgQ4VDDZNoRMFF4yCDuTrBSHmYhL3bet' in str(err.value)

    def testNewFile(self, site):
        if False:
            print('Hello World!')
        privatekey = '5KUh3PvNm5HUWoCfSUfcYvfQ2g3PrRNJWr6Q9eqdBGu23mtMntv'
        inner_path = 'data/users/1NEWrZMkarjVg5ax9W4qThir3BFUikbW6C/content.json'
        site.storage.writeJson(inner_path, {'test': 'data'})
        site.content_manager.sign(inner_path, privatekey)
        assert 'test' in site.storage.loadJson(inner_path)
        site.storage.delete(inner_path)
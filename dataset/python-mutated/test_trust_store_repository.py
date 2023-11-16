from sslyze.plugins.certificate_info.trust_stores.trust_store_repository import TrustStoresRepository

class TestTrustStoresRepository:

    def test_get_default(self):
        if False:
            for i in range(10):
                print('nop')
        repo = TrustStoresRepository.get_default()
        assert repo.get_main_store()
        assert len(repo.get_all_stores()) == 5

    def test_update_default(self):
        if False:
            i = 10
            return i + 15
        repo = TrustStoresRepository.update_default()
        assert repo.get_main_store()
        assert len(repo.get_all_stores()) == 5
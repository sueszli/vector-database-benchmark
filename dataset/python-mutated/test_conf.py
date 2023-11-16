from ninja.conf import settings

def test_default_configuration():
    if False:
        while True:
            i = 10
    assert settings.PAGINATION_CLASS == 'ninja.pagination.LimitOffsetPagination'
    assert settings.PAGINATION_PER_PAGE == 100
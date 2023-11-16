from Adyen.settings import API_CHECKOUT_VERSION, API_PAYMENT_VERSION

def test_adyen_api_version_not_changed():
    if False:
        print('Hello World!')
    assert API_CHECKOUT_VERSION == 'v64'
    assert API_PAYMENT_VERSION == 'v64'
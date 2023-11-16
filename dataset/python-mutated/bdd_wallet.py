from pytest_bdd import given
from pytest_bdd import scenario
from pytest_bdd import then
from pytest_bdd import when
import pytest

@scenario('bdd_wallet.feature', 'Buy fruits')
def test_publish():
    if False:
        for i in range(10):
            print('nop')
    pass

@pytest.fixture
def wallet():
    if False:
        print('Hello World!')

    class Wallet:
        amount = 0
    return Wallet()

@given('A wallet with 50')
def fill_wallet(wallet):
    if False:
        print('Hello World!')
    wallet.amount = 50

@when('I buy some apples for 1')
def buy_apples(wallet):
    if False:
        return 10
    wallet.amount -= 1

@when('I buy some bananas for 2')
def buy_bananas(wallet):
    if False:
        print('Hello World!')
    wallet.amount -= 2

@then('I have 47 left')
def check(wallet):
    if False:
        for i in range(10):
            print('nop')
    assert wallet.amount == 47
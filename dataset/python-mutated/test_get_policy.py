import pytest
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows, pytest.mark.slow_test]

@pytest.fixture(scope='module')
def lgpo(modules):
    if False:
        return 10
    return modules.lgpo

def test_hierarchical_return(lgpo):
    if False:
        print('Hello World!')
    result = lgpo.get_policy(policy_name='Calculator', policy_class='Machine', hierarchical_return=True)
    result = result['Administrative Templates']
    result = result['Windows Components']
    result = result['Microsoft User Experience Virtualization']
    result = result['Applications']
    result = result['Calculator']
    assert result in ('Enabled', 'Disabled', 'Not Configured')

def test_return_value_only_false(lgpo):
    if False:
        i = 10
        return i + 15
    result = lgpo.get_policy(policy_name='Calculator', policy_class='Machine', return_value_only=False)
    assert result['Windows Components\\Microsoft User Experience Virtualization\\Applications\\Calculator'] in ('Enabled', 'Disabled', 'Not Configured')

def test_return_full_policy_names_false(lgpo):
    if False:
        i = 10
        return i + 15
    result = lgpo.get_policy(policy_name='Calculator', policy_class='Machine', return_full_policy_names=False, return_value_only=False)
    assert result['Calculator'] in ('Enabled', 'Disabled', 'Not Configured')

def test_61860_calculator(lgpo):
    if False:
        return 10
    result = lgpo.get_policy(policy_name='Calculator', policy_class='Machine')
    assert result in ('Enabled', 'Disabled', 'Not Configured')
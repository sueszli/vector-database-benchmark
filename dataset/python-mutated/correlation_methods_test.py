import pandas as pd
from hamcrest import assert_that, close_to
from deepchecks.utils import correlation_methods
df = pd.DataFrame({'fName': ['Noam', 'Nir', 'Nadav', 'Sol', 'Noam'], 'Age': [1, 5, 2, 3, 4], 'Size': [310, 900, 1000, 300, 290], 'lName': ['Shir', 'Matan', 'Matan', 'Shir', 'Shir'], 'sName': ['JKL', 'JKL', 'JKL', 'JKL', 'JKL']})

def test_conditional_entropy():
    if False:
        return 10
    s_fname_lname = correlation_methods.conditional_entropy(df['fName'], df['lName'])
    assert_that(s_fname_lname, close_to(0.659, 0.01))
    s_lname_fname = correlation_methods.conditional_entropy(df['lName'], df['fName'])
    assert_that(s_lname_fname, close_to(0, 1e-05))
    s_fname_sname = correlation_methods.conditional_entropy(df['fName'], df['sName'])
    assert_that(s_fname_sname, close_to(1.3322, 0.01))

def test_theil_u():
    if False:
        return 10
    s_fname_lname = correlation_methods.theil_u_correlation(df['fName'], df['lName'])
    assert_that(s_fname_lname, close_to(0.50519, 0.01))
    s_lname_fname = correlation_methods.theil_u_correlation(df['lName'], df['fName'])
    assert_that(s_lname_fname, close_to(1, 1e-05))
    s_fname_sname = correlation_methods.theil_u_correlation(df['fName'], df['sName'])
    assert_that(s_fname_sname, close_to(0, 1e-05))
    s_fname_fname = correlation_methods.theil_u_correlation(df['fName'], df['fName'])
    assert_that(s_fname_fname, close_to(1, 1e-05))

def test_symmetric_theil_u():
    if False:
        for i in range(10):
            print('nop')
    s_fname_lname = correlation_methods.symmetric_theil_u_correlation(df['fName'], df['lName'])
    assert_that(s_fname_lname, close_to(0.6713, 0.01))
    s_lname_fname = correlation_methods.symmetric_theil_u_correlation(df['lName'], df['fName'])
    assert_that(s_lname_fname, close_to(0.6713, 0.01))
    s_fname_sname = correlation_methods.symmetric_theil_u_correlation(df['fName'], df['sName'])
    assert_that(s_fname_sname, close_to(0, 1e-05))
    s_fname_fname = correlation_methods.symmetric_theil_u_correlation(df['fName'], df['fName'])
    assert_that(s_fname_fname, close_to(1, 1e-05))

def test_correlation_ratio():
    if False:
        while True:
            i = 10
    cat_features = ['fName', 'lName', 'sName']
    df.loc[:, cat_features] = df.loc[:, cat_features].apply(lambda x: pd.factorize(x)[0])
    c_lname_age = correlation_methods.correlation_ratio(df['lName'], df['Age'])
    c_lname_size = correlation_methods.correlation_ratio(df['lName'], df['Size'])
    assert_that(c_lname_age, close_to(0.28867, 0.001))
    assert_that(c_lname_size, close_to(0.995, 0.001))
    c_lname_age = correlation_methods.correlation_ratio(df['lName'], df['Age'], [False, True, True, False, False])
    assert_that(c_lname_age, close_to(0, 0.001))
    c_sname_age = correlation_methods.correlation_ratio(df['sName'], df['Age'])
    c_sname_size = correlation_methods.correlation_ratio(df['sName'], df['Size'])
    assert_that(c_sname_age, close_to(0, 1e-05))
    assert_that(c_sname_size, close_to(0, 1e-05))
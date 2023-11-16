import pandas as pd
import pytest
from featuretools.tests.testing_utils import make_ecommerce_entityset

def test_tokenize_entityset(pd_es, pd_int_es):
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('dask', reason='Dask not installed, skipping')
    from dask.base import tokenize
    dupe = make_ecommerce_entityset()
    assert tokenize(pd_es) == tokenize(dupe)
    productless = make_ecommerce_entityset()
    productless.relationships.pop()
    assert tokenize(pd_es) != tokenize(productless)
    assert tokenize(pd_es) != tokenize(pd_int_es)
    cohorts_df = dupe['cohorts']
    new_row = pd.DataFrame(data={'cohort': [2], 'cohort_name': None, 'cohort_end': [pd.Timestamp('2011-04-08 12:00:00')]}, columns=['cohort', 'cohort_name', 'cohort_end'], index=[2])
    more_cohorts = pd.concat([cohorts_df, new_row])
    dupe.replace_dataframe(dataframe_name='cohorts', df=more_cohorts)
    assert tokenize(pd_es) == tokenize(dupe)
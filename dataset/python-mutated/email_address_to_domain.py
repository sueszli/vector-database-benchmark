import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, EmailAddress
from featuretools.primitives.base import TransformPrimitive

class EmailAddressToDomain(TransformPrimitive):
    """Determines the domain of an email

    Description:
        EmailAddress input should be a string. Will return Nan
        if an invalid email address is provided, or if the input is
        not a string.

    Examples:
        >>> email_address_to_domain = EmailAddressToDomain()
        >>> email_address_to_domain(['name@gmail.com', 'name@featuretools.com']).tolist()
        ['gmail.com', 'featuretools.com']
    """
    name = 'email_address_to_domain'
    input_types = [ColumnSchema(logical_type=EmailAddress)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        if False:
            i = 10
            return i + 15

        def email_address_to_domain(emails):
            if False:
                for i in range(10):
                    print('nop')
            if len(emails) == 0:
                return pd.Series([], dtype='category')
            emails_df = pd.DataFrame({'email': emails})
            if emails_df['email'].isnull().all():
                emails_df['domain'] = np.nan
                emails_df['domain'] = emails_df['domain'].astype(object)
            else:
                emails_df['domain'] = emails_df['email'].str.strip().str.split('@', expand=True)[1]
            return emails_df.domain.values
        return email_address_to_domain
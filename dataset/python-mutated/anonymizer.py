"""
Helper class to anonymize a dataframe head by replacing the values of the columns
that contain personal or sensitive information with random values.
"""
import random
import re
import string
import pandas as pd

class Anonymizer:

    def _is_valid_email(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if the given email is valid based on regex pattern.\n\n        Args:\n            email (str): email address to be checked.\n\n        Returns (bool): True if the email is valid, otherwise False.\n        '
        email_regex = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        return re.match(email_regex, self) is not None

    def _is_valid_phone_number(self) -> bool:
        if False:
            print('Hello World!')
        'Check if the given phone number is valid based on regex pattern.\n\n        Args:\n            phone_number (str): phone number to be checked.\n\n        Returns (bool): True if the phone number is valid, otherwise False.\n        '
        pattern = '\\b(?:\\+?\\d{1,3}[- ]?)?\\(?\\d{3}\\)?[- ]?\\d{3}[- ]?\\d{4}\\b'
        return re.search(pattern, self) is not None

    def _is_valid_credit_card(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if the given credit card number is valid based on regex pattern.\n\n        Args:\n            credit_card_number (str): credit card number to be checked.\n\n        Returns (str): True if the credit card number is valid, otherwise False.\n        '
        pattern = '^\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}$'
        return re.search(pattern, self) is not None

    def _generate_random_email() -> str:
        if False:
            print('Hello World!')
        'Generates a random email address using predefined domains.\n\n        Returns (str): generated random email address.\n        '
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com', 'aol.com', 'protonmail.com', 'zoho.com']
        name_length = random.randint(6, 12)
        domain = random.choice(domains)
        letters = string.ascii_lowercase + string.digits + '-_'
        username = ''.join((random.choice(letters) for _ in range(name_length)))
        return f'{username}@' + domain

    def _generate_random_phone_number(self) -> str:
        if False:
            return 10
        'Generate a random phone number with country code if originally present.\n\n        Args:\n            original_field (str): original phone number field.\n\n        Returns (str): generated random phone number.\n        '
        country_code = self.split()[0] if self.startswith('+') else ''
        number = ''.join(random.choices('0123456789', k=10))
        return f'{country_code} {number}' if country_code else number

    def _generate_random_credit_card() -> str:
        if False:
            i = 10
            return i + 15
        'Generate a random credit card number.\n\n        Returns (str): generated random credit card number.\n        '
        groups = []
        for _i in range(4):
            group = ''.join(random.choices('0123456789', k=4))
            groups.append(group)
        separator = random.choice(['-', ' '])
        return separator.join(groups)

    def anonymize_dataframe_head(self) -> pd.DataFrame:
        if False:
            print('Hello World!')
        '\n        Anonymize a dataframe head by replacing the values of the columns\n        that contain personal or sensitive information with random values.\n\n        Args:\n            df (pd.DataFrame): Dataframe to anonymize.\n\n        Returns:\n            pd.DataFrame: Anonymized dataframe.\n        '
        if len(self) == 0:
            return self
        df_head = self.head().copy()
        for col in df_head.columns:
            if Anonymizer._is_valid_email(str(df_head[col].iloc[0])):
                df_head[col] = df_head[col].apply(lambda x: Anonymizer._generate_random_email())
            elif Anonymizer._is_valid_phone_number(str(df_head[col].iloc[0])):
                df_head[col] = df_head[col].apply(lambda x: Anonymizer._generate_random_phone_number(str(x)))
            elif Anonymizer._is_valid_credit_card(str(df_head[col].iloc[0])):
                df_head[col] = df_head[col].apply(lambda x: Anonymizer._generate_random_credit_card())
        return df_head
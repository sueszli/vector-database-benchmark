"""
Purpose

Shows how to use AWS Identity and Access Management (IAM) accounts.
"""
import logging
import pprint
import sys
import time
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)
iam = boto3.resource('iam')

def create_alias(alias):
    if False:
        print('Hello World!')
    '\n    Creates an alias for the current account. The alias can be used in place of the\n    account ID in the sign-in URL. An account can have only one alias. When a new\n    alias is created, it replaces any existing alias.\n\n    :param alias: The alias to assign to the account.\n    '
    try:
        iam.create_account_alias(AccountAlias=alias)
        logger.info("Created an alias '%s' for your account.", alias)
    except ClientError:
        logger.exception("Couldn't create alias '%s' for your account.", alias)
        raise

def delete_alias(alias):
    if False:
        print('Hello World!')
    '\n    Removes the alias from the current account.\n\n    :param alias: The alias to remove.\n    '
    try:
        iam.meta.client.delete_account_alias(AccountAlias=alias)
        logger.info("Removed alias '%s' from your account.", alias)
    except ClientError:
        logger.exception("Couldn't remove alias '%s' from your account.", alias)
        raise

def list_aliases():
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets the list of aliases for the current account. An account has at most one alias.\n\n    :return: The list of aliases for the account.\n    '
    try:
        response = iam.meta.client.list_account_aliases()
        aliases = response['AccountAliases']
        if len(aliases) > 0:
            logger.info('Got aliases for your account: %s.', ','.join(aliases))
        else:
            logger.info('Got no aliases for your account.')
    except ClientError:
        logger.exception("Couldn't list aliases for your account.")
        raise
    else:
        return response['AccountAliases']

def get_authorization_details(response_filter):
    if False:
        print('Hello World!')
    '\n    Gets an authorization detail report for the current account.\n\n    :param response_filter: A list of resource types to include in the report, such\n                            as users or roles. When not specified, all resources\n                            are included.\n    :return: The authorization detail report.\n    '
    try:
        account_details = iam.meta.client.get_account_authorization_details(Filter=response_filter)
        logger.debug(account_details)
    except ClientError:
        logger.exception("Couldn't get details for your account.")
        raise
    else:
        return account_details

def get_summary():
    if False:
        i = 10
        return i + 15
    '\n    Gets a summary of account usage.\n\n    :return: The summary of account usage.\n    '
    try:
        summary = iam.AccountSummary()
        logger.debug(summary.summary_map)
    except ClientError:
        logger.exception("Couldn't get a summary for your account.")
        raise
    else:
        return summary.summary_map

def generate_credential_report():
    if False:
        return 10
    '\n    Starts generation of a credentials report about the current account. After\n    calling this function to generate the report, call get_credential_report\n    to get the latest report. A new report can be generated a minimum of four hours\n    after the last one was generated.\n    '
    try:
        response = iam.meta.client.generate_credential_report()
        logger.info('Generating credentials report for your account. Current state is %s.', response['State'])
    except ClientError:
        logger.exception("Couldn't generate a credentials report for your account.")
        raise
    else:
        return response

def get_credential_report():
    if False:
        print('Hello World!')
    '\n    Gets the most recently generated credentials report about the current account.\n\n    :return: The credentials report.\n    '
    try:
        response = iam.meta.client.get_credential_report()
        logger.debug(response['Content'])
    except ClientError:
        logger.exception("Couldn't get credentials report.")
        raise
    else:
        return response['Content']

def print_password_policy():
    if False:
        while True:
            i = 10
    '\n    Prints the password policy for the account.\n    '
    try:
        pw_policy = iam.AccountPasswordPolicy()
        print('Current account password policy:')
        print(f'\tallow_users_to_change_password: {pw_policy.allow_users_to_change_password}')
        print(f'\texpire_passwords: {pw_policy.expire_passwords}')
        print(f'\thard_expiry: {pw_policy.hard_expiry}')
        print(f'\tmax_password_age: {pw_policy.max_password_age}')
        print(f'\tminimum_password_length: {pw_policy.minimum_password_length}')
        print(f'\tpassword_reuse_prevention: {pw_policy.password_reuse_prevention}')
        print(f'\trequire_lowercase_characters: {pw_policy.require_lowercase_characters}')
        print(f'\trequire_numbers: {pw_policy.require_numbers}')
        print(f'\trequire_symbols: {pw_policy.require_symbols}')
        print(f'\trequire_uppercase_characters: {pw_policy.require_uppercase_characters}')
        printed = True
    except ClientError as error:
        if error.response['Error']['Code'] == 'NoSuchEntity':
            print('The account does not have a password policy set.')
        else:
            logger.exception("Couldn't get account password policy.")
            raise
    else:
        return printed

def list_saml_providers(count):
    if False:
        i = 10
        return i + 15
    '\n    Lists the SAML providers for the account.\n\n    :param count: The maximum number of providers to list.\n    '
    try:
        found = 0
        for provider in iam.saml_providers.limit(count):
            logger.info('Got SAML provider %s.', provider.arn)
            found += 1
        if found == 0:
            logger.info('Your account has no SAML providers.')
    except ClientError:
        logger.exception("Couldn't list SAML providers.")
        raise

def usage_demo():
    if False:
        i = 10
        return i + 15
    'Shows how to use the account functions.'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('-' * 88)
    print('Welcome to the AWS Identity and Account Management account demo.')
    print('-' * 88)
    print('Setting an account alias lets you use the alias in your sign-in URL instead of your account number.')
    old_aliases = list_aliases()
    if len(old_aliases) > 0:
        print(f"Your account currently uses '{old_aliases[0]}' as its alias.")
    else:
        print('Your account currently has no alias.')
    for index in range(1, 3):
        new_alias = f'alias-{index}-{time.time_ns()}'
        print(f'Setting your account alias to {new_alias}')
        create_alias(new_alias)
    current_aliases = list_aliases()
    print(f'Your account alias is now {current_aliases}.')
    delete_alias(current_aliases[0])
    print(f'Your account now has no alias.')
    if len(old_aliases) > 0:
        print(f'Restoring your original alias back to {old_aliases[0]}...')
        create_alias(old_aliases[0])
    print('-' * 88)
    print('You can get various reports about your account.')
    print("Let's generate a credentials report...")
    report_state = None
    while report_state != 'COMPLETE':
        cred_report_response = generate_credential_report()
        old_report_state = report_state
        report_state = cred_report_response['State']
        if report_state != old_report_state:
            print(report_state, sep='')
        else:
            print('.', sep='')
        sys.stdout.flush()
        time.sleep(1)
    print()
    cred_report = get_credential_report()
    col_count = 3
    print(f'Got credentials report. Showing only the first {col_count} columns.')
    cred_lines = [line.split(',')[:col_count] for line in cred_report.decode('utf-8').split('\n')]
    col_width = max([len(item) for line in cred_lines for item in line]) + 2
    for line in cred_report.decode('utf-8').split('\n'):
        print(''.join((element.ljust(col_width) for element in line.split(',')[:col_count])))
    print('-' * 88)
    print("Let's get an account summary.")
    summary = get_summary()
    print("Here's your summary:")
    pprint.pprint(summary)
    print('-' * 88)
    print("Let's get authorization details!")
    details = get_authorization_details([])
    see_details = input('These are pretty long, do you want to see them (y/n)? ')
    if see_details.lower() == 'y':
        pprint.pprint(details)
    print('-' * 88)
    pw_policy_created = None
    see_pw_policy = input('Want to see the password policy for the account (y/n)? ')
    if see_pw_policy.lower() == 'y':
        while True:
            if print_password_policy():
                break
            else:
                answer = input('Do you want to create a default password policy (y/n)? ')
                if answer.lower() == 'y':
                    pw_policy_created = iam.create_account_password_policy()
                else:
                    break
    if pw_policy_created is not None:
        answer = input('Do you want to delete the password policy (y/n)? ')
        if answer.lower() == 'y':
            pw_policy_created.delete()
            print('Password policy deleted.')
    print('The SAML providers for your account are:')
    list_saml_providers(10)
    print('-' * 88)
    print('Thanks for watching.')
if __name__ == '__main__':
    usage_demo()
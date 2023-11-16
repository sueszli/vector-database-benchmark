import logging
import sys
from moneyed import CURRENCIES
from django.db import migrations, connection
from company.models import SupplierPriceBreak
logger = logging.getLogger('inventree')

def migrate_currencies(apps, schema_editor):
    if False:
        return 10
    '\n    Migrate from the \'old\' method of handling currencies,\n    to the new method which uses the django-money library.\n\n    Previously, we created a custom Currency model,\n    which was very simplistic.\n\n    Here we will attempt to map each existing "currency" reference\n    for the SupplierPriceBreak model, to a new django-money compatible currency.\n    '
    logger.debug('Updating currency references for SupplierPriceBreak model...')
    currency_codes = CURRENCIES.keys()
    cursor = connection.cursor()
    response = cursor.execute('SELECT id, suffix, description from common_currency;')
    results = cursor.fetchall()
    remap = {}
    for (index, row) in enumerate(results):
        (pk, suffix, description) = row
        suffix = suffix.strip().upper()
        if suffix not in currency_codes:
            logger.warning(f"Missing suffix: '{suffix}'")
            while suffix not in currency_codes:
                print(f"Could not find a valid currency matching '{suffix}'.")
                print('Please enter a valid currency code')
                suffix = str(input('> ')).strip()
        if pk not in remap.keys():
            remap[pk] = suffix
    response = cursor.execute('SELECT id, cost, currency_id, price, price_currency from part_supplierpricebreak;')
    results = cursor.fetchall()
    count = 0
    for (index, row) in enumerate(results):
        (pk, cost, currency_id, price, price_currency) = row
        response = cursor.execute(f'UPDATE part_supplierpricebreak set price={cost} where id={pk};')
        currency_code = remap.get(currency_id, 'USD')
        response = cursor.execute(f"UPDATE part_supplierpricebreak set price_currency= '{currency_code}' where id={pk};")
        count += 1
    if count > 0:
        logger.info(f'Updated {count} SupplierPriceBreak rows')

def reverse_currencies(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reverse the "update" process.\n\n    Here we may be in the situation that the legacy "Currency" table is empty,\n    and so we have to re-populate it based on the new price_currency codes.\n    '
    print('Reversing currency migration...')
    cursor = connection.cursor()
    response = cursor.execute(f'SELECT id, price, price_currency from part_supplierpricebreak;')
    results = cursor.fetchall()
    codes_in_use = set()
    for (index, row) in enumerate(results):
        (pk, price, code) = row
        codes_in_use.add(code)
        response = cursor.execute(f'UPDATE part_supplierpricebreak set cost={price} where id={pk};')
    code_map = {}
    for code in codes_in_use:
        response = cursor.execute(f"SELECT id, suffix from common_currency where suffix='{code}';")
        row = cursor.fetchone()
        if row is not None:
            (pk, suffix) = row
            code_map[suffix] = pk
        else:
            description = CURRENCIES[code]
            print(f'Creating new Currency object for {code}')
            query = f'INSERT into common_currency (symbol, suffix, description, value, base) VALUES ("$", "{code}", "{description}", 1.0, False);'
            response = cursor.execute(query)
            code_map[code] = cursor.lastrowid
    for suffix in code_map.keys():
        pk = code_map[suffix]
        print(f'Currency {suffix} -> pk {pk}')
        response = cursor.execute(f"UPDATE part_supplierpricebreak set currency_id={pk} where price_currency='{suffix}';")

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('company', '0025_auto_20201110_1001')]
    operations = [migrations.RunPython(migrate_currencies, reverse_code=reverse_currencies)]
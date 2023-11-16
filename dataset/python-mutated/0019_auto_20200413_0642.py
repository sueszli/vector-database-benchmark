import sys
import os
from rapidfuzz import fuzz
from django.db import migrations, connection
from django.db.utils import OperationalError, ProgrammingError
'\nWhen this migration is tested by CI, it cannot accept user input.\nSo a simplified version of the migration is implemented.\n'
TESTING = 'test' in sys.argv

def clear():
    if False:
        while True:
            i = 10
    if not TESTING:
        os.system('cls' if os.name == 'nt' else 'clear')

def reverse_association(apps, schema_editor):
    if False:
        return 10
    "\n    This is the 'reverse' operation of the manufacturer reversal.\n    This operation is easier:\n\n    For each SupplierPart object, copy the name of the 'manufacturer' field\n    into the 'manufacturer_name' field.\n    "
    cursor = connection.cursor()
    response = cursor.execute('select id, "MPN" from part_supplierpart;')
    supplier_parts = cursor.fetchall()
    if len(supplier_parts) == 0:
        return
    print('Reversing migration for manufacturer association')
    for (index, row) in enumerate(supplier_parts):
        (supplier_part_id, MPN) = row
        print(f'Checking SupplierPart [{supplier_part_id}]:')
        response = cursor.execute(f'SELECT manufacturer_id FROM part_supplierpart WHERE id={supplier_part_id};')
        manufacturer_id = None
        row = cursor.fetchone()
        if len(row) > 0:
            try:
                manufacturer_id = int(row[0])
            except (TypeError, ValueError):
                pass
        if manufacturer_id is None:
            print(' - Manufacturer ID not set: Skipping')
            continue
        print(' - Manufacturer ID: [{id}]'.format(id=manufacturer_id))
        response = cursor.execute(f'SELECT name from company_company where id={manufacturer_id};')
        row = cursor.fetchone()
        name = row[0]
        print(" - Manufacturer name: '{name}'".format(name=name))
        response = cursor.execute("UPDATE part_supplierpart SET manufacturer_name='{name}' WHERE id={ID};".format(name=name, ID=supplier_part_id))

def associate_manufacturers(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    '\n    This migration is the "middle step" in migration of the "manufacturer" field for the SupplierPart model.\n\n    Previously the "manufacturer" field was a simple text field with the manufacturer name.\n    This is quite insufficient.\n    The new "manufacturer" field is a link to Company object which has the "is_manufacturer" parameter set to True\n\n    This migration requires user interaction to create new "manufacturer" Company objects,\n    based on the text value in the "manufacturer_name" field (which was created in the previous migration).\n\n    It uses fuzzy pattern matching to help the user out as much as possible.\n    '

    def get_manufacturer_name(part_id):
        if False:
            return 10
        "\n        THIS IS CRITICAL!\n\n        Once the pythonic representation of the model has removed the 'manufacturer_name' field,\n        it is NOT ACCESSIBLE by calling SupplierPart.manufacturer_name.\n\n        However, as long as the migrations are applied in order, then the table DOES have a field called 'manufacturer_name'.\n\n        So, we just need to request it using dirty SQL.\n        "
        query = 'SELECT manufacturer_name from part_supplierpart where id={ID};'.format(ID=part_id)
        cursor = connection.cursor()
        response = cursor.execute(query)
        row = cursor.fetchone()
        if len(row) > 0:
            return row[0]
        return ''
    cursor = connection.cursor()
    response = cursor.execute(f'select id, "MPN" from part_supplierpart;')
    supplier_parts = cursor.fetchall()
    if len(supplier_parts) == 0:
        return
    links = {}
    companies = {}
    response = cursor.execute('select id, name from company_company;')
    results = cursor.fetchall()
    for (index, row) in enumerate(results):
        (pk, name) = row
        companies[name] = pk

    def link_part(part_id, name):
        if False:
            return 10
        ' Attempt to link Part to an existing Company '
        if name in companies.keys():
            print(" - Part[{pk}]: '{n}' maps to existing manufacturer".format(pk=part_id, n=name))
            manufacturer_id = companies[name]
            query = f'update part_supplierpart set manufacturer_id={manufacturer_id} where id={part_id};'
            result = cursor.execute(query)
            return True
        if name in links.keys():
            print(" - Part[{pk}]: Mapped '{n}' - manufacturer <{c}>".format(pk=part_id, n=name, c=links[name]))
            manufacturer_id = links[name]
            query = f'update part_supplierpart set manufacturer_id={manufacturer_id} where id={part_id};'
            result = cursor.execute(query)
            return True
        return False

    def create_manufacturer(part_id, input_name, company_name):
        if False:
            i = 10
            return i + 15
        ' Create a new manufacturer '
        Company = apps.get_model('company', 'company')
        manufacturer = Company.objects.create(name=company_name, description=company_name, is_manufacturer=True)
        links[input_name] = manufacturer.pk
        links[company_name] = manufacturer.pk
        companies[company_name] = manufacturer.pk
        print(" - Part[{pk}]: Created new manufacturer: '{name}'".format(pk=part_id, name=company_name))
        cursor.execute(f'update part_supplierpart set manufacturer_id={manufacturer.pk} where id={part_id};')

    def find_matches(text, threshold=65):
        if False:
            while True:
                i = 10
        "\n        Attempt to match a 'name' to an existing Company.\n        A list of potential matches will be returned.\n        "
        matches = []
        for name in companies.keys():
            ratio = fuzz.partial_ratio(name.lower(), text.lower())
            if ratio > threshold:
                matches.append({'name': name, 'match': ratio})
        if len(matches) > 0:
            return [match['name'] for match in sorted(matches, key=lambda item: item['match'], reverse=True)]
        else:
            return []

    def map_part_to_manufacturer(part_id, idx, total):
        if False:
            i = 10
            return i + 15
        cursor = connection.cursor()
        name = get_manufacturer_name(part_id)
        if not name or len(name) == 0:
            print(' - Part[{pk}]: No manufacturer_name provided, skipping'.format(pk=part_id))
            return
        if link_part(part_id, name):
            return
        matches = find_matches(name)
        clear()
        if not TESTING:
            print('----------------------------------')
        print('Checking part [{pk}] ({idx} of {total})'.format(pk=part_id, idx=idx + 1, total=total))
        if not TESTING:
            print("Manufacturer name: '{n}'".format(n=name))
            print('----------------------------------')
            print('Select an option from the list below:')
            print("0) - Create new manufacturer '{n}'".format(n=name))
            print('')
            for (i, m) in enumerate(matches[:10]):
                print("{i}) - Use manufacturer '{opt}'".format(i=i + 1, opt=m))
            print('')
            print('OR - Type a new custom manufacturer name')
        while True:
            if TESTING:
                response = '0'
            else:
                response = str(input('> ')).strip()
            try:
                n = int(response)
                if n == 0:
                    create_manufacturer(part_id, name, name)
                    return
                else:
                    n = n - 1
                    if n < len(matches):
                        company_name = matches[n]
                        company_id = companies[company_name]
                        cursor.execute(f'update company_company set is_manufacturer=true where id={company_id};')
                        cursor.execute(f'update part_supplierpart set manufacturer_id={company_id} where id={part_id};')
                        links[name] = company_id
                        links[company_name] = company_id
                        print(" - Part[{pk}]: Linked '{n}' to manufacturer '{m}'".format(pk=part_id, n=name, m=company_name))
                        return
                    else:
                        print('Please select a valid option')
            except ValueError:
                if not response or len(response) == 0:
                    print('Please select an option')
                elif response in companies.keys():
                    link_part(part, companies[response])
                    return
                elif response in links.keys():
                    link_part(part, links[response])
                    return
                else:
                    create_manufacturer(part_id, name, response)
                    return
    clear()
    print('')
    clear()
    if not TESTING:
        print('---------------------------------------')
        print('The SupplierPart model needs to be migrated,')
        print("as the new 'manufacturer' field maps to a 'Company' reference.")
        print("The existing 'manufacturer_name' field will be used to match")
        print('against possible companies.')
        print('This process requires user input.')
        print('')
        print('Note: This process MUST be completed to migrate the database.')
        print('---------------------------------------')
        print('')
        input('Press <ENTER> to continue.')
    clear()
    cursor = connection.cursor()
    response = cursor.execute('select id, "MPN", "SKU", manufacturer_id, manufacturer_name from part_supplierpart;')
    results = cursor.fetchall()
    part_count = len(results)
    for (index, row) in enumerate(results):
        (pk, MPN, SKU, manufacturer_id, manufacturer_name) = row
        if manufacturer_id is not None:
            print(f' - SupplierPart <{pk}> already has a manufacturer associated (skipping)')
            continue
        map_part_to_manufacturer(pk, index, part_count)
    print('Done!')

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('company', '0018_supplierpart_manufacturer')]
    operations = [migrations.RunPython(associate_manufacturers, reverse_code=reverse_association)]
from typing import Optional
from django.db import migrations
from django_countries import countries
VATLAYER_ID = 'mirumee.taxes.vatlayer'
TAX_CLASS_ZERO_RATE = 'No Taxes'

def _clear_country_code(country_code: str) -> Optional[str]:
    if False:
        return 10
    return countries.alpha2(country_code.strip()) if country_code else None

def _clear_str_list_country_codes(country_codes: str) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    countries = [_clear_country_code(cc) for cc in country_codes.split(',')]
    return [cc for cc in countries if cc]

def create_tax_configurations(apps, vatlayer_configs):
    if False:
        i = 10
        return i + 15
    TaxConfigurationPerCountry = apps.get_model('tax', 'TaxConfigurationPerCountry')
    use_origin_country_map = {}
    for vatlayer_config in vatlayer_configs:
        config_dict = {item['name']: item['value'] for item in vatlayer_config.configuration}
        channel = vatlayer_config.channel
        origin_country = _clear_country_code(config_dict.get('origin_country', ''))
        countries_to_calculate_taxes_from_origin = _clear_str_list_country_codes(config_dict.get('countries_to_calculate_taxes_from_origin', ''))
        if origin_country and countries_to_calculate_taxes_from_origin:
            for country in countries_to_calculate_taxes_from_origin:
                use_origin_country_map[country] = origin_country
        excluded_countries = _clear_str_list_country_codes(config_dict.get('excluded_countries', ''))
        if excluded_countries:
            tax_configuration = channel.tax_configuration
            for country in excluded_countries:
                TaxConfigurationPerCountry.objects.update_or_create(tax_configuration=tax_configuration, country=country, defaults={'charge_taxes': False})
    return use_origin_country_map

def create_tax_rates(apps, use_origin_country_map):
    if False:
        for i in range(10):
            print('nop')
    TaxClass = apps.get_model('tax', 'TaxClass')
    TaxClassCountryRate = apps.get_model('tax', 'TaxClassCountryRate')
    tax_classes = TaxClass.objects.exclude(name=TAX_CLASS_ZERO_RATE)
    try:
        VAT = apps.get_model('django_prices_vatlayer', 'VAT')
    except LookupError:
        vat_rates = []
    else:
        vat_rates = VAT.objects.all()
    rates = {}
    for tax_class in tax_classes:
        for vat in vat_rates:
            standard_rate = TaxClassCountryRate(tax_class=tax_class, country=vat.country_code, rate=vat.data['standard_rate'])
            rates[tax_class.id, vat.country_code] = standard_rate
            if tax_class.name in vat.data['reduced_rates']:
                reduced_rate = TaxClassCountryRate(tax_class=tax_class, country=vat.country_code, rate=vat.data['reduced_rates'][tax_class.name])
                rates[tax_class.id, vat.country_code] = reduced_rate
        for (country_code, origin) in use_origin_country_map.items():
            country_rate_obj = rates.get((tax_class.id, country_code))
            origin_rate_obj = rates.get((tax_class.id, origin))
            if country_rate_obj and origin_rate_obj:
                country_rate_obj.rate = origin_rate_obj.rate
                rates[tax_class.id, country_code] = country_rate_obj
    TaxClassCountryRate.objects.bulk_create(rates.values())

def migrate_vatlayer(apps, _schema_editor):
    if False:
        while True:
            i = 10
    PluginConfiguration = apps.get_model('plugins', 'PluginConfiguration')
    vatlayer_configs = PluginConfiguration.objects.filter(active=True, identifier=VATLAYER_ID)
    is_vatlayer_enabled = vatlayer_configs.exists()
    if is_vatlayer_enabled:
        use_origin_country_map = create_tax_configurations(apps, vatlayer_configs)
        create_tax_rates(apps, use_origin_country_map)

class Migration(migrations.Migration):
    dependencies = [('tax', '0004_migrate_tax_classes')]
    operations = [migrations.RunPython(migrate_vatlayer, migrations.RunPython.noop)]
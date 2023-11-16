from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.google_sheets import GoogleSheets
from pandas import DataFrame
from os import path
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_to_google_sheet(df: DataFrame, **kwargs) -> None:
    if False:
        while True:
            i = 10
    '\n    Template for exporting data to a worksheet in a Google Sheet.\n    Specify your configuration settings in \'io_config.yaml\'.\n\n    Sheet Name or ID may also be used instead of URL\n    sheet_id = "your_sheet_id"\n    sheet_name = "your_sheet_name"\n\n    Worksheet position or name may also be specified\n    worksheet_position = 0\n    worksheet_name = "your_worksheet_name"\n\n    Docs: [TODO]\n    '
    config_path = path.join(get_repo_path(), 'io_config.yaml')
    config_profile = 'default'
    sheet_url = 'your_sheet_url'
    GoogleSheets.with_config(ConfigFileLoader(config_path, config_profile)).export(df, sheet_url=sheet_url)
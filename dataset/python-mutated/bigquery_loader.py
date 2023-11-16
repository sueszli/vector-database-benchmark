import os
from pathlib import Path
from google.oauth2.service_account import Credentials
base_path = Path(__file__).parent.parent.parent
creds_file = base_path / 'bigquery_utils' / 'bigquery_service_account.json'
credentials = Credentials.from_service_account_file(creds_file)

def insert_to_bigquery(df, table):
    if False:
        print('Hello World!')
    df.to_gbq(destination_table=f"{os.environ['dataset']}.{table}", project_id=os.environ['project_id'], if_exists='append', credentials=credentials)

def transit_insert_to_bigquery(db, batch):
    if False:
        while True:
            i = 10
    ...
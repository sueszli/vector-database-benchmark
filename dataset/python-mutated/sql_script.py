import sys
from dagster_pipes import open_dagster_pipes

class SomeSqlClient:

    def query(self, query_str: str) -> None:
        if False:
            while True:
                i = 10
        sys.stderr.write(f'Querying "{query_str}"\n')
if __name__ == '__main__':
    sql = sys.argv[1]
    with open_dagster_pipes() as context:
        client = SomeSqlClient()
        client.query(sql)
        context.report_asset_materialization(metadata={'sql': sql})
        context.log.info(f'Ran {sql}')
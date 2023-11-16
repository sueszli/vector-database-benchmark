import os
from typing import Dict, List
from h2o_wave import main, app, Q, ui
import pandas as pd
all_issues_df = pd.DataFrame([[i + 1, 'Closed' if i % 2 == 0 else 'Open'] for i in range(100)], columns=['text', 'status'])
rows_per_page = 10
total_rows = len(all_issues_df)

def df_to_table_rows(df: pd.DataFrame) -> List[ui.TableRow]:
    if False:
        for i in range(10):
            print('nop')
    return [ui.table_row(name=str(r[0]), cells=[str(r[0]), r[1]]) for r in df.itertuples(index=False)]

def get_df(base: pd.DataFrame, sort: Dict[str, bool]=None, search: Dict=None, filters: Dict[str, List[str]]=None) -> pd.DataFrame:
    if False:
        print('Hello World!')
    df = base.copy()
    if sort:
        ascending = [not v for v in list(sort.values())]
        df = df.sort_values(by=list(sort.keys()), ascending=ascending)
    if search:
        search_val = search['value'].lower()
        df = df[df.apply(lambda r: any((search_val in str(r[col]).lower() for col in search['cols'])), axis=1)]
    if filters:
        query = ' & '.join([f'({not bool(filters)} | {col} in {filters})' for (col, filters) in filters.items()])
        df = df.query(query)
    return df

@app('/demo')
async def serve(q: Q):
    if not q.client.initialized:
        q.page['meta'] = ui.meta_card(box='')
        q.page['form'] = ui.form_card(box='1 1 -1 -1', items=[ui.table(name='table', columns=[ui.table_column(name='text', label='Text', sortable=True, searchable=True, link=False), ui.table_column(name='status', label='Status', filterable=True, filters=['Open', 'Closed'])], rows=df_to_table_rows(get_df(all_issues_df)[0:rows_per_page]), resettable=True, downloadable=True, pagination=ui.table_pagination(total_rows, rows_per_page), events=['sort', 'filter', 'search', 'page_change', 'download', 'reset'])])
        q.client.initialized = True
    if q.events.table:
        table = q.page['form'].table
        if q.events.table.sort:
            q.client.sort = q.events.table.sort
            q.client.page_offset = 0
        if q.events.table.filter:
            q.client.filters = q.events.table.filter
            q.client.page_offset = 0
        if q.events.table.search is not None:
            q.client.search = q.events.table.search
            q.client.page_offset = 0
        if q.events.table.page_change:
            q.client.page_offset = q.events.table.page_change.get('offset', 0)
        if q.events.table.reset:
            q.client.search = None
            q.client.sort = None
            q.client.filters = None
            q.client.page_offset = 0
            table.pagination = ui.table_pagination(total_rows, rows_per_page)
        offset = q.client.page_offset or 0
        df = get_df(all_issues_df, q.client.sort, q.client.search, q.client.filters)
        if q.events.table.download:
            df.to_csv('data_download.csv')
            (download_url,) = await q.site.upload(['data_download.csv'])
            os.remove('data_download.csv')
            q.page['meta'].script = ui.inline_script(f'window.open("{download_url}")')
        if q.client.search is not None or q.client.filters:
            table.pagination = ui.table_pagination(len(df), rows_per_page)
        table.rows = df_to_table_rows(df[offset:offset + rows_per_page])
    await q.page.save()
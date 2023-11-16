import os
from time import time
import h2o
from h2o_wave import Q, app, main, ui
from loguru import logger

@app('/demo')
async def serve(q: Q):
    logger.info(q.args)
    logger.info(q.events)
    if not q.app.initialized:
        try:
            h2o.connect(url='http://127.0.0.1:54321')
        except:
            q.page['err'] = ui.form_card(box='1 1 4 2', items=[ui.message_bar(type='error', text='Could not connect to H2O3. Please ensure H2O3 is running.')])
            await q.page.save()
            logger.error('H2O-3 is not running')
            return
        q.app.h2o_df = h2o.get_frame('py_6_sid_aff3')
        q.app.rows_per_page = 10
        q.app.column_sortable = q.app.h2o_df.isnumeric()
        q.app.column_filterable = q.app.h2o_df.isfactor()
        q.app.column_searchable = q.app.h2o_df.isfactor() + q.app.h2o_df.isstring()
        q.app.initialized = True
    if not q.client.initialized:
        q.client.search = None
        q.client.sort = None
        q.client.filters = None
        q.client.page_offset = 0
        q.client.total_rows = len(q.app.h2o_df)
        q.page['meta'] = ui.meta_card(box='')
        q.page['table_card'] = ui.form_card(box='1 1 -1 -1', items=[ui.table(name='h2o_table', columns=[ui.table_column(name=q.app.h2o_df.columns[i], label=q.app.h2o_df.columns[i], sortable=q.app.column_sortable[i], filterable=q.app.column_filterable[i], searchable=q.app.column_searchable[i]) for i in range(len(q.app.h2o_df.columns))], rows=get_table_rows(q), resettable=True, downloadable=True, pagination=ui.table_pagination(total_rows=q.client.total_rows, rows_per_page=q.app.rows_per_page), events=['page_change', 'sort', 'filter', 'search', 'reset', 'download'])])
        q.client.initialized = True
    if q.events.h2o_table:
        logger.info('table event occurred')
        if q.events.h2o_table.page_change:
            logger.info(f'table page change: {q.events.h2o_table.page_change}')
            q.client.page_offset = q.events.h2o_table.page_change.get('offset', 0)
        if q.events.h2o_table.sort:
            logger.info(f'table sort: {q.events.h2o_table.sort}')
            q.client.sort = q.events.h2o_table.sort
            q.client.page_offset = 0
        if q.events.h2o_table.filter:
            logger.info(f'table filter: {q.events.h2o_table.filter}')
            q.client.filters = q.events.h2o_table.filter
            q.client.page_offset = 0
        if q.events.h2o_table.search is not None:
            logger.info(f'table search: {q.events.h2o_table.search}')
            q.client.search = q.events.h2o_table.search
            q.client.page_offset = 0
        if q.events.h2o_table.download:
            await download_h2o_table(q)
        if q.events.h2o_table.reset:
            logger.info('table reset')
            q.client.search = None
            q.client.sort = None
            q.client.filters = None
            q.client.page_offset = 0
            q.client.total_rows = len(q.app.h2o_df)
        q.page['table_card'].h2o_table.rows = get_table_rows(q)
        q.page['table_card'].h2o_table.pagination.total_rows = q.client.total_rows
    await q.page.save()

def get_table_rows(q: Q):
    if False:
        for i in range(10):
            print('nop')
    logger.info(f'Creating new table for rows: {q.client.page_offset} to {q.client.page_offset + q.app.rows_per_page}')
    working_frame = prepare_h2o_data(q)
    local_df = working_frame[q.client.page_offset:q.client.page_offset + q.app.rows_per_page, :].as_data_frame()
    q.client.total_rows = len(working_frame)
    table_rows = [ui.table_row(name=str(q.client.page_offset + i), cells=[str(local_df[col].values[i]) for col in local_df.columns.to_list()]) for i in range(len(local_df))]
    h2o.remove(working_frame)
    return table_rows

async def download_h2o_table(q: Q):
    local_file_path = f'h2o3_data_{str(int(time()))}.csv'
    working_frame = prepare_h2o_data(q)
    h2o.download_csv(working_frame, local_file_path)
    (wave_file_path,) = await q.site.upload([local_file_path])
    os.remove(local_file_path)
    q.page['meta'].script = ui.inline_script(f'window.open("{wave_file_path}")')

def prepare_h2o_data(q: Q):
    if False:
        return 10
    working_frame = h2o.deep_copy(q.app.h2o_df, 'working_df')
    if q.client.sort is not None:
        working_frame = working_frame.sort(by=list(q.client.sort.keys()), ascending=list(q.client.sort.values()))
    if q.client.filters is not None:
        for key in q.client.filters.keys():
            working_frame = working_frame[working_frame[key].match(q.client.filters[key])]
    if q.client.search is not None:
        index = h2o.create_frame(rows=len(working_frame), cols=1, integer_fraction=1, integer_range=1)
        index['C1'] = 0
        for i in range(len(q.app.h2o_df.columns)):
            if q.app.column_searchable[i]:
                index = index + working_frame[q.app.h2o_df.columns[i]].grep(pattern=q.client.search, ignore_case=True, output_logical=True)
        working_frame = working_frame[index]
    return working_frame
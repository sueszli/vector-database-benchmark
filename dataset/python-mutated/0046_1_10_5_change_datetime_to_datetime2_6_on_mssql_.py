"""change datetime to datetime2(6) on MSSQL tables.

Revision ID: 74effc47d867
Revises: 6e96a59344a4
Create Date: 2019-08-01 15:19:57.585620

"""
from __future__ import annotations
from collections import defaultdict
from alembic import op
from sqlalchemy import text
from sqlalchemy.dialects import mssql
revision = '74effc47d867'
down_revision = '6e96a59344a4'
branch_labels = None
depends_on = None
airflow_version = '1.10.5'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Change datetime to datetime2(6) when using MSSQL as backend.'
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        result = conn.execute(text("SELECT CASE WHEN CONVERT(VARCHAR(128), SERVERPROPERTY ('productversion'))\n            like '8%' THEN '2000' WHEN CONVERT(VARCHAR(128), SERVERPROPERTY ('productversion'))\n            like '9%' THEN '2005' ELSE '2005Plus' END AS MajorVersion")).fetchone()
        mssql_version = result[0]
        if mssql_version in ('2000', '2005'):
            return
        with op.batch_alter_table('task_reschedule') as task_reschedule_batch_op:
            task_reschedule_batch_op.drop_index('idx_task_reschedule_dag_task_date')
            task_reschedule_batch_op.drop_constraint('task_reschedule_dag_task_date_fkey', type_='foreignkey')
            task_reschedule_batch_op.alter_column(column_name='execution_date', type_=mssql.DATETIME2(precision=6), nullable=False)
            task_reschedule_batch_op.alter_column(column_name='start_date', type_=mssql.DATETIME2(precision=6))
            task_reschedule_batch_op.alter_column(column_name='end_date', type_=mssql.DATETIME2(precision=6))
            task_reschedule_batch_op.alter_column(column_name='reschedule_date', type_=mssql.DATETIME2(precision=6))
        with op.batch_alter_table('task_instance') as task_instance_batch_op:
            task_instance_batch_op.drop_index('ti_state_lkp')
            task_instance_batch_op.drop_index('ti_dag_date')
            modify_execution_date_with_constraint(conn, task_instance_batch_op, 'task_instance', mssql.DATETIME2(precision=6), False)
            task_instance_batch_op.alter_column(column_name='start_date', type_=mssql.DATETIME2(precision=6))
            task_instance_batch_op.alter_column(column_name='end_date', type_=mssql.DATETIME2(precision=6))
            task_instance_batch_op.alter_column(column_name='queued_dttm', type_=mssql.DATETIME2(precision=6))
            task_instance_batch_op.create_index('ti_state_lkp', ['dag_id', 'task_id', 'execution_date'], unique=False)
            task_instance_batch_op.create_index('ti_dag_date', ['dag_id', 'execution_date'], unique=False)
        with op.batch_alter_table('task_reschedule') as task_reschedule_batch_op:
            task_reschedule_batch_op.create_foreign_key('task_reschedule_dag_task_date_fkey', 'task_instance', ['task_id', 'dag_id', 'execution_date'], ['task_id', 'dag_id', 'execution_date'], ondelete='CASCADE')
            task_reschedule_batch_op.create_index('idx_task_reschedule_dag_task_date', ['dag_id', 'task_id', 'execution_date'], unique=False)
        with op.batch_alter_table('dag_run') as dag_run_batch_op:
            modify_execution_date_with_constraint(conn, dag_run_batch_op, 'dag_run', mssql.DATETIME2(precision=6), None)
            dag_run_batch_op.alter_column(column_name='start_date', type_=mssql.DATETIME2(precision=6))
            dag_run_batch_op.alter_column(column_name='end_date', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='log', column_name='execution_date', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='log', column_name='dttm', type_=mssql.DATETIME2(precision=6))
        with op.batch_alter_table('sla_miss') as sla_miss_batch_op:
            modify_execution_date_with_constraint(conn, sla_miss_batch_op, 'sla_miss', mssql.DATETIME2(precision=6), False)
            sla_miss_batch_op.alter_column(column_name='timestamp', type_=mssql.DATETIME2(precision=6))
        op.drop_index('idx_task_fail_dag_task_date', table_name='task_fail')
        op.alter_column(table_name='task_fail', column_name='execution_date', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='task_fail', column_name='start_date', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='task_fail', column_name='end_date', type_=mssql.DATETIME2(precision=6))
        op.create_index('idx_task_fail_dag_task_date', 'task_fail', ['dag_id', 'task_id', 'execution_date'], unique=False)
        op.drop_index('idx_xcom_dag_task_date', table_name='xcom')
        op.alter_column(table_name='xcom', column_name='execution_date', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='xcom', column_name='timestamp', type_=mssql.DATETIME2(precision=6))
        op.create_index('idx_xcom_dag_task_date', 'xcom', ['dag_id', 'task_id', 'execution_date'], unique=False)
        op.alter_column(table_name='dag', column_name='last_scheduler_run', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='dag', column_name='last_pickled', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='dag', column_name='last_expired', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='dag_pickle', column_name='created_dttm', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='import_error', column_name='timestamp', type_=mssql.DATETIME2(precision=6))
        op.drop_index('job_type_heart', table_name='job')
        op.drop_index('idx_job_state_heartbeat', table_name='job')
        op.alter_column(table_name='job', column_name='start_date', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='job', column_name='end_date', type_=mssql.DATETIME2(precision=6))
        op.alter_column(table_name='job', column_name='latest_heartbeat', type_=mssql.DATETIME2(precision=6))
        op.create_index('idx_job_state_heartbeat', 'job', ['state', 'latest_heartbeat'], unique=False)
        op.create_index('job_type_heart', 'job', ['job_type', 'latest_heartbeat'], unique=False)

def downgrade():
    if False:
        while True:
            i = 10
    'Change datetime2(6) back to datetime.'
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        result = conn.execute(text("SELECT CASE WHEN CONVERT(VARCHAR(128), SERVERPROPERTY ('productversion'))\n            like '8%' THEN '2000' WHEN CONVERT(VARCHAR(128), SERVERPROPERTY ('productversion'))\n            like '9%' THEN '2005' ELSE '2005Plus' END AS MajorVersion")).fetchone()
        mssql_version = result[0]
        if mssql_version in ('2000', '2005'):
            return
        with op.batch_alter_table('task_reschedule') as task_reschedule_batch_op:
            task_reschedule_batch_op.drop_index('idx_task_reschedule_dag_task_date')
            task_reschedule_batch_op.drop_constraint('task_reschedule_dag_task_date_fkey', type_='foreignkey')
            task_reschedule_batch_op.alter_column(column_name='execution_date', type_=mssql.DATETIME, nullable=False)
            task_reschedule_batch_op.alter_column(column_name='start_date', type_=mssql.DATETIME)
            task_reschedule_batch_op.alter_column(column_name='end_date', type_=mssql.DATETIME)
            task_reschedule_batch_op.alter_column(column_name='reschedule_date', type_=mssql.DATETIME)
        with op.batch_alter_table('task_instance') as task_instance_batch_op:
            task_instance_batch_op.drop_index('ti_state_lkp')
            task_instance_batch_op.drop_index('ti_dag_date')
            modify_execution_date_with_constraint(conn, task_instance_batch_op, 'task_instance', mssql.DATETIME, False)
            task_instance_batch_op.alter_column(column_name='start_date', type_=mssql.DATETIME)
            task_instance_batch_op.alter_column(column_name='end_date', type_=mssql.DATETIME)
            task_instance_batch_op.alter_column(column_name='queued_dttm', type_=mssql.DATETIME)
            task_instance_batch_op.create_index('ti_state_lkp', ['dag_id', 'task_id', 'execution_date'], unique=False)
            task_instance_batch_op.create_index('ti_dag_date', ['dag_id', 'execution_date'], unique=False)
        with op.batch_alter_table('task_reschedule') as task_reschedule_batch_op:
            task_reschedule_batch_op.create_foreign_key('task_reschedule_dag_task_date_fkey', 'task_instance', ['task_id', 'dag_id', 'execution_date'], ['task_id', 'dag_id', 'execution_date'], ondelete='CASCADE')
            task_reschedule_batch_op.create_index('idx_task_reschedule_dag_task_date', ['dag_id', 'task_id', 'execution_date'], unique=False)
        with op.batch_alter_table('dag_run') as dag_run_batch_op:
            modify_execution_date_with_constraint(conn, dag_run_batch_op, 'dag_run', mssql.DATETIME, None)
            dag_run_batch_op.alter_column(column_name='start_date', type_=mssql.DATETIME)
            dag_run_batch_op.alter_column(column_name='end_date', type_=mssql.DATETIME)
        op.alter_column(table_name='log', column_name='execution_date', type_=mssql.DATETIME)
        op.alter_column(table_name='log', column_name='dttm', type_=mssql.DATETIME)
        with op.batch_alter_table('sla_miss') as sla_miss_batch_op:
            modify_execution_date_with_constraint(conn, sla_miss_batch_op, 'sla_miss', mssql.DATETIME, False)
            sla_miss_batch_op.alter_column(column_name='timestamp', type_=mssql.DATETIME)
        op.drop_index('idx_task_fail_dag_task_date', table_name='task_fail')
        op.alter_column(table_name='task_fail', column_name='execution_date', type_=mssql.DATETIME)
        op.alter_column(table_name='task_fail', column_name='start_date', type_=mssql.DATETIME)
        op.alter_column(table_name='task_fail', column_name='end_date', type_=mssql.DATETIME)
        op.create_index('idx_task_fail_dag_task_date', 'task_fail', ['dag_id', 'task_id', 'execution_date'], unique=False)
        op.drop_index('idx_xcom_dag_task_date', table_name='xcom')
        op.alter_column(table_name='xcom', column_name='execution_date', type_=mssql.DATETIME)
        op.alter_column(table_name='xcom', column_name='timestamp', type_=mssql.DATETIME)
        op.create_index('idx_xcom_dag_task_date', 'xcom', ['dag_id', 'task_ild', 'execution_date'], unique=False)
        op.alter_column(table_name='dag', column_name='last_scheduler_run', type_=mssql.DATETIME)
        op.alter_column(table_name='dag', column_name='last_pickled', type_=mssql.DATETIME)
        op.alter_column(table_name='dag', column_name='last_expired', type_=mssql.DATETIME)
        op.alter_column(table_name='dag_pickle', column_name='created_dttm', type_=mssql.DATETIME)
        op.alter_column(table_name='import_error', column_name='timestamp', type_=mssql.DATETIME)
        op.drop_index('job_type_heart', table_name='job')
        op.drop_index('idx_job_state_heartbeat', table_name='job')
        op.alter_column(table_name='job', column_name='start_date', type_=mssql.DATETIME)
        op.alter_column(table_name='job', column_name='end_date', type_=mssql.DATETIME)
        op.alter_column(table_name='job', column_name='latest_heartbeat', type_=mssql.DATETIME)
        op.create_index('idx_job_state_heartbeat', 'job', ['state', 'latest_heartbeat'], unique=False)
        op.create_index('job_type_heart', 'job', ['job_type', 'latest_heartbeat'], unique=False)

def get_table_constraints(conn, table_name) -> dict[tuple[str, str], list[str]]:
    if False:
        i = 10
        return i + 15
    'Return primary and unique constraint along with column name.\n\n    This function return primary and unique constraint\n    along with column name. some tables like task_instance\n    is missing primary key constraint name and the name is\n    auto-generated by sql server. so this function helps to\n    retrieve any primary or unique constraint name.\n\n    :param conn: sql connection object\n    :param table_name: table name\n    :return: a dictionary of ((constraint name, constraint type), column name) of table\n    '
    query = text(f"SELECT tc.CONSTRAINT_NAME , tc.CONSTRAINT_TYPE, ccu.COLUMN_NAME\n     FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc\n     JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE AS ccu ON ccu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME\n     WHERE tc.TABLE_NAME = '{table_name}' AND\n     (tc.CONSTRAINT_TYPE = 'PRIMARY KEY' or UPPER(tc.CONSTRAINT_TYPE) = 'UNIQUE')\n    ")
    result = conn.execute(query).fetchall()
    constraint_dict = defaultdict(list)
    for (constraint, constraint_type, column) in result:
        constraint_dict[constraint, constraint_type].append(column)
    return constraint_dict

def reorder_columns(columns):
    if False:
        return 10
    "Reorder the columns for creating constraint.\n    Preserve primary key ordering\n    ``['task_id', 'dag_id', 'execution_date']``\n\n    :param columns: columns retrieved from DB related to constraint\n    :return: ordered column\n    "
    ordered_columns = []
    for column in ['task_id', 'dag_id', 'execution_date']:
        if column in columns:
            ordered_columns.append(column)
    for column in columns:
        if column not in ['task_id', 'dag_id', 'execution_date']:
            ordered_columns.append(column)
    return ordered_columns

def drop_constraint(operator, constraint_dict):
    if False:
        while True:
            i = 10
    'Drop a primary key or unique constraint.\n\n    :param operator: batch_alter_table for the table\n    :param constraint_dict: a dictionary of ((constraint name, constraint type), column name) of table\n    '
    for (constraint, columns) in constraint_dict.items():
        if 'execution_date' in columns:
            if constraint[1].lower().startswith('primary'):
                operator.drop_constraint(constraint[0], type_='primary')
            elif constraint[1].lower().startswith('unique'):
                operator.drop_constraint(constraint[0], type_='unique')

def create_constraint(operator, constraint_dict):
    if False:
        while True:
            i = 10
    'Create a primary key or unique constraint.\n\n    :param operator: batch_alter_table for the table\n    :param constraint_dict: a dictionary of ((constraint name, constraint type), column name) of table\n    '
    for (constraint, columns) in constraint_dict.items():
        if 'execution_date' in columns:
            if constraint[1].lower().startswith('primary'):
                operator.create_primary_key(constraint_name=constraint[0], columns=reorder_columns(columns))
            elif constraint[1].lower().startswith('unique'):
                operator.create_unique_constraint(constraint_name=constraint[0], columns=reorder_columns(columns))

def modify_execution_date_with_constraint(conn, batch_operator, table_name, type_, nullable) -> None:
    if False:
        print('Hello World!')
    'Change type of column execution_date.\n    Helper function changes type of column execution_date by\n    dropping and recreating any primary/unique constraint associated with\n    the column\n\n    :param conn: sql connection object\n    :param batch_operator: batch_alter_table for the table\n    :param table_name: table name\n    :param type_: DB column type\n    :param nullable: nullable (boolean)\n    :return: a dictionary of ((constraint name, constraint type), column name) of table\n    '
    constraint_dict = get_table_constraints(conn, table_name)
    drop_constraint(batch_operator, constraint_dict)
    batch_operator.alter_column(column_name='execution_date', type_=type_, nullable=nullable)
    create_constraint(batch_operator, constraint_dict)
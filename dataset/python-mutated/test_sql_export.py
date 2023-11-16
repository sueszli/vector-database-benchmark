from .context import lux
import pytest
import pandas as pd
import numpy as np
import psycopg2
from lux.vis.Vis import Vis
from lux.executor.SQLExecutor import SQLExecutor

def test_scatter_code_export(global_var):
    if False:
        print('Hello World!')
    tbl = lux.LuxSQLTable()
    tbl.set_SQL_table('cars')
    vis = Vis([lux.Clause('horsepower'), lux.Clause('acceleration')], tbl)
    SQLExecutor.execute([vis], tbl)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(tbl, vis)
    except:
        assert False

def test_color_scatter_code_export(global_var):
    if False:
        for i in range(10):
            print('nop')
    tbl = lux.LuxSQLTable()
    tbl.set_SQL_table('cars')
    vis = Vis([lux.Clause('horsepower'), lux.Clause('acceleration'), lux.Clause('origin')], tbl)
    SQLExecutor.execute([vis], tbl)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(tbl, vis)
    except:
        assert False

def test_histogram_code_export(global_var):
    if False:
        i = 10
        return i + 15
    tbl = lux.LuxSQLTable()
    tbl.set_SQL_table('cars')
    vis = Vis([lux.Clause('horsepower')], tbl)
    SQLExecutor.execute([vis], tbl)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(tbl, vis)
    except:
        assert False

def test_barchart_code_export(global_var):
    if False:
        while True:
            i = 10
    tbl = lux.LuxSQLTable()
    tbl.set_SQL_table('cars')
    vis = Vis([lux.Clause('origin')], tbl)
    SQLExecutor.execute([vis], tbl)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(tbl, vis)
    except:
        assert False

def test_color_barchart_code_export(global_var):
    if False:
        for i in range(10):
            print('nop')
    tbl = lux.LuxSQLTable()
    tbl.set_SQL_table('cars')
    vis = Vis([lux.Clause('origin'), lux.Clause('cylinders')], tbl)
    SQLExecutor.execute([vis], tbl)
    code = vis.to_code('python')
    try:
        exec(code, globals())
        create_chart_data(tbl, vis)
    except:
        assert False
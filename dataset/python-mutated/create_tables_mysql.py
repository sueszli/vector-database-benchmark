import sys, os
import csv
import mysql.connector
import traceback

class CreateMysqlTables:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self = self

    def drop_testcases_table(self):
        if False:
            print('Hello World!')
        h2o = mysql.connector.connect(user='root', password='0xdata', host='172.16.2.178', database='h2o')
        cursor = h2o.cursor()
        try:
            drop_testcases_query = '\n                         DROP TABLE IF EXISTS TestCases;\n                         '
            cursor.execute(drop_testcases_query)
        except:
            traceback.print_exc()
            h2o.rollback()
            assert False, 'Failed to drop TestCases table!'
        cursor.close()
        h2o.close()

    def drop_acc_datasets_tables(self):
        if False:
            for i in range(10):
                print('nop')
        h2o = mysql.connector.connect(user='root', password='0xdata', host='172.16.2.178', database='h2o')
        cursor = h2o.cursor()
        try:
            drop_accuracydata_query = '\n                        DROP TABLES IF EXISTS AccuracyDatasets;\n                        '
            cursor.execute(drop_accuracydata_query)
        except:
            traceback.print_exc()
            h2o.rollback()
            assert False, 'Failed to drop AccuracyDatasets table!'
        cursor.close()
        h2o.close()

    def create_testcases_table(self):
        if False:
            print('Hello World!')
        h2o = mysql.connector.connect(user='root', password='0xdata', host='172.16.2.178', database='h2o')
        cursor = h2o.cursor()
        self.drop_testcases_table()
        try:
            test_cases_query = '\n                                CREATE TABLE TestCases(\n                                test_case_id int(100) NOT NULL AUTO_INCREMENT,\n                                algorithm varchar(100) NOT NULL,\n                                algo_parameters varchar(200) NOT NULL,\n                                tuned int(100) NOT NULL,\n                                regression int(100) NOT NULL,\n                                training_data_set_id int(100) NOT NULL,\n                                testing_data_set_id int(100) NOT NULL,\n                                PRIMARY KEY (`test_case_id`)\n                                )'
            cursor.execute(test_cases_query)
        except:
            traceback.print_exc()
            h2o.rollback()
            assert False, 'Failed to build TestCases table for h2o database!'
        cursor.close()
        h2o.close()

    def create_accuracy_datasets(self):
        if False:
            while True:
                i = 10
        h2o = mysql.connector.connect(user='root', password='0xdata', host='172.16.2.178', database='h2o')
        cursor = h2o.cursor()
        self.drop_acc_datasets_tables()
        try:
            acc_data_query = '\n                                CREATE TABLE IF NOT EXISTS AccuracyDatasets(\n                                data_set_id int(100) NOT NULL AUTO_INCREMENT,\n                                uri varchar(100) NOT NULL,\n                                respose_col_idx int(100) NOT NULL,\n                                PRIMARY KEY (`data_set_id`)\n                                )'
            cursor.execute(acc_data_query)
        except:
            traceback.print_exc()
            h2o.rollback()
            assert False, 'Failed to build AccuracyDatasets table for h2o database!'
        cursor.close()
        h2o.close()
if __name__ == '__main__':
    CreateMysqlTables().create_testcases_table()
    CreateMysqlTables().create_accuracy_datasets()
import sys, os
import csv
import mysql.connector
from mysql.connector.constants import ClientFlag
import traceback

class SendDataToMysql:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self = self

    def add_test_cases_to_h2o(self):
        if False:
            for i in range(10):
                print('nop')
        h2o = mysql.connector.connect(client_flags=[ClientFlag.LOCAL_FILES], user='root', password='0xdata', host='172.16.2.178', database='h2o')
        cursor = h2o.cursor()
        try:
            cursor.execute("LOAD DATA LOCAL INFILE '../h2o-test-accuracy/src/test/resources/accuracyTestCases.csv' INTO TABLE TestCases COLUMNS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES;")
            h2o.commit()
        except:
            traceback.print_exc()
            h2o.rollback()
            assert False, 'Failed to add accuracy test cases to h2o database!'

    def add_accuracy_data(self):
        if False:
            while True:
                i = 10
        h2o = mysql.connector.connect(client_flags=[ClientFlag.LOCAL_FILES], user='root', password='0xdata', host='172.16.2.178', database='h2o')
        cursor = h2o.cursor()
        try:
            cursor.execute("LOAD DATA LOCAL INFILE '../h2o-test-accuracy/src/test/resources/accuracyDataSets.csv' INTO TABLE AccuracyDatasets COLUMNS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES;")
            h2o.commit()
        except:
            traceback.print_exc()
            h2o.rollback()
            assert False, 'Failed to add accuracy test cases to h2o database!'

    def drop_join_test_cases_tables(self):
        if False:
            return 10
        h2o = mysql.connector.connect(user='root', password='0xdata', host='172.16.2.178', database='h2o')
        cursor = h2o.cursor()
        try:
            drop_join_test_cases_query = '\n                        DROP TABLES IF EXISTS TestCasesResults;\n                        '
            cursor.execute(drop_join_test_cases_query)
        except:
            traceback.print_exc()
            h2o.rollback()
            assert False, 'Failed to drop TestCasesResults table!'

    def join_test_cases_results(self):
        if False:
            while True:
                i = 10
        h2o = mysql.connector.connect(client_flags=[ClientFlag.LOCAL_FILES], user='root', password='0xdata', host='172.16.2.178', database='h2o')
        cursor = h2o.cursor()
        self.drop_join_test_cases_tables()
        try:
            join_query = '\n                         CREATE TABLE TestCasesResults AS(\n                         SELECT *\n                         FROM AccuracyTestCaseResults\n                         LEFT JOIN TestCases\n                         ON AccuracyTestCaseResults.testcase_id = TestCases.test_case_id\n                         LEFT JOIN AccuracyDatasets\n                         ON TestCases.training_data_set_id = AccuracyDatasets.data_set_id);\n                         '
            cursor.execute(join_query)
        except:
            traceback.print_exc()
            h2o.rollback()
            assert False, 'Failed to join AccuracyTestCaseResults, TestCases, and AccuracyDatasets!'
        cursor.close()
        h2o.close()
if __name__ == '__main__':
    SendDataToMysql().join_test_cases_results()
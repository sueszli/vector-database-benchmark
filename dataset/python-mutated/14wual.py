import mysql.connector
import datetime

class Credentials:

    def __init__(self) -> None:
        if False:
            return 10
        self.host = 'mysql-5707.dinaserver.com'
        self.port = 3306
        self.user = 'mouredev_read'
        self.password = 'mouredev_pass'
        self.database = 'moure_test'

class Info:

    def about():
        if False:
            while True:
                i = 10
        _about = '# Code By 14Wual\n# Challenge #23: The database\n# Challenge of Mouredev\n# Link: https://github.com/mouredev/retos-programacion-2023\n'
        print(_about)

    def return_str_current_datetime():
        if False:
            i = 10
            return i + 15
        current_datetime = datetime.datetime.now()
        return str(current_datetime.strftime('%Y-%m-%d %H:%M:%S'))

class Challeng:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        Info.about()
        self.ddbb_credentials = Credentials()
        self.connection = self.create_connection()
        if self.connection:
            print(f'[✓] - {Info.return_str_current_datetime()} Established connection.')
            print(f'[Info] - {Info.return_str_current_datetime()} Running query.')
            try:
                cursor = self.connection.cursor()
                cursor.execute('SELECT * FROM challenges')
                output = cursor.fetchall()
                print(f'[✓] - {Info.return_str_current_datetime()} Query executed successfully.')
                print(f'[Info] - {Info.return_str_current_datetime()} Closing connection.')
                try:
                    cursor.close()
                    self.connection.close()
                finally:
                    print(f'[✓] - {Info.return_str_current_datetime()} Connection closed correctly.')
                print(f'[Info] - {Info.return_str_current_datetime()} Showing results.')
                Challeng.beautifull_print_output(output)
            except mysql.connector.Error as error:
                Exceptions(expecific='Error executing query', error=error)
        else:
            print('[x] The connection could not be completed.')

    def create_connection(self):
        if False:
            return 10
        print(f'[Info] - {Info.return_str_current_datetime()} Connecting to the database.')
        try:
            return mysql.connector.connect(host=self.ddbb_credentials.host, port=self.ddbb_credentials.port, user=self.ddbb_credentials.user, password=self.ddbb_credentials.password, database=self.ddbb_credentials.database)
        except mysql.connector.errors.InterfaceError as error:
            Exceptions(expecific='InterfaceError', error=error)
        except mysql.connector.Error as error:
            Exceptions(expecific='Other', error=error)
        return None

    def beautifull_print_output(output):
        if False:
            while True:
                i = 10
        print('\n----- Challenges -----')
        for row in output:
            print('ID:', row[0])
            print('Title:', row[1])
            print('Difficulty:', row[2])
            print('Date:', row[3])
            print('----------------------')

class Exceptions:

    def __init__(self, expecific, error) -> None:
        if False:
            print('Hello World!')
        print(f'[x] - {Info.return_str_current_datetime()} Type of error: ', expecific, f'\n[x] - {Info.return_str_current_datetime()} Error: ', error)
if __name__ == '__main__':
    Challeng()
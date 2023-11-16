import os
from flask import Flask
import sqlalchemy
db_user = os.environ.get('CLOUD_SQL_USERNAME')
db_password = os.environ.get('CLOUD_SQL_PASSWORD')
db_name = os.environ.get('CLOUD_SQL_DATABASE_NAME')
db_connection_name = os.environ.get('CLOUD_SQL_CONNECTION_NAME')
if os.environ.get('GAE_ENV') == 'standard':
    unix_socket = f'/cloudsql/{db_connection_name}'
    engine_url = 'mysql+pymysql://{}:{}@/{}?unix_socket={}'.format(db_user, db_password, db_name, unix_socket)
else:
    host = '127.0.0.1'
    engine_url = 'mysql+pymysql://{}:{}@{}/{}'.format(db_user, db_password, host, db_name)
engine = sqlalchemy.create_engine(engine_url, pool_size=3)
app = Flask(__name__)

@app.route('/')
def main():
    if False:
        for i in range(10):
            print('nop')
    cnx = engine.connect()
    cursor = cnx.execute('SELECT NOW() as now;')
    result = cursor.fetchall()
    current_time = result[0][0]
    cnx.close()
    return str(current_time)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
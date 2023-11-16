import os
import click
import fnmatch
import sys
for dirs in os.listdir('/ppml/trusted-big-data-ml/work'):
    if fnmatch.fnmatch(dirs, 'bigdl-*'):
        path = '/ppml/trusted-big-data-ml/work/' + dirs + '/python/'
        for files in os.listdir(path):
            if fnmatch.fnmatch(files, 'bigdl-ppml-*-python-api.zip'):
                sys.path.append(path + files)
                sys.path.append(path + files + '/bigdl/ppml/fl/nn/generated')
from bigdl.ppml.fl.fl_server import FLServer

@click.command()
@click.option('--client_num', default=2)
@click.option('--port', default=8980)
@click.option('--servermodelpath', default='/tmp/fgboost_server_model')
def run(port, client_num, servermodelpath):
    if False:
        print('Hello World!')
    conf = open('ppml-conf.yaml', 'w')
    conf.write('serverPort: ' + str(port) + '\n')
    conf.write('clientNum: ' + str(client_num) + '\n')
    conf.write('fgBoostServerModelPath: ' + servermodelpath + '\n')
    conf.close()
    fl_server = FLServer()
    fl_server.build()
    fl_server.start()
    fl_server.block_until_shutdown()
if __name__ == '__main__':
    run()
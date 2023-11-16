"""Binary for showing C++ backward compatibility.

This creates a SavedModel using the "old" op and C++ kernel from multiplex_2.

https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/saved_model/save
"""
import os
import shutil
from absl import app
from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op
from tensorflow.examples.custom_ops_doc.multiplex_4 import model_using_multiplex

def main(argv):
    if False:
        i = 10
        return i + 15
    del argv
    path = 'model_using_multiplex'
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    model_using_multiplex.save(multiplex_2_op.multiplex, path)
    print('Saved model to', path)
if __name__ == '__main__':
    app.run(main)
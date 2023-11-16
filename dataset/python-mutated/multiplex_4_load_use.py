"""Binary for showing C++ backward compatibility.

This loads a previously created SavedModel (esp. a model created by
multiplex_2_save.py which uses the "old" op and C++ kernel from multiplex_2)
and runs the model using the "new" multiplex_4 C++ kernel.

https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/saved_model/save
"""
from absl import app
from tensorflow.examples.custom_ops_doc.multiplex_4 import model_using_multiplex

def main(argv):
    if False:
        return 10
    del argv
    path = 'model_using_multiplex'
    result = model_using_multiplex.load_and_use(path)
    print('Result:', result)
if __name__ == '__main__':
    app.run(main)
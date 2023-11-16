import pandas as pd
import autokeras as ak
from autokeras import test_utils

def test_io_api(tmp_path):
    if False:
        i = 10
        return i + 15
    num_instances = 20
    image_x = test_utils.generate_data(num_instances=num_instances, shape=(28, 28))
    text_x = test_utils.generate_text_data(num_instances=num_instances)
    image_x = image_x[:num_instances]
    structured_data_x = pd.read_csv(test_utils.TRAIN_CSV_PATH).to_numpy().astype(str)[:num_instances]
    classification_y = test_utils.generate_one_hot_labels(num_instances=num_instances, num_classes=3)
    regression_y = test_utils.generate_data(num_instances=num_instances, shape=(1,))
    automodel = ak.AutoModel(inputs=[ak.ImageInput(), ak.TextInput(), ak.StructuredDataInput()], outputs=[ak.RegressionHead(metrics=['mae']), ak.ClassificationHead(loss='categorical_crossentropy', metrics=['accuracy'])], directory=tmp_path, max_trials=2, tuner=ak.RandomSearch, seed=test_utils.SEED)
    automodel.fit([image_x, text_x, structured_data_x], [regression_y, classification_y], epochs=1, validation_split=0.2, batch_size=4)
    automodel.predict([image_x, text_x, structured_data_x])
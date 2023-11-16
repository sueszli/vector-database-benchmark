from deepchecks.tabular import Context

def test_task_type_same_with_model_or_y_pred(diabetes_split_dataset_and_model):
    if False:
        for i in range(10):
            print('nop')
    (train, _, model) = diabetes_split_dataset_and_model
    ctx1 = Context(train, model=model)
    ctx2 = Context(train, y_pred_train=model.predict(train.features_columns))
    assert ctx1.task_type == ctx2.task_type
def update_param(name, param):
    if False:
        i = 10
        return i + 15
    if name == 'stopping_metric':
        param['values'] = ['AUTO', 'anomaly_score']
        return param
    return None
extensions = dict(required_params=['training_frame', 'x'], validate_required_params='', set_required_params='\nparms$training_frame <- training_frame\nif(!missing(x))\n  parms$ignored_columns <- .verify_datacols(training_frame, x)$cols_ignore\n', skip_default_set_params_for=['training_frame', 'ignored_columns'])
doc = dict(preamble='\nTrains an Isolation Forest model\n', params=dict(x='A vector containing the \\code{character} names of the predictors in the model.'), examples='\nlibrary(h2o)\nh2o.init()\n\n# Import the cars dataset\nf <- "https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv"\ncars <- h2o.importFile(f)\n\n# Set the predictors\npredictors <- c("displacement", "power", "weight", "acceleration", "year")\n\n# Train the IF model\ncars_if <- h2o.isolationForest(x = predictors, training_frame = cars,\n                               seed = 1234, stopping_metric = "anomaly_score",\n                               stopping_rounds = 3, stopping_tolerance = 0.1)\n')
def update_param(name, param):
    if False:
        i = 10
        return i + 15
    if name == 'distribution':
        values = param['values']
        param['values'] = [v for v in values if v not in ['custom', 'ordinal', 'quasibinomial']]
        return param
    elif name == 'stopping_metric':
        param['values'].remove('anomaly_score')
        return param
    elif name in ['ignored_columns', 'response_column', 'max_confusion_matrix_size']:
        return {}
    return None
extensions = dict(required_params=['x', 'y', 'training_frame'], frame_params=['training_frame', 'validation_frame'], validate_required_params='\n# If x is missing, then assume user wants to use all columns as features.\nif (missing(x)) {\n   if (is.numeric(y)) {\n       x <- setdiff(col(training_frame), y)\n   } else {\n       x <- setdiff(colnames(training_frame), y)\n   }\n}\n', set_required_params='\nparms$training_frame <- training_frame\nargs <- .verify_dataxy(training_frame, x, y)\nif( !missing(offset_column) && !is.null(offset_column))  args$x_ignore <- args$x_ignore[!( offset_column == args$x_ignore )]\nif( !missing(weights_column) && !is.null(weights_column)) args$x_ignore <- args$x_ignore[!( weights_column == args$x_ignore )]\nif( !missing(fold_column) && !is.null(fold_column)) args$x_ignore <- args$x_ignore[!( fold_column == args$x_ignore )]\nparms$ignored_columns <- args$x_ignore\nparms$response_column <- args$y\n', skip_default_set_params_for=['training_frame', 'ignored_columns', 'response_column', 'max_confusion_matrix_size'])
doc = dict(params=dict(x='\n(Optional) A vector containing the names or indices of the predictor variables to use in building the model.\nIf x is missing, then all columns except y are used.\n', y='\nThe name or column index of the response variable in the data. \nThe response must be either a numeric or a categorical/factor variable. \nIf the response is numeric, then a regression model will be trained, otherwise it will train a classification model.\n', seed='\nSeed for random numbers (affects certain parts of the algo that are stochastic and those might or might not be enabled by default).\nDefaults to -1 (time-based random number).\n', ignored_columns=None, response_column=None, max_confusion_matrix_size=None))
from copy import copy

def update_param(name, param):
    if False:
        for i in range(10):
            print('nop')
    if name == 'min_sdev':
        threshold = copy(param)
        threshold['name'] = 'threshold'
        return [threshold, param]
    if name == 'eps_sdev':
        eps = copy(param)
        eps['name'] = 'eps'
        return [eps, param]
    return None
extensions = dict(validate_params='\n.naivebayes.map <- c("x" = "ignored_columns", "y" = "response_column", \n                     "threshold" = "min_sdev", "eps" = "eps_sdev")\n', set_required_params='\nparms$training_frame <- training_frame\nargs <- .verify_dataxy(training_frame, x, y)\nif( !missing(fold_column) && !is.null(fold_column)) args$x_ignore <- args$x_ignore[!( fold_column == args$x_ignore )]\nparms$ignored_columns <- args$x_ignore\nparms$response_column <- args$y\n', skip_default_set_params_for=['training_frame', 'ignored_columns', 'response_column', 'max_confusion_matrix_size', 'threshold', 'eps'], set_params='\nif (!missing(threshold) && missing(min_sdev)) {\n  warning("argument \'threshold\' is deprecated; use \'min_sdev\' instead.")\n  parms$min_sdev <- threshold\n}\nif (!missing(eps) && missing(eps_sdev)) {\n  warning("argument \'eps\' is deprecated; use \'eps_sdev\' instead.")\n  parms$eps_sdev <- eps\n}\n')
doc = dict(preamble='\nCompute naive Bayes probabilities on an H2O dataset.\n\nThe naive Bayes classifier assumes independence between predictor variables conditional\non the response, and a Gaussian distribution of numeric predictors with mean and standard\ndeviation computed from the training dataset. When building a naive Bayes classifier,\nevery row in the training dataset that contains at least one NA will be skipped completely.\nIf the test dataset has missing values, then those predictors are omitted in the probability\ncalculation during prediction.\n', params=dict(threshold='\nThis argument is deprecated, use `min_sdev` instead. The minimum standard deviation to use for observations without enough data.\nMust be at least 1e-10.\n', min_sdev='\nThe minimum standard deviation to use for observations without enough data.\nMust be at least 1e-10.\n', eps='\nThis argument is deprecated, use `eps_sdev` instead. A threshold cutoff to deal with numeric instability, must be positive.\n', eps_sdev='\nA threshold cutoff to deal with numeric instability, must be positive.\n', min_prob='\nMin. probability to use for observations with not enough data.\n', eps_prob='\nCutoff below which probability is replaced with min_prob.\n'), returns='\nan object of class \\linkS4class{H2OBinomialModel} if the response has two categorical levels,\nand \\linkS4class{H2OMultinomialModel} otherwise.\n', examples='\nh2o.init()\nvotes_path <- system.file("extdata", "housevotes.csv", package = "h2o")\nvotes <- h2o.uploadFile(path = votes_path, header = TRUE)\nh2o.naiveBayes(x = 2:17, y = 1, training_frame = votes, laplace = 3)\n')
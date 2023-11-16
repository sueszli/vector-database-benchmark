def update_param(name, param):
    if False:
        return 10
    if name == 'distribution':
        param['values'].remove('ordinal')
        return param
    return None
extensions = dict(extra_params=[('verbose', 'FALSE')], validate_params='\n# Required maps for different names params, including deprecated params\n.gbm.map <- c("x" = "ignored_columns",\n              "y" = "response_column")\n')
doc = dict(preamble='\nBuild gradient boosted classification or regression trees\n\nBuilds gradient boosted classification trees and gradient boosted regression trees on a parsed data set.\nThe default distribution function will guess the model type based on the response column type.\nIn order to run properly, the response column must be an numeric for "gaussian" or an\nenum for "bernoulli" or "multinomial".\n', params=dict(verbose='\n\\code{Logical}. Print scoring history to the console (Metrics per tree). Defaults to FALSE.\n'), seealso='\n\\code{\\link{predict.H2OModel}} for prediction\n', examples='\nlibrary(h2o)\nh2o.init()\n\n# Run regression GBM on australia data\naustralia_path <- system.file("extdata", "australia.csv", package = "h2o")\naustralia <- h2o.uploadFile(path = australia_path)\nindependent <- c("premax", "salmax", "minairtemp", "maxairtemp", "maxsst",\n                 "maxsoilmoist", "Max_czcs")\ndependent <- "runoffnew"\nh2o.gbm(y = dependent, x = independent, training_frame = australia,\n        ntrees = 3, max_depth = 3, min_rows = 2)\n')
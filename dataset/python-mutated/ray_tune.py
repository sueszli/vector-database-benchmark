from ray import train, tune

def objective(step, alpha, beta):
    if False:
        i = 10
        return i + 15
    return (0.1 + alpha * step / 100) ** (-1) + beta * 0.1

def training_function(config):
    if False:
        for i in range(10):
            print('nop')
    (alpha, beta) = (config['alpha'], config['beta'])
    for step in range(10):
        intermediate_score = objective(step, alpha, beta)
        train.report({'mean_loss': intermediate_score})
tuner = tune.Tuner(training_function, param_space={'alpha': tune.grid_search([0.001, 0.01, 0.1]), 'beta': tune.choice([1, 2, 3])})
results = tuner.fit()
best_result = results.get_best_result(metric='mean_loss', mode='min')
print('Best result: ', best_result.metrics)
df = results.get_dataframe()
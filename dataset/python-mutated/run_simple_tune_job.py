import ray
from ray import train, tune

def objective(step, alpha, beta):
    if False:
        while True:
            i = 10
    return (0.1 + alpha * step / 100) ** (-1) + beta * 0.1

def training_function(config):
    if False:
        for i in range(10):
            print('nop')
    (alpha, beta) = (config['alpha'], config['beta'])
    for step in range(10):
        intermediate_score = objective(step, alpha, beta)
        train.report(dict(mean_loss=intermediate_score))
ray.init(address='auto')
print('Starting Ray Tune job')
analysis = tune.run(training_function, config={'alpha': tune.grid_search([0.001, 0.01, 0.1]), 'beta': tune.choice([1, 2, 3])})
print('Best config: ', analysis.get_best_config(metric='mean_loss', mode='min'))
df = analysis.results_df
print(df)
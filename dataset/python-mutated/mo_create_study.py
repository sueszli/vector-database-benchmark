import json
import sys
from kurobako import solver
from kurobako.solver.optuna import OptunaSolverFactory
import optuna
optuna.logging.disable_default_handler()

def create_study(seed: int) -> optuna.Study:
    if False:
        i = 10
        return i + 15
    seed
    n_objectives = 2
    directions = ['minimize'] * n_objectives
    sampler_name = sys.argv[1]
    sampler_cls = getattr(optuna.samplers, sampler_name, getattr(optuna.integration, sampler_name, None))
    if sampler_cls is None:
        raise ValueError('Unknown sampler: {}.'.format(sampler_name))
    sampler_kwargs = json.loads(sys.argv[2])
    sampler = sampler_cls(**sampler_kwargs)
    return optuna.create_study(directions=directions, sampler=sampler, pruner=optuna.pruners.NopPruner())
if __name__ == '__main__':
    factory = OptunaSolverFactory(create_study)
    runner = solver.SolverRunner(factory)
    runner.run()
"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np
from deap import tools, gp
from inspect import isclass
from .operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from collections import defaultdict
import warnings
from stopit import threading_timeoutable, TimeoutException

def pick_two_individuals_eligible_for_crossover(population):
    if False:
        for i in range(10):
            print('nop')
    'Pick two individuals from the population which can do crossover, that is, they share a primitive.\n\n    Parameters\n    ----------\n    population: array of individuals\n\n    Returns\n    ----------\n    tuple: (individual, individual)\n        Two individuals which are not the same, but share at least one primitive.\n        Alternatively, if no such pair exists in the population, (None, None) is returned instead.\n    '
    primitives_by_ind = [set([node.name for node in ind if isinstance(node, gp.Primitive)]) for ind in population]
    pop_as_str = [str(ind) for ind in population]
    eligible_pairs = [(i, i + 1 + j) for (i, ind1_prims) in enumerate(primitives_by_ind) for (j, ind2_prims) in enumerate(primitives_by_ind[i + 1:]) if not ind1_prims.isdisjoint(ind2_prims) and pop_as_str[i] != pop_as_str[i + 1 + j]]
    eligible_pairs += [(j, i) for (i, j) in eligible_pairs]
    if not eligible_pairs:
        return (None, None)
    pair = np.random.randint(0, len(eligible_pairs))
    (idx1, idx2) = eligible_pairs[pair]
    return (population[idx1], population[idx2])

def mutate_random_individual(population, toolbox):
    if False:
        while True:
            i = 10
    'Picks a random individual from the population, and performs mutation on a copy of it.\n\n    Parameters\n    ----------\n    population: array of individuals\n\n    Returns\n    ----------\n    individual: individual\n        An individual which is a mutated copy of one of the individuals in population,\n        the returned individual does not have fitness.values\n    '
    idx = np.random.randint(0, len(population))
    ind = population[idx]
    (ind,) = toolbox.mutate(ind)
    del ind.fitness.values
    return ind

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    if False:
        return 10
    'Part of an evolutionary algorithm applying only the variation part\n    (crossover, mutation **or** reproduction). The modified individuals have\n    their fitness invalidated. The individuals are cloned so returned\n    population is independent of the input population.\n    :param population: A list of individuals to vary.\n    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution\n                    operators.\n    :param lambda\\_: The number of children to produce\n    :param cxpb: The probability of mating two individuals.\n    :param mutpb: The probability of mutating an individual.\n    :returns: The final population\n    :returns: A class:`~deap.tools.Logbook` with the statistics of the\n              evolution\n    The variation goes as follow. On each of the *lambda_* iteration, it\n    selects one of the three operations; crossover, mutation or reproduction.\n    In the case of a crossover, two individuals are selected at random from\n    the parental population :math:`P_\\mathrm{p}`, those individuals are cloned\n    using the :meth:`toolbox.clone` method and then mated using the\n    :meth:`toolbox.mate` method. Only the first child is appended to the\n    offspring population :math:`P_\\mathrm{o}`, the second child is discarded.\n    In the case of a mutation, one individual is selected at random from\n    :math:`P_\\mathrm{p}`, it is cloned and then mutated using using the\n    :meth:`toolbox.mutate` method. The resulting mutant is appended to\n    :math:`P_\\mathrm{o}`. In the case of a reproduction, one individual is\n    selected at random from :math:`P_\\mathrm{p}`, cloned and appended to\n    :math:`P_\\mathrm{o}`.\n    This variation is named *Or* beceause an offspring will never result from\n    both operations crossover and mutation. The sum of both probabilities\n    shall be in :math:`[0, 1]`, the reproduction probability is\n    1 - *cxpb* - *mutpb*.\n    '
    offspring = []
    for _ in range(lambda_):
        op_choice = np.random.random()
        if op_choice < cxpb:
            (ind1, ind2) = pick_two_individuals_eligible_for_crossover(population)
            if ind1 is not None:
                (ind1_cx, _, evaluated_individuals_) = toolbox.mate(ind1, ind2)
                del ind1_cx.fitness.values
                if str(ind1_cx) in evaluated_individuals_:
                    ind1_cx = mutate_random_individual(population, toolbox)
                offspring.append(ind1_cx)
            else:
                ind_mu = mutate_random_individual(population, toolbox)
                offspring.append(ind_mu)
        elif op_choice < cxpb + mutpb:
            ind = mutate_random_individual(population, toolbox)
            offspring.append(ind)
        else:
            idx = np.random.randint(0, len(population))
            offspring.append(toolbox.clone(population[idx]))
    return offspring

def initialize_stats_dict(individual):
    if False:
        return 10
    "\n    Initializes the stats dict for individual\n    The statistics initialized are:\n        'generation': generation in which the individual was evaluated. Initialized as: 0\n        'mutation_count': number of mutation operations applied to the individual and its predecessor cumulatively. Initialized as: 0\n        'crossover_count': number of crossover operations applied to the individual and its predecessor cumulatively. Initialized as: 0\n        'predecessor': string representation of the individual. Initialized as: ('ROOT',)\n\n    Parameters\n    ----------\n    individual: deap individual\n\n    Returns\n    -------\n    object\n    "
    individual.statistics['generation'] = 0
    individual.statistics['mutation_count'] = 0
    individual.statistics['crossover_count'] = 0
    individual.statistics['predecessor'] = ('ROOT',)

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, pbar, stats=None, halloffame=None, verbose=0, per_generation_function=None, log_file=None):
    if False:
        return 10
    'This is the :math:`(\\mu + \\lambda)` evolutionary algorithm.\n    :param population: A list of individuals.\n    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution\n                    operators.\n    :param mu: The number of individuals to select for the next generation.\n    :param lambda\\_: The number of children to produce at each generation.\n    :param cxpb: The probability that an offspring is produced by crossover.\n    :param mutpb: The probability that an offspring is produced by mutation.\n    :param ngen: The number of generation.\n    :param pbar: processing bar\n    :param stats: A :class:`~deap.tools.Statistics` object that is updated\n                  inplace, optional.\n    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will\n                       contain the best individuals, optional.\n    :param verbose: Whether or not to log the statistics.\n    :param per_generation_function: if supplied, call this function before each generation\n                            used by tpot to save best pipeline before each new generation\n    :param log_file: io.TextIOWrapper or io.StringIO, optional (defaul: sys.stdout)\n    :returns: The final population\n    :returns: A class:`~deap.tools.Logbook` with the statistics of the\n              evolution.\n    The algorithm takes in a population and evolves it in place using the\n    :func:`varOr` function. It returns the optimized population and a\n    :class:`~deap.tools.Logbook` with the statistics of the evolution. The\n    logbook will contain the generation number, the number of evalutions for\n    each generation and the statistics if a :class:`~deap.tools.Statistics` is\n    given as argument. The *cxpb* and *mutpb* arguments are passed to the\n    :func:`varOr` function. The pseudocode goes as follow ::\n        evaluate(population)\n        for g in range(ngen):\n            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)\n            evaluate(offspring)\n            population = select(population + offspring, mu)\n    First, the individuals having an invalid fitness are evaluated. Second,\n    the evolutionary loop begins by producing *lambda_* offspring from the\n    population, the offspring are generated by the :func:`varOr` function. The\n    offspring are then evaluated and the next generation population is\n    selected from both the offspring **and** the population. Finally, when\n    *ngen* generations are done, the algorithm returns a tuple with the final\n    population and a :class:`~deap.tools.Logbook` of the evolution.\n    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,\n    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be\n    registered in the toolbox. This algorithm uses the :func:`varOr`\n    variation.\n    '
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    for ind in population:
        initialize_stats_dict(ind)
    population[:] = toolbox.evaluate(population)
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)
    for gen in range(1, ngen + 1):
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        for ind in offspring:
            if ind.statistics['generation'] == 'INVALID':
                ind.statistics['generation'] = gen
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        offspring = toolbox.evaluate(offspring)
        population[:] = toolbox.select(population + offspring, mu)
        if not pbar.disable:
            if verbose == 2:
                high_score = max((halloffame.keys[x].wvalues[1] for x in range(len(halloffame.keys))))
                pbar.write('\nGeneration {0} - Current best internal CV score: {1}'.format(gen, high_score), file=log_file)
            elif verbose == 3:
                pbar.write('\nGeneration {} - Current Pareto front scores:'.format(gen), file=log_file)
                for (pipeline, pipeline_scores) in zip(halloffame.items, reversed(halloffame.keys)):
                    pbar.write('\n{}\t{}\t{}'.format(int(pipeline_scores.wvalues[0]), pipeline_scores.wvalues[1], pipeline), file=log_file)
        if per_generation_function is not None:
            per_generation_function(gen)
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    return (population, logbook)

def cxOnePoint(ind1, ind2):
    if False:
        for i in range(10):
            print('nop')
    'Randomly select in each individual and exchange each subtree with the\n    point as root between each individual.\n    :param ind1: First tree participating in the crossover.\n    :param ind2: Second tree participating in the crossover.\n    :returns: A tuple of two trees.\n    '
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for (idx, node) in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    common_types = []
    for (idx, node) in enumerate(ind2[1:], 1):
        if node.ret in types1 and node.ret not in types2:
            common_types.append(node.ret)
        types2[node.ret].append(idx)
    if len(common_types) > 0:
        type_ = np.random.choice(common_types)
        index1 = np.random.choice(types1[type_])
        index2 = np.random.choice(types2[type_])
        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        (ind1[slice1], ind2[slice2]) = (ind2[slice2], ind1[slice1])
    return (ind1, ind2)

def mutNodeReplacement(individual, pset):
    if False:
        print('Hello World!')
    'Replaces a randomly chosen primitive from *individual* by a randomly\n    chosen primitive no matter if it has the same number of arguments from the :attr:`pset`\n    attribute of the individual.\n    Parameters\n    ----------\n    individual: DEAP individual\n        A list of pipeline operators and model parameters that can be\n        compiled by DEAP into a callable function\n\n    Returns\n    -------\n    individual: DEAP individual\n        Returns the individual with one of point mutation applied to it\n\n    '
    index = np.random.randint(0, len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    if node.arity == 0:
        term = np.random.choice(pset.terminals[node.ret])
        if isclass(term):
            term = term()
        individual[index] = term
    else:
        rindex = None
        if index + 1 < len(individual):
            for (i, tmpnode) in enumerate(individual[index + 1:], index + 1):
                if isinstance(tmpnode, gp.Primitive) and tmpnode.ret in node.args:
                    rindex = i
                    break
        primitives = pset.primitives[node.ret]
        if len(primitives) != 0:
            new_node = np.random.choice(primitives)
            new_subtree = [None] * len(new_node.args)
            if rindex:
                rnode = individual[rindex]
                rslice = individual.searchSubtree(rindex)
                position = np.random.choice([i for (i, a) in enumerate(new_node.args) if a == rnode.ret])
            else:
                position = None
            for (i, arg_type) in enumerate(new_node.args):
                if i != position:
                    term = np.random.choice(pset.terminals[arg_type])
                    if isclass(term):
                        term = term()
                    new_subtree[i] = term
            if rindex:
                new_subtree[position:position + 1] = individual[rslice]
            new_subtree.insert(0, new_node)
            individual[slice_] = new_subtree
    return (individual,)

@threading_timeoutable(default='Timeout')
def _wrapped_cross_val_score(sklearn_pipeline, features, target, cv, scoring_function, sample_weight=None, groups=None, use_dask=False):
    if False:
        i = 10
        return i + 15
    "Fit estimator and compute scores for a given dataset split.\n\n    Parameters\n    ----------\n    sklearn_pipeline : pipeline object implementing 'fit'\n        The object to use to fit the data.\n    features : array-like of shape at least 2D\n        The data to fit.\n    target : array-like, optional, default: None\n        The target variable to try to predict in the case of\n        supervised learning.\n    cv: cross-validation generator\n        Object to be used as a cross-validation generator.\n    scoring_function : callable\n        A scorer callable object / function with signature\n        ``scorer(estimator, X, y)``.\n    sample_weight : array-like, optional\n        List of sample weights to balance (or un-balanace) the dataset target as needed\n    groups: array-like {n_samples, }, optional\n        Group labels for the samples used while splitting the dataset into train/test set\n    use_dask : bool, default False\n        Whether to use dask\n    "
    sample_weight_dict = set_sample_weight(sklearn_pipeline.steps, sample_weight)
    (features, target, groups) = indexable(features, target, groups)
    cv_iter = list(cv.split(features, target, groups))
    scorer = check_scoring(sklearn_pipeline, scoring=scoring_function)
    if use_dask:
        try:
            import dask_ml.model_selection
            import dask
            from dask.delayed import Delayed
        except Exception as e:
            msg = "'use_dask' requires the optional dask and dask-ml depedencies.\n{}".format(e)
            raise ImportError(msg)
        (dsk, keys, n_splits) = dask_ml.model_selection._search.build_graph(estimator=sklearn_pipeline, cv=cv, scorer=scorer, candidate_params=[{}], X=features, y=target, groups=groups, fit_params=sample_weight_dict, refit=False, error_score=float('-inf'))
        cv_results = Delayed(keys[0], dsk)
        scores = [cv_results['split{}_test_score'.format(i)] for i in range(n_splits)]
        CV_score = dask.delayed(np.array)(scores)[:, 0]
        return dask.delayed(np.nanmean)(CV_score)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores = [_fit_and_score(estimator=clone(sklearn_pipeline), X=features, y=target, scorer=scorer, train=train, test=test, verbose=0, parameters=None, error_score='raise', fit_params=sample_weight_dict) for (train, test) in cv_iter]
                if isinstance(scores[0], list):
                    CV_score = np.array(scores)[:, 0]
                elif isinstance(scores[0], dict):
                    from sklearn.model_selection._validation import _aggregate_score_dicts
                    CV_score = _aggregate_score_dicts(scores)['test_scores']
                else:
                    raise ValueError('Incorrect output format from _fit_and_score!')
                CV_score_mean = np.nanmean(CV_score)
            return CV_score_mean
        except TimeoutException:
            return 'Timeout'
        except Exception as e:
            return -float('inf')
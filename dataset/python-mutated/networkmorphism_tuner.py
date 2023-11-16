"""
networkmorphsim_tuner.py
"""
import logging
import os
from schema import Optional, Schema
from nni import ClassArgsValidator
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward
from .bayesian import BayesianOptimizer
from .nn import CnnGenerator, MlpGenerator
from .utils import Constant
from .graph import graph_to_json, json_to_graph
logger = logging.getLogger('NetworkMorphism_AutoML')

class NetworkMorphismClassArgsValidator(ClassArgsValidator):

    def validate_class_args(self, **kwargs):
        if False:
            return 10
        Schema({Optional('optimize_mode'): self.choices('optimize_mode', 'maximize', 'minimize'), Optional('task'): self.choices('task', 'cv', 'nlp', 'common'), Optional('input_width'): int, Optional('input_channel'): int, Optional('n_output_node'): int}).validate(kwargs)

class NetworkMorphismTuner(Tuner):
    """
    NetworkMorphismTuner is a tuner which using network morphism techniques.

    Attributes
    ----------
    n_classes : int
        The class number or output node number (default: ``10``)
    input_shape : tuple
        A tuple including: (input_width, input_width, input_channel)
    t_min : float
        The minimum temperature for simulated annealing. (default: ``Constant.T_MIN``)
    beta : float
        The beta in acquisition function. (default: ``Constant.BETA``)
    algorithm_name : str
        algorithm name used in the network morphism (default: ``"Bayesian"``)
    optimize_mode : str
        optimize mode "minimize" or "maximize" (default: ``"minimize"``)
    verbose : bool
        verbose to print the log (default: ``True``)
    bo : BayesianOptimizer
        The optimizer used in networkmorphsim tuner.
    max_model_size : int
        max model size to the graph (default: ``Constant.MAX_MODEL_SIZE``)
    default_model_len : int
        default model length (default: ``Constant.MODEL_LEN``)
    default_model_width : int
        default model width (default: ``Constant.MODEL_WIDTH``)
    search_space : dict
    """

    def __init__(self, task='cv', input_width=32, input_channel=3, n_output_node=10, algorithm_name='Bayesian', optimize_mode='maximize', path='model_path', verbose=True, beta=Constant.BETA, t_min=Constant.T_MIN, max_model_size=Constant.MAX_MODEL_SIZE, default_model_len=Constant.MODEL_LEN, default_model_width=Constant.MODEL_WIDTH):
        if False:
            print('Hello World!')
        '\n        initilizer of the NetworkMorphismTuner.\n        '
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(os.getcwd(), path)
        if task == 'cv':
            self.generators = [CnnGenerator]
        elif task == 'common':
            self.generators = [MlpGenerator]
        else:
            raise NotImplementedError('{} task not supported in List ["cv","common"]')
        self.n_classes = n_output_node
        self.input_shape = (input_width, input_width, input_channel)
        self.t_min = t_min
        self.beta = beta
        self.algorithm_name = algorithm_name
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.json = None
        self.total_data = {}
        self.verbose = verbose
        self.model_count = 0
        self.bo = BayesianOptimizer(self, self.t_min, self.optimize_mode, self.beta)
        self.training_queue = []
        self.descriptors = []
        self.history = []
        self.max_model_size = max_model_size
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width
        self.search_space = dict()

    def update_search_space(self, search_space):
        if False:
            print('Hello World!')
        '\n        Update search space definition in tuner by search_space in neural architecture.\n        '
        self.search_space = search_space

    def generate_parameters(self, parameter_id, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a set of trial neural architecture, as a serializable object.\n\n        Parameters\n        ----------\n        parameter_id : int\n        '
        if not self.history:
            self.init_search()
        new_father_id = None
        generated_graph = None
        if not self.training_queue:
            (new_father_id, generated_graph) = self.generate()
            new_model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((generated_graph, new_father_id, new_model_id))
            self.descriptors.append(generated_graph.extract_descriptor())
        (graph, father_id, model_id) = self.training_queue.pop(0)
        json_model_path = os.path.join(self.path, str(model_id) + '.json')
        json_out = graph_to_json(graph, json_model_path)
        self.total_data[parameter_id] = (json_out, father_id, model_id)
        return json_out

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Record an observation of the objective function.\n\n        Parameters\n        ----------\n        parameter_id : int\n            the id of a group of paramters that generated by nni manager.\n        parameters : dict\n            A group of parameters.\n        value : dict/float\n            if value is dict, it should have "default" key.\n        '
        reward = extract_scalar_reward(value)
        if parameter_id not in self.total_data:
            raise RuntimeError('Received parameter_id not in total_data.')
        (_, father_id, model_id) = self.total_data[parameter_id]
        graph = self.bo.searcher.load_model_by_id(model_id)
        self.add_model(reward, model_id)
        self.update(father_id, graph, reward, model_id)

    def init_search(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call the generators to generate the initial architectures for the search.\n        '
        if self.verbose:
            logger.info('Initializing search.')
        for generator in self.generators:
            graph = generator(self.n_classes, self.input_shape).generate(self.default_model_len, self.default_model_width)
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())
        if self.verbose:
            logger.info('Initialization finished.')

    def generate(self):
        if False:
            return 10
        '\n        Generate the next neural architecture.\n\n        Returns\n        -------\n        other_info : any object\n            Anything to be saved in the training queue together with the architecture.\n        generated_graph : Graph\n            An instance of Graph.\n        '
        (generated_graph, new_father_id) = self.bo.generate(self.descriptors)
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape).generate(self.default_model_len, self.default_model_width)
        return (new_father_id, generated_graph)

    def update(self, other_info, graph, metric_value, model_id):
        if False:
            while True:
                i = 10
        '\n        Update the controller with evaluation result of a neural architecture.\n\n        Parameters\n        ----------\n        other_info: any object\n            In our case it is the father ID in the search tree.\n        graph: graph.Graph\n            An instance of Graph. The trained neural architecture.\n        metric_value: float\n            The final evaluated metric value.\n        model_id: int\n        '
        father_id = other_info
        self.bo.fit([graph.extract_descriptor()], [metric_value])
        self.bo.add_child(father_id, model_id)

    def add_model(self, metric_value, model_id):
        if False:
            i = 10
            return i + 15
        '\n        Add model to the history, x_queue and y_queue\n\n        Parameters\n        ----------\n        metric_value : float\n        graph : dict\n        model_id : int\n\n        Returns\n        -------\n        model : dict\n        '
        if self.verbose:
            logger.info('Saving model.')
        ret = {'model_id': model_id, 'metric_value': metric_value}
        self.history.append(ret)
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, 'best_model.txt'), 'w')
            file.write('best model: ' + str(model_id))
            file.close()
        return ret

    def get_best_model_id(self):
        if False:
            print('Hello World!')
        '\n        Get the best model_id from history using the metric value\n        '
        if self.optimize_mode is OptimizeMode.Maximize:
            return max(self.history, key=lambda x: x['metric_value'])['model_id']
        return min(self.history, key=lambda x: x['metric_value'])['model_id']

    def load_model_by_id(self, model_id):
        if False:
            i = 10
            return i + 15
        '\n        Get the model by model_id\n\n        Parameters\n        ----------\n        model_id : int\n            model index\n\n        Returns\n        -------\n        load_model : graph.Graph\n            the model graph representation\n        '
        with open(os.path.join(self.path, str(model_id) + '.json')) as fin:
            json_str = fin.read().replace('\n', '')
        load_model = json_to_graph(json_str)
        return load_model

    def load_best_model(self):
        if False:
            return 10
        '\n        Get the best model by model id\n\n        Returns\n        -------\n        load_model : graph.Graph\n            the model graph representation\n        '
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        if False:
            while True:
                i = 10
        '\n        Get the model metric valud by its model_id\n\n        Parameters\n        ----------\n        model_id : int\n            model index\n\n        Returns\n        -------\n        float\n             the model metric\n        '
        for item in self.history:
            if item['model_id'] == model_id:
                return item['metric_value']
        return None

    def import_data(self, data):
        if False:
            i = 10
            return i + 15
        pass
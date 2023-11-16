"""
This module implements Gradient Matching clean-label attacks (a.k.a. Witches' Brew) on Neural Networks.

| Paper link: https://arxiv.org/abs/2009.02276
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Any, Dict, Tuple, TYPE_CHECKING, List
import numpy as np
from tqdm.auto import trange, tqdm
from art.attacks.attack import Attack
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class GradientMatchingAttack(Attack):
    """
    Implementation of Gradient Matching Attack by Geiping, et. al. 2020.
    "Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching"

    | Paper link: https://arxiv.org/abs/2009.02276
    """
    attack_params = Attack.attack_params + ['percent_poison', 'max_trials', 'max_epochs', 'learning_rate_schedule', 'epsilon', 'clip_values', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', percent_poison: float, epsilon: float=0.1, max_trials: int=8, max_epochs: int=250, learning_rate_schedule: Tuple[List[float], List[int]]=([0.1, 0.01, 0.001, 0.0001], [100, 150, 200, 220]), batch_size: int=128, clip_values: Tuple[float, float]=(0, 1.0), verbose: int=1):
        if False:
            while True:
                i = 10
        "\n        Initialize a Gradient Matching Clean-Label poisoning attack (Witches' Brew).\n\n        :param classifier: The proxy classifier used for the attack.\n        :param percent_poison: The ratio of samples to poison among x_train, with range [0,1].\n        :param epsilon: The L-inf perturbation budget.\n        :param max_trials: The maximum number of restarts to optimize the poison.\n        :param max_epochs: The maximum number of epochs to optimize the train per trial.\n        :param learning_rate_schedule: The learning rate schedule to optimize the poison.\n            A List of (learning rate, epoch) pairs. The learning rate is used\n            if the current epoch is less than the specified epoch.\n        :param batch_size: Batch size.\n        :param clip_values: The range of the input features to the classifier.\n        :param verbose: Show progress bars.\n        "
        self.substitute_classifier = classifier
        super().__init__(classifier)
        self.percent_poison = percent_poison
        self.epsilon = epsilon
        self.learning_rate_schedule = learning_rate_schedule
        self.max_trials = max_trials
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.clip_values = clip_values
        self.initial_epoch = 0
        if verbose is True:
            verbose = 1
        self.verbose = verbose
        self._check_params()

    def _initialize_poison(self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_poison: np.ndarray, y_poison: np.ndarray):
        if False:
            print('Hello World!')
        '\n        Initialize poison noises to be optimized.\n\n        :param x_trigger: A list of samples to use as triggers.\n        :param y_trigger: A list of target classes to classify the triggers into.\n        :param x_train: A list of training data to poison a portion of.\n        :param y_train: A list of labels for x_train.\n        '
        from art.estimators.classification.pytorch import PyTorchClassifier
        from art.estimators.classification.tensorflow import TensorFlowV2Classifier
        if isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            initializer = self._initialize_poison_tensorflow
        elif isinstance(self.substitute_classifier, PyTorchClassifier):
            initializer = self._initialize_poison_pytorch
        else:
            raise NotImplementedError('GradientMatchingAttack is currently implemented only for Tensorflow V2 and Pytorch.')
        return initializer(x_trigger, y_trigger, x_poison, y_poison)

    def _finish_poison_tensorflow(self):
        if False:
            while True:
                i = 10
        '\n        Releases any resource and revert back unwanted change to the model.\n        '
        self.substitute_classifier.model.trainable = self.model_trainable

    def _finish_poison_pytorch(self):
        if False:
            return 10
        '\n        Releases any resource and revert back unwanted change to the model.\n        '
        if self.model_trainable:
            self.substitute_classifier.model.train()
        else:
            self.substitute_classifier.model.eval()

    def _initialize_poison_tensorflow(self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_poison: np.ndarray, y_poison: np.ndarray):
        if False:
            i = 10
            return i + 15
        '\n        Initialize poison noises to be optimized.\n\n        :param x_trigger: A list of samples to use as triggers.\n        :param y_trigger: A list of target classes to classify the triggers into.\n        :param x_poison: A list of training data to poison a portion of.\n        :param y_poison: A list of true labels for x_poison.\n        '
        from tensorflow.keras import backend as K
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Embedding, Add, Lambda
        from art.estimators.classification.tensorflow import TensorFlowV2Classifier
        if isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            classifier = self.substitute_classifier
        else:
            raise Exception("This method requires `TensorFlowV2Classifier` as `substitute_classifier`'s type")
        self.model_trainable = classifier.model.trainable
        classifier.model.trainable = False

        def _weight_grad(classifier: TensorFlowV2Classifier, x: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
            if False:
                return 10
            import tensorflow as tf
            with tf.GradientTape() as t:
                t.watch(classifier.model.weights)
                output = classifier.model(x, training=False)
                loss = classifier.loss_object(target, output)
            d_w = t.gradient(loss, classifier.model.weights)
            d_w = [w for w in d_w if w is not None]
            d_w = tf.concat([tf.reshape(d, [-1]) for d in d_w], 0)
            d_w_norm = d_w / tf.sqrt(tf.reduce_sum(tf.square(d_w)))
            return d_w_norm
        self.grad_ws_norm = _weight_grad(classifier, tf.constant(x_trigger), tf.constant(y_trigger))
        input_poison = Input(batch_shape=classifier.model.input.shape)
        input_indices = Input(shape=())
        y_true_poison = Input(shape=np.shape(y_poison)[1:])
        embedding_layer = Embedding(len(x_poison), np.prod(x_poison.shape[1:]), embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=self.epsilon * 0.01))
        embeddings = embedding_layer(input_indices)
        embeddings = tf.tanh(embeddings) * self.epsilon
        embeddings = tf.reshape(embeddings, tf.shape(input_poison))
        input_noised = Add()([input_poison, embeddings])
        input_noised = Lambda(lambda x: K.clip(x, self.clip_values[0], self.clip_values[1]))(input_noised)

        def loss_fn(input_noised: tf.Tensor, target: tf.Tensor, grad_ws_norm: tf.Tensor):
            if False:
                while True:
                    i = 10
            d_w2_norm = _weight_grad(classifier, input_noised, target)
            B = 1 - tf.reduce_sum(grad_ws_norm * d_w2_norm)
            return B
        B = tf.keras.layers.Lambda(lambda x: loss_fn(x[0], x[1], x[2]))([input_noised, y_true_poison, self.grad_ws_norm])
        self.backdoor_model = tf.keras.models.Model([input_poison, y_true_poison, input_indices], [input_noised, B])
        self.backdoor_model.add_loss(B)

        class PredefinedLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            """
            Use a preset learning rate based on the current training epoch.
            """

            def __init__(self, learning_rates: List[float], milestones: List[int]):
                if False:
                    while True:
                        i = 10
                self.schedule = list(zip(milestones, learning_rates))

            def __call__(self, step: int) -> float:
                if False:
                    i = 10
                    return i + 15
                lr_prev = self.schedule[0][1]
                for (m, learning_rate) in self.schedule:
                    if step < m:
                        return lr_prev
                    lr_prev = learning_rate
                return lr_prev

            def get_config(self) -> Dict:
                if False:
                    return 10
                '\n                Returns the parameters.\n                '
                return {'schedule': self.schedule}
        self.optimizer = tf.keras.optimizers.Adam(gradient_transformers=[lambda grads_and_vars: [(tf.sign(g), v) for (g, v) in grads_and_vars]])
        self.lr_schedule = tf.keras.callbacks.LearningRateScheduler(PredefinedLRSchedule(*self.learning_rate_schedule))

    def _initialize_poison_pytorch(self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_poison: np.ndarray, y_poison: np.ndarray):
        if False:
            i = 10
            return i + 15
        import torch
        from torch import nn
        from art.estimators.classification.pytorch import PyTorchClassifier
        if isinstance(self.substitute_classifier, PyTorchClassifier):
            classifier = self.substitute_classifier
        else:
            raise Exception("This method requires `PyTorchClassifier` as `substitute_classifier`'s type")
        num_poison = len(x_poison)
        len_noise = np.prod(x_poison.shape[1:])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_trainable = self.substitute_classifier.model.training
        self.substitute_classifier.model.eval()

        def _weight_grad(classifier: PyTorchClassifier, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            classifier.model.zero_grad()
            y = classifier.model(x)
            loss_ = classifier.loss(y, target)
            gradspred = torch.autograd.grad(loss_, list(classifier.model.parameters()), create_graph=True, retain_graph=True)
            d_w = torch.cat([w.flatten() for w in gradspred])
            d_w_norm = d_w / torch.sqrt(torch.sum(torch.square(d_w)))
            return d_w_norm

        class NoiseEmbedding(nn.Module):
            """
            Gradient matching noise layer.
            """

            def __init__(self, num_poison: int, len_noise: int, epsilon: float, clip_values: Tuple[float, float]):
                if False:
                    print('Hello World!')
                super().__init__()
                self.embedding_layer = nn.Embedding(num_poison, len_noise)
                torch.nn.init.normal_(self.embedding_layer.weight, std=epsilon * 0.0001)
                self.epsilon = epsilon
                self.clip_values = clip_values

            def forward(self, input_poison: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
                if False:
                    return 10
                '\n                Applies the noise variable to the input.\n                Input to the model must match its index as the noise is specific to the input.\n                '
                embeddings = self.embedding_layer(input_indices).to(device)
                embeddings = torch.tanh(embeddings) * self.epsilon
                embeddings = embeddings.view(input_poison.shape)
                input_noised = input_poison + embeddings
                input_noised = torch.clip(input_noised, self.clip_values[0], self.clip_values[1])
                return input_noised

        class BackdoorModel(nn.Module):
            """
            Backdoor model computing the B loss.
            """

            def __init__(self, gradient_matching: GradientMatchingAttack, classifier: PyTorchClassifier, epsilon: float, num_poison: int, len_noise: int, min_: float, max_: float):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.gradient_matching = gradient_matching
                self.classifier = classifier
                self.noise_embedding = NoiseEmbedding(num_poison, len_noise, epsilon, (min_, max_))
                self.cos = nn.CosineSimilarity(dim=-1)

            def forward(self, x: torch.Tensor, indices_poison: torch.Tensor, y: torch.Tensor, grad_ws_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if False:
                    print('Hello World!')
                '\n                Applies the poison noise and compute the loss with respect to the target gradient.\n                '
                poisoned_samples = self.noise_embedding(x, indices_poison)
                d_w2_norm = _weight_grad(self.classifier, poisoned_samples, y)
                d_w2_norm.requires_grad_(True)
                B_score = 1 - self.cos(grad_ws_norm, d_w2_norm)
                return (B_score, poisoned_samples)
        self.grad_ws_norm = _weight_grad(classifier, torch.tensor(x_trigger, device=device, dtype=torch.float32), torch.tensor(y_trigger, device=device)).detach()
        self.grad_ws_norm.requires_grad_(False)
        self.backdoor_model = BackdoorModel(self, classifier, self.epsilon, num_poison, len_noise, self.clip_values[0], self.clip_values[1]).to(device)
        self.optimizer = torch.optim.Adam(self.backdoor_model.noise_embedding.embedding_layer.parameters(), lr=1)

        class PredefinedLRSchedule:
            """
            Use a preset learning rate based on the current training epoch.
            """

            def __init__(self, learning_rates: List[float], milestones: List[int]):
                if False:
                    print('Hello World!')
                self.schedule = list(zip(milestones, learning_rates))

            def __call__(self, step: int) -> float:
                if False:
                    while True:
                        i = 10
                lr_prev = self.schedule[0][1]
                for (m, learning_rate) in self.schedule:
                    if step < m:
                        return lr_prev
                    lr_prev = learning_rate
                return lr_prev

            def get_config(self) -> Dict:
                if False:
                    for i in range(10):
                        print('nop')
                '\n                returns a dictionary of parameters.\n                '
                return {'schedule': self.schedule}
        self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer, PredefinedLRSchedule(*self.learning_rate_schedule))

    def poison(self, x_trigger: np.ndarray, y_trigger: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Optimizes a portion of poisoned samples from x_train to make a model classify x_target\n        as y_target by matching the gradients.\n\n        :param x_trigger: A list of samples to use as triggers.\n        :param y_trigger: A list of target classes to classify the triggers into.\n        :param x_train: A list of training data to poison a portion of.\n        :param y_train: A list of labels for x_train.\n        :return: A list of poisoned samples, and y_train.\n        '
        from art.estimators.classification.pytorch import PyTorchClassifier
        from art.estimators.classification.tensorflow import TensorFlowV2Classifier
        if isinstance(self.substitute_classifier, TensorFlowV2Classifier):
            poisoner = self._poison__tensorflow
            finish_poisoning = self._finish_poison_tensorflow
        elif isinstance(self.substitute_classifier, PyTorchClassifier):
            poisoner = self._poison__pytorch
            finish_poisoning = self._finish_poison_pytorch
        else:
            raise NotImplementedError('GradientMatchingAttack is currently implemented only for Tensorflow V2 and Pytorch.')
        x_train = np.copy(x_train)
        y_train = np.copy(y_train)
        if len(np.shape(y_trigger)) == 2:
            classes_target = set(np.argmax(y_trigger, axis=-1))
        else:
            classes_target = set(y_trigger)
        num_poison_samples = int(self.percent_poison * len(x_train))
        best_B = np.finfo(np.float32).max
        best_x_poisoned = None
        best_indices_poison = None
        if len(np.shape(y_train)) == 2:
            y_train_classes = np.argmax(y_train, axis=-1)
        else:
            y_train_classes = y_train
        for _ in trange(self.max_trials):
            indices_poison = np.random.permutation(np.where([y in classes_target for y in y_train_classes])[0])[:num_poison_samples]
            x_poison = x_train[indices_poison]
            y_poison = y_train[indices_poison]
            self._initialize_poison(x_trigger, y_trigger, x_poison, y_poison)
            (x_poisoned, B_) = poisoner(x_poison, y_poison)
            finish_poisoning()
            B_ = np.mean(B_)
            if B_ < best_B:
                best_B = B_
                best_x_poisoned = x_poisoned
                best_indices_poison = indices_poison
        if self.verbose > 0:
            print('Best B-score:', best_B)
        x_train[best_indices_poison] = best_x_poisoned
        return (x_train, y_train)

    def _poison__pytorch(self, x_poison: np.ndarray, y_poison: np.ndarray) -> Tuple[Any, Any]:
        if False:
            print('Hello World!')
        '\n        Optimize the poison by matching the gradient within the perturbation budget.\n\n        :param x_poison: List of samples to poison.\n        :param y_poison: List of the labels for x_poison.\n        :return: A pair of poisoned samples, B-score (cosine similarity of the gradients).\n        '
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        class PoisonDataset(torch.utils.data.Dataset):
            """
            Iterator for a dataset to poison.
            """

            def __init__(self, x: np.ndarray, y: np.ndarray):
                if False:
                    while True:
                        i = 10
                self.len = x.shape[0]
                self.x = torch.as_tensor(x, dtype=torch.float)
                self.y = torch.as_tensor(y)

            def __getitem__(self, index):
                if False:
                    i = 10
                    return i + 15
                return (self.x[index], torch.as_tensor([index]), self.y[index])

            def __len__(self):
                if False:
                    return 10
                return self.len
        trainloader = torch.utils.data.DataLoader(PoisonDataset(x_poison, y_poison), batch_size=self.batch_size, shuffle=False, num_workers=1)
        epoch_iterator = trange(self.initial_epoch, self.max_epochs) if self.verbose > 0 else range(self.initial_epoch, self.max_epochs)
        for _ in epoch_iterator:
            batch_iterator = tqdm(trainloader) if isinstance(self.verbose, int) and self.verbose >= 2 else trainloader
            sum_loss = 0
            count = 0
            for (x, indices, y) in batch_iterator:
                x = x.to(device)
                y = y.to(device)
                indices = indices.to(device)
                self.backdoor_model.zero_grad()
                (loss, poisoned_samples) = self.backdoor_model(x, indices, y, self.grad_ws_norm)
                loss.backward()
                self.backdoor_model.noise_embedding.embedding_layer.weight.grad.sign_()
                self.optimizer.step()
                sum_loss += loss.clone().cpu().detach().numpy()
                count += 1
            if self.verbose > 0:
                epoch_iterator.set_postfix(loss=sum_loss / count)
            self.lr_schedule.step()
        B_sum = 0
        count = 0
        all_poisoned_samples = []
        self.backdoor_model.eval()
        poisonloader = torch.utils.data.DataLoader(PoisonDataset(x_poison, y_poison), batch_size=self.batch_size, shuffle=False, num_workers=1)
        for (x, indices, y) in poisonloader:
            x = x.to(device)
            y = y.to(device)
            indices = indices.to(device)
            (B, poisoned_samples) = self.backdoor_model(x, indices, y, self.grad_ws_norm)
            all_poisoned_samples.append(poisoned_samples.detach().cpu().numpy())
            B_sum += B.detach().cpu().numpy()
            count += 1
        return (np.concatenate(all_poisoned_samples, axis=0), B_sum / count)

    def _poison__tensorflow(self, x_poison: np.ndarray, y_poison: np.ndarray) -> Tuple[Any, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Optimize the poison by matching the gradient within the perturbation budget.\n\n        :param x_poison: List of samples to poison.\n        :param y_poison: List of the labels for x_poison.\n        :return: A pair of poisoned samples, B-score (cosine similarity of the gradients).\n        '
        self.backdoor_model.compile(loss=None, optimizer=self.optimizer)
        callbacks = [self.lr_schedule]
        if self.verbose > 0:
            from tqdm.keras import TqdmCallback
            callbacks.append(TqdmCallback(verbose=self.verbose - 1))
        self.backdoor_model.fit([x_poison, y_poison, np.arange(len(y_poison))], callbacks=callbacks, batch_size=self.batch_size, initial_epoch=self.initial_epoch, epochs=self.max_epochs, verbose=0)
        [input_noised_, B_] = self.backdoor_model.predict([x_poison, y_poison, np.arange(len(y_poison))], batch_size=self.batch_size)
        return (input_noised_, B_)

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not isinstance(self.learning_rate_schedule, tuple) or len(self.learning_rate_schedule) != 2:
            raise ValueError('learning_rate_schedule must be a pair of a list of learning rates and a list of epochs')
        if self.percent_poison > 1 or self.percent_poison < 0:
            raise ValueError('percent_poison must be in [0, 1]')
        if self.max_epochs < 1:
            raise ValueError('max_epochs must be positive')
        if self.max_trials < 1:
            raise ValueError('max_trials must be positive')
        if not isinstance(self.clip_values, tuple) or len(self.clip_values) != 2:
            raise ValueError('clip_values must be a pair (min, max) of floats')
        if self.epsilon <= 0:
            raise ValueError('epsilon must be nonnegative')
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if isinstance(self.verbose, int) and self.verbose < 0 or (not isinstance(self.verbose, int) and (not isinstance(self.verbose, bool))):
            raise ValueError('verbose must be nonnegative integer or Boolean')
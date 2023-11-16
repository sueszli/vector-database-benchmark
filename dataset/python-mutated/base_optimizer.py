import re
import warnings
from keras import backend
from keras import initializers
from keras import ops
from keras.optimizers.schedules import learning_rate_schedule
from keras.saving import serialization_lib
from keras.utils import tracking
from keras.utils.naming import auto_name

class BaseOptimizer:

    def __init__(self, learning_rate, weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, loss_scale_factor=None, name=None, **kwargs):
        if False:
            return 10
        self._lock = False
        if kwargs.pop('decay', None) is not None:
            warnings.warn('Argument `decay` is no longer supported and will be ignored.')
        if kwargs:
            raise ValueError(f'Argument(s) not recognized: {kwargs}')
        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.global_clipnorm = global_clipnorm
        self.clipvalue = clipvalue
        self.use_ema = use_ema
        self.loss_scale_factor = loss_scale_factor
        if use_ema:
            if ema_momentum > 1 or ema_momentum < 0:
                raise ValueError(f'`ema_momentum` must be in the range [0, 1]. Received: ema_momentum={ema_momentum}')
            if ema_overwrite_frequency and (not isinstance(ema_overwrite_frequency, int) or ema_overwrite_frequency < 1):
                raise ValueError(f'`ema_overwrite_frequency` must be an integer >= 1 or None. Received: ema_overwrite_frequency={ema_overwrite_frequency}')
        self.ema_momentum = ema_momentum
        self.ema_overwrite_frequency = ema_overwrite_frequency
        if self.clipnorm is not None and self.global_clipnorm is not None:
            raise ValueError(f'Only one of `clipnorm` and `global_clipnorm` can be set. Received: clipnorm={self.clipnorm}, global_clipnorm={self.global_clipnorm}')
        self.built = False
        self._variables = []
        self._trainable_variables = []
        self._tracker = tracking.Tracker({'variables': (lambda x: isinstance(x, backend.Variable), self._variables)})
        self._trainable_variables_indices = {}
        with backend.name_scope(self.name, caller=self):
            iterations = backend.Variable(0, name='iteration', dtype='int', trainable=False)
        self._track_variable(iterations)
        self.iterations = iterations
        if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if not isinstance(learning_rate, float):
                raise ValueError(f'Argument `learning_rate` should be float, or an instance of LearningRateSchedule, or a callable (that takes in the current iteration value and returns the corresponding learning rate value). Received instead: learning_rate={learning_rate}')
            with backend.name_scope(self.name, caller=self):
                learning_rate = backend.Variable(learning_rate, name='learning_rate', dtype=backend.floatx(), trainable=False)
            self._track_variable(learning_rate)
            self._learning_rate = learning_rate

    def _track_variable(self, variable):
        if False:
            i = 10
            return i + 15
        self._tracker.add_to_store('variables', variable)

    @tracking.no_automatic_dependency_tracking
    def build(self, variables):
        if False:
            i = 10
            return i + 15
        if self.use_ema:
            self._model_variables_moving_average = []
            self._ema_vars_initialized = False
        for (i, variable) in enumerate(variables):
            self._trainable_variables_indices[self._var_key(variable)] = i
            if self.use_ema:
                self._model_variables_moving_average.append(self.add_variable_from_reference(variable, 'average'))
        self._trainable_variables = variables[:]
        self.built = True

    def _var_key(self, variable):
        if False:
            i = 10
            return i + 15
        return id(variable)

    @property
    def variables(self):
        if False:
            return 10
        return self._variables[:]

    def _get_variable_index(self, variable):
        if False:
            i = 10
            return i + 15
        return self._trainable_variables_indices[self._var_key(variable)]

    def add_variable(self, shape, initializer='zeros', dtype=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        self._check_super_called()
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(initializer=initializer, shape=shape, dtype=dtype, trainable=False, name=name)
        self._track_variable(variable)
        return variable

    def add_variable_from_reference(self, reference_variable, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Add an all-zeros variable with the shape and dtype of a reference\n        variable.\n        '
        initializer = initializers.Zeros()
        name = name or 'var'
        if hasattr(reference_variable, 'path'):
            name = reference_variable.path.replace('/', '_') + '_' + name
        else:
            name = reference_variable.name + '_' + name
        return self.add_variable(shape=reference_variable.shape, initializer=initializer, dtype=reference_variable.dtype, name=name)

    def _check_variables_are_known(self, variables):
        if False:
            while True:
                i = 10
        for v in variables:
            if self._var_key(v) not in self._trainable_variables_indices:
                raise ValueError(f'Unknown variable: {v}. This optimizer can only be called for the variables it was originally built with. When working with a new set of variables, you should recreate a new optimizer instance.')

    def assign(self, variable, value):
        if False:
            for i in range(10):
                print('nop')
        'Assign a value to a variable.\n\n        This should be used in optimizers instead of `variable.assign(value)` to\n        support backend specific optimizations.\n        Note that the variable can be a model variable or an optimizer variable;\n        it can be a backend native variable or a Keras variable.\n\n        Args:\n            variable: The variable to update.\n            value: The value to add to the variable.\n        '
        variable.assign(value)

    def assign_add(self, variable, value):
        if False:
            print('Hello World!')
        'Add a value to a variable.\n\n        This should be used in optimizers instead of\n        `variable.assign_add(value)` to support backend specific optimizations.\n        Note that the variable can be a model variable or an optimizer variable;\n        it can be a backend native variable or a Keras variable.\n\n        Args:\n            variable: The variable to update.\n            value: The value to add to the variable.\n        '
        variable.assign_add(value)

    def assign_sub(self, variable, value):
        if False:
            return 10
        'Subtract a value from a variable.\n\n        This should be used in optimizers instead of\n        `variable.assign_sub(value)` to support backend specific optimizations.\n        Note that the variable can be a model variable or an optimizer variable;\n        it can be a backend native variable or a Keras variable.\n\n        Args:\n            variable: The variable to update.\n            value: The value to add to the variable.\n        '
        variable.assign_sub(value)

    def update_step(self, gradient, variable, learning_rate):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars):
        if False:
            for i in range(10):
                print('nop')
        (grads, trainable_variables) = zip(*grads_and_vars)
        self.apply(grads, trainable_variables)
        return self.iterations

    def apply(self, grads, trainable_variables=None):
        if False:
            while True:
                i = 10
        '\n        `grads` should be a list of gradient tensors\n        with 1:1 mapping to the list of variables the optimizer was built with.\n\n        `variables` can be provided on the first call to build the optimizer.\n        '
        if len(grads) == 0:
            return
        if trainable_variables is None:
            if not self.built:
                raise ValueError('When passing `grads` without `variables`, the optimizer must already be built on a list of variables. Call `optimizer.build(trainable_variables)` first. ')
            if len(grads) != len(self._trainable_variables_indices):
                raise ValueError(f'When passing `grads` as a list of gradient tensors, the gradients must match `optimizer.variables` one-to-on. Received a list of {len(grads)} gradients, but the optimizer is tracking {len(self._trainable_variables)} trainable variables.')
            trainable_variables = self._trainable_variables
        else:
            trainable_variables = list(trainable_variables)
            if not self.built:
                with backend.name_scope(self.name, caller=self):
                    self.build(trainable_variables)
                self.built = True
            self._check_variables_are_known(trainable_variables)
        with backend.name_scope(self.name, caller=self):
            (grads, trainable_variables) = self._filter_empty_gradients(grads, trainable_variables)
            if len(list(grads)) == 0:
                return
            scale = self.loss_scale_factor
            if scale is not None:
                grads = [g if g is None else g / scale for g in grads]
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)
            self._internal_apply_gradients(list(zip(grads, trainable_variables)))
            for variable in trainable_variables:
                if getattr(variable, 'constraint', None) is not None:
                    variable.assign(variable.constraint(variable))

    def _internal_apply_gradients(self, grads_and_vars):
        if False:
            while True:
                i = 10
        for (grad, var) in grads_and_vars:
            self.update_step(grad, var, self.learning_rate)
        self.iterations.assign(self.iterations + 1)

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        if False:
            print('Hello World!')
        self._check_super_called()
        if not self.built:
            raise ValueError(f'To call `stateless_apply`, {self.__class__.__name__} must be built (i.e. its variables must have been created). You can build it via `optimizer.build(trainable_variables)`.')
        if len(optimizer_variables) != len(self.variables):
            raise ValueError(f'Argument `optimizer_variables` must be a list of tensors corresponding 1:1 to {self.__class__.__name__}().variables. Received list with length {len(optimizer_variables)}, but expected {len(self.variables)} variables.')
        if len(trainable_variables) != len(self._trainable_variables):
            raise ValueError(f'Argument `optimizer_variables` must be a list of tensors corresponding 1:1 to the trainable variables list that the optimizer was built with. Received len(trainable_variables) == {len(trainable_variables)} whereas the optimizer was built with {len(self._trainable_variables)} variables.')
        mapping = list(zip(self._trainable_variables, trainable_variables)) + list(zip(self.variables, optimizer_variables))
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.apply(grads)
        trainable_variables = []
        for v in self._trainable_variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                trainable_variables.append(new_v)
            else:
                trainable_variables.append(v)
        optimizer_variables = []
        for v in self.variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                optimizer_variables.append(new_v)
            else:
                optimizer_variables.append(v)
        return (trainable_variables, optimizer_variables)

    def scale_loss(self, loss):
        if False:
            i = 10
            return i + 15
        'Scale the loss before computing gradients.\n\n        Scales the loss before gradients are computed in a `train_step`. This\n        is primarily useful during mixed precision training to prevent numeric\n        underflow.\n        '
        if self.loss_scale_factor is not None:
            return loss * self.loss_scale_factor
        return loss

    @property
    def learning_rate(self):
        if False:
            while True:
                i = 10
        return self._get_current_learning_rate()

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if False:
            while True:
                i = 10
        if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
            self._learning_rate = learning_rate
        elif callable(learning_rate):
            self._learning_rate = learning_rate
        else:
            if isinstance(self._learning_rate, learning_rate_schedule.LearningRateSchedule):
                raise TypeError('This optimizer was created with a `LearningRateSchedule` object as its `learning_rate` constructor argument, hence its learning rate is not settable. If you need the learning rate to be settable, you should instantiate the optimizer with a float `learning_rate` argument.')
            self._learning_rate.assign(learning_rate)

    def set_weights(self, weights):
        if False:
            print('Hello World!')
        'Set the weights of the optimizer.'
        if not self.built:
            raise ValueError('You are calling `set_weights()` on an optimizer that has not yet been built. Please call `optimizer.build(trainable_variables)` to create the optimizer weights before calling `set_weights()`.')
        for (variable, weight) in zip(self._variables, weights):
            if variable.shape != weight.shape:
                raise ValueError(f'Optimizer variable {self._var_key(variable)} has shape {str(variable.shape)} not compatible with provided weight shape {str(weight.shape)}.')
            variable.assign(weight)

    def save_own_variables(self, store):
        if False:
            while True:
                i = 10
        'Get the state of this optimizer object.'
        for (i, variable) in enumerate(self.variables):
            store[str(i)] = variable.numpy()

    def load_own_variables(self, store):
        if False:
            while True:
                i = 10
        'Set the state of this optimizer object.'
        if len(store.keys()) != len(self.variables):
            msg = f"Skipping variable loading for optimizer '{self.name}', because it has {len(self.variables)} variables whereas the saved optimizer has {len(store.keys())} variables. "
            if len(self.variables) == 0:
                msg += 'This is likely because the optimizer has not been called/built yet.'
            warnings.warn(msg, stacklevel=2)
            return
        for (i, variable) in enumerate(self.variables):
            variable.assign(store[str(i)])

    def _get_current_learning_rate(self):
        if False:
            return 10
        if isinstance(self._learning_rate, learning_rate_schedule.LearningRateSchedule):
            return self._learning_rate(self.iterations)
        elif callable(self._learning_rate):
            return self._learning_rate(self.iterations)
        return self._learning_rate

    def _filter_empty_gradients(self, grads, vars):
        if False:
            for i in range(10):
                print('nop')
        for grad in grads:
            if grad is None:
                filtered = [(g, v) for (g, v) in zip(grads, vars) if g is not None]
                if not filtered:
                    raise ValueError('No gradients provided for any variable.')
                if len(filtered) < len(grads):
                    missing_grad_vars = [v for (g, v) in zip(grads, vars) if g is None]
                    warnings.warn(f'Gradients do not exist for variables {[v.name for v in missing_grad_vars]} when minimizing the loss. If using `model.compile()`, did you forget to provide a `loss` argument?')
                return zip(*filtered)
        return (grads, vars)

    def _clip_gradients(self, grads):
        if False:
            for i in range(10):
                print('nop')
        if self.clipnorm and self.clipnorm > 0:
            clipped_grads = []
            for g in grads:
                if g is None:
                    clipped_grads.append(g)
                else:
                    clipped_grads.append(clip_by_norm(g, self.clipnorm))
            return clipped_grads
        if self.global_clipnorm and self.global_clipnorm > 0:
            return clip_by_global_norm(grads, self.global_clipnorm)
        if self.clipvalue and self.clipvalue > 0:
            clipped_grads = []
            for g in grads:
                if g is None:
                    clipped_grads.append(g)
                else:
                    clipped_grads.append(ops.clip(g, -self.clipvalue, self.clipvalue))
            return clipped_grads
        return grads

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        if False:
            for i in range(10):
                print('nop')
        "Exclude variables from weight decay.\n\n        This method must be called before the optimizer's `build` method is\n        called. You can set specific variables to exclude out, or set a list of\n        strings as the anchor words, if any of which appear in a variable's\n        name, then the variable is excluded.\n\n        Args:\n            var_list: A list of `tf.Variable`s to exclude from weight decay.\n            var_names: A list of strings. If any string in `var_names` appear\n                in the model variable's name, then this model variable is\n                excluded from weight decay. For example, `var_names=['bias']`\n                excludes all bias variables from weight decay.\n        "
        if hasattr(self, '_built') and self._built:
            raise ValueError('`exclude_from_weight_decay()` can only be configued before the optimizer is built.')
        if var_list:
            self._exclude_from_weight_decay = [self._var_key(variable) for variable in var_list]
        else:
            self._exclude_from_weight_decay = []
        self._exclude_from_weight_decay_names = var_names or []

    def _use_weight_decay(self, variable):
        if False:
            i = 10
            return i + 15
        exclude_from_weight_decay = getattr(self, '_exclude_from_weight_decay', [])
        exclude_from_weight_decay_names = getattr(self, '_exclude_from_weight_decay_names', [])
        variable_id = self._var_key(variable)
        for exclude_id in exclude_from_weight_decay:
            if variable_id == exclude_id:
                return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def _apply_weight_decay(self, variables):
        if False:
            return 10
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = ops.cast(self.learning_rate, variable.dtype)
                wd = ops.cast(self.weight_decay, variable.dtype)
                variable.assign(variable - variable * wd * lr)

    def _check_super_called(self):
        if False:
            return 10
        if not hasattr(self, '_lock'):
            raise RuntimeError(f"In optimizer '{self.__class__.__name__}', you forgot to call `super().__init__()` as the first statement in the `__init__()` method. Go add it!")

    def _update_model_variables_moving_average(self, var_list):
        if False:
            while True:
                i = 10
        'Update the stored moving average using the latest value.'
        if self.use_ema:
            for (var, average) in zip(var_list, self._model_variables_moving_average):
                if self._ema_vars_initialized:
                    average.assign(self.ema_momentum * average + (1 - self.ema_momentum) * var)
                else:
                    average.assign(var)
            self._ema_vars_initialized = True

    def _overwrite_model_variables_with_average_value(self, var_list):
        if False:
            for i in range(10):
                print('nop')
        'Overwrite model variables with its moving average.'
        if len(var_list) != len(self._model_variables_moving_average):
            raise ValueError(f'The length of model variables ({len(var_list)}) to override does not match the length of model variables stored in the optimizer ({len(self._model_variables_moving_average)}). Please check if the optimizer was called on your model.')
        self._overwrite_model_variables_with_average_value_helper(var_list)

    def _overwrite_model_variables_with_average_value_helper(self, var_list):
        if False:
            return 10
        'Helper function that overwrites model variables.'
        for (var, average_var) in zip(var_list, self._model_variables_moving_average):
            var.assign(average_var)

    def finalize_variable_values(self, var_list):
        if False:
            return 10
        "Set the final value of model's trainable variables.\n\n        Sometimes there are some extra steps before ending the variable updates,\n        such as overriding the model variables with its average value.\n\n        Args:\n          var_list: list of model variables.\n        "
        if self.use_ema:
            self._overwrite_model_variables_with_average_value(var_list)

    def get_config(self):
        if False:
            return 10
        'Returns the config of the optimizer.\n\n        An optimizer config is a Python dictionary (serializable)\n        containing the configuration of an optimizer.\n        The same optimizer can be reinstantiated later\n        (without any saved state) from this configuration.\n\n        Subclass optimizer should override this method to include other\n        hyperparameters.\n\n        Returns:\n            Python dictionary.\n        '
        if isinstance(self._learning_rate, learning_rate_schedule.LearningRateSchedule):
            learning_rate = learning_rate_schedule.serialize(self._learning_rate)
        elif isinstance(self._learning_rate, backend.Variable):
            learning_rate = float(self._learning_rate.numpy())
        elif ops.is_tensor(self._learning_rate):
            learning_rate = float(self._learning_rate)
        elif callable(self._learning_rate):
            learning_rate = serialization_lib.serialize_keras_object(self._learning_rate)
        config = {'name': self.name, 'learning_rate': learning_rate, 'weight_decay': self.weight_decay, 'clipnorm': self.clipnorm, 'global_clipnorm': self.global_clipnorm, 'clipvalue': self.clipvalue, 'use_ema': self.use_ema, 'ema_momentum': self.ema_momentum, 'ema_overwrite_frequency': self.ema_overwrite_frequency, 'loss_scale_factor': self.loss_scale_factor}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if False:
            while True:
                i = 10
        'Creates an optimizer from its config.\n\n        This method is the reverse of `get_config`, capable of instantiating the\n        same optimizer from the config dictionary.\n\n        Args:\n            config: A Python dictionary, typically the output of get_config.\n            custom_objects: A Python dictionary mapping names to additional\n              user-defined Python objects needed to recreate this optimizer.\n\n        Returns:\n            An optimizer instance.\n        '
        if 'learning_rate' in config:
            if isinstance(config['learning_rate'], dict):
                config['learning_rate'] = serialization_lib.deserialize_keras_object(config['learning_rate'], custom_objects=custom_objects)
        return cls(**config)

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        if name != '_lock':
            self._check_super_called()
        if hasattr(self, '_tracker'):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)
base_optimizer_keyword_args = "name: String. The name to use\n          for momentum accumulator weights created by\n          the optimizer.\n        weight_decay: Float. If set, weight decay is applied.\n        clipnorm: Float. If set, the gradient of each weight is individually\n          clipped so that its norm is no higher than this value.\n        clipvalue: Float. If set, the gradient of each weight is clipped to be\n          no higher than this value.\n        global_clipnorm: Float. If set, the gradient of all weights is clipped\n          so that their global norm is no higher than this value.\n        use_ema: Boolean, defaults to False. If True, exponential moving average\n          (EMA) is applied. EMA consists of computing an exponential moving\n          average of the weights of the model (as the weight values change after\n          each training batch), and periodically overwriting the weights with\n          their moving average.\n        ema_momentum: Float, defaults to 0.99. Only used if `use_ema=True`.\n          This is the momentum to use when computing\n          the EMA of the model's weights:\n          `new_average = ema_momentum * old_average + (1 - ema_momentum) *\n          current_variable_value`.\n        ema_overwrite_frequency: Int or None, defaults to None. Only used if\n          `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,\n          we overwrite the model variable by its moving average.\n          If None, the optimizer\n          does not overwrite model variables in the middle of training, and you\n          need to explicitly overwrite the variables at the end of training\n          by calling `optimizer.finalize_variable_values()`\n          (which updates the model\n          variables in-place). When using the built-in `fit()` training loop,\n          this happens automatically after the last epoch,\n          and you don't need to do anything.\n        loss_scale_factor: Float or `None`. If a float, the scale factor will\n          be multiplied the loss before computing gradients, and the inverse of\n          the scale factor will be multiplied by the gradients before updating\n          variables. Useful for preventing underflow during mixed precision\n          training. Alternately, `keras.optimizers.LossScaleOptimizer` will\n          automatically set a loss scale factor.\n"

def clip_by_norm(values, clip_norm, axes=None):
    if False:
        return 10
    l2sum = ops.sum(values * values, axes, keepdims=True)
    pred = l2sum > 0
    l2sum_safe = ops.where(pred, l2sum, ops.ones_like(l2sum))
    l2norm = ops.where(pred, ops.sqrt(l2sum_safe), l2sum)
    intermediate = values * clip_norm
    values_clip = ops.convert_to_tensor(intermediate) / ops.maximum(l2norm, clip_norm)
    return values_clip

def global_norm(value_list):
    if False:
        return 10
    'Computes the global norm of multiple tensors.'
    squared_norms = []
    for v in value_list:
        if v is not None:
            squared_norms.append(ops.sum(ops.square(v)))
    squared_norm = ops.sum(ops.stack(squared_norms))
    return ops.sqrt(squared_norm)

def clip_by_global_norm(value_list, clip_norm):
    if False:
        i = 10
        return i + 15
    use_norm = global_norm(value_list)
    scale_for_finite = clip_norm * ops.minimum(1.0 / use_norm, 1.0 / clip_norm)
    scale = scale_for_finite + (use_norm - use_norm)
    values_clipped = []
    for v in value_list:
        if v is None:
            values_clipped.append(None)
        else:
            values_clipped.append(v * scale)
    return values_clipped
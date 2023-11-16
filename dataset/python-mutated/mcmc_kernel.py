from abc import ABCMeta, abstractmethod

class MCMCKernel(object, metaclass=ABCMeta):

    def setup(self, warmup_steps, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Optional method to set up any state required at the start of the\n        simulation run.\n\n        :param int warmup_steps: Number of warmup iterations.\n        :param \\*args: Algorithm specific positional arguments.\n        :param \\*\\*kwargs: Algorithm specific keyword arguments.\n        '
        pass

    def cleanup(self):
        if False:
            return 10
        '\n        Optional method to clean up any residual state on termination.\n        '
        pass

    def logging(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Relevant logging information to be printed at regular intervals\n        of the MCMC run. Returns `None` by default.\n\n        :return: String containing the diagnostic summary. e.g. acceptance rate\n        :rtype: string\n        '
        return None

    def diagnostics(self):
        if False:
            print('Hello World!')
        '\n        Returns a dict of useful diagnostics after finishing sampling process.\n        '
        return {}

    def end_warmup(self):
        if False:
            while True:
                i = 10
        '\n        Optional method to tell kernel that warm-up phase has been finished.\n        '
        pass

    @property
    def initial_params(self):
        if False:
            return 10
        '\n        Returns a dict of initial params (by default, from the prior) to initiate the MCMC run.\n\n        :return: dict of parameter values keyed by their name.\n        '
        raise NotImplementedError

    @initial_params.setter
    def initial_params(self, params):
        if False:
            return 10
        '\n        Sets the parameters to initiate the MCMC run. Note that the parameters must\n        have unconstrained support.\n        '
        raise NotImplementedError

    @abstractmethod
    def sample(self, params):
        if False:
            while True:
                i = 10
        '\n        Samples parameters from the posterior distribution, when given existing parameters.\n\n        :param dict params: Current parameter values.\n        :param int time_step: Current time step.\n        :return: New parameters from the posterior distribution.\n        '
        raise NotImplementedError

    def __call__(self, params):
        if False:
            print('Hello World!')
        '\n        Alias for MCMCKernel.sample() method.\n        '
        return self.sample(params)
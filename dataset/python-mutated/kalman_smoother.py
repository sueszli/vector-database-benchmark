"""
State Space Representation and Kalman Filter, Smoother

Author: Chad Fulton
License: Simplified-BSD
"""
import numpy as np
from types import SimpleNamespace
from statsmodels.tsa.statespace.representation import OptionWrapper
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter, FilterResults
from statsmodels.tsa.statespace.tools import reorder_missing_matrix, reorder_missing_vector, copy_index_matrix
from statsmodels.tsa.statespace import tools, initialization
SMOOTHER_STATE = 1
SMOOTHER_STATE_COV = 2
SMOOTHER_DISTURBANCE = 4
SMOOTHER_DISTURBANCE_COV = 8
SMOOTHER_STATE_AUTOCOV = 16
SMOOTHER_ALL = SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE | SMOOTHER_DISTURBANCE_COV | SMOOTHER_STATE_AUTOCOV
SMOOTH_CONVENTIONAL = 1
SMOOTH_CLASSICAL = 2
SMOOTH_ALTERNATIVE = 4
SMOOTH_UNIVARIATE = 8

class KalmanSmoother(KalmanFilter):
    """
    State space representation of a time series process, with Kalman filter
    and smoother.

    Parameters
    ----------
    k_endog : {array_like, int}
        The observed time-series process :math:`y` if array like or the
        number of variables in the process if an integer.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int, optional
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation. Must be less than
        or equal to `k_states`. Default is `k_states`.
    results_class : class, optional
        Default results class to use to save filtering output. Default is
        `SmootherResults`. If specified, class must extend from
        `SmootherResults`.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices, for Kalman filtering options, or for Kalman smoothing
        options. See `Representation` for more details.
    """
    smoother_outputs = ['smoother_state', 'smoother_state_cov', 'smoother_state_autocov', 'smoother_disturbance', 'smoother_disturbance_cov', 'smoother_all']
    smoother_state = OptionWrapper('smoother_output', SMOOTHER_STATE)
    smoother_state_cov = OptionWrapper('smoother_output', SMOOTHER_STATE_COV)
    smoother_disturbance = OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE)
    smoother_disturbance_cov = OptionWrapper('smoother_output', SMOOTHER_DISTURBANCE_COV)
    smoother_state_autocov = OptionWrapper('smoother_output', SMOOTHER_STATE_AUTOCOV)
    smoother_all = OptionWrapper('smoother_output', SMOOTHER_ALL)
    smooth_methods = ['smooth_conventional', 'smooth_alternative', 'smooth_classical']
    smooth_conventional = OptionWrapper('smooth_method', SMOOTH_CONVENTIONAL)
    '\n    (bool) Flag for conventional (Durbin and Koopman, 2012) Kalman smoothing.\n    '
    smooth_alternative = OptionWrapper('smooth_method', SMOOTH_ALTERNATIVE)
    '\n    (bool) Flag for alternative (modified Bryson-Frazier) smoothing.\n    '
    smooth_classical = OptionWrapper('smooth_method', SMOOTH_CLASSICAL)
    '\n    (bool) Flag for classical (see e.g. Anderson and Moore, 1979) smoothing.\n    '
    smooth_univariate = OptionWrapper('smooth_method', SMOOTH_UNIVARIATE)
    '\n    (bool) Flag for univariate smoothing (uses modified Bryson-Frazier timing).\n    '
    smoother_output = SMOOTHER_ALL
    smooth_method = 0

    def __init__(self, k_endog, k_states, k_posdef=None, results_class=None, kalman_smoother_classes=None, **kwargs):
        if False:
            while True:
                i = 10
        if results_class is None:
            results_class = SmootherResults
        keys = ['smoother_output'] + KalmanSmoother.smoother_outputs
        smoother_output_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        keys = ['smooth_method'] + KalmanSmoother.smooth_methods
        smooth_method_kwargs = {key: kwargs.pop(key) for key in keys if key in kwargs}
        super(KalmanSmoother, self).__init__(k_endog, k_states, k_posdef, results_class=results_class, **kwargs)
        self.prefix_kalman_smoother_map = kalman_smoother_classes if kalman_smoother_classes is not None else tools.prefix_kalman_smoother_map.copy()
        self._kalman_smoothers = {}
        self.set_smoother_output(**smoother_output_kwargs)
        self.set_smooth_method(**smooth_method_kwargs)

    def _clone_kwargs(self, endog, **kwargs):
        if False:
            print('Hello World!')
        kwargs = super(KalmanSmoother, self)._clone_kwargs(endog, **kwargs)
        kwargs.setdefault('smoother_output', self.smoother_output)
        kwargs.setdefault('smooth_method', self.smooth_method)
        return kwargs

    @property
    def _kalman_smoother(self):
        if False:
            for i in range(10):
                print('nop')
        prefix = self.prefix
        if prefix in self._kalman_smoothers:
            return self._kalman_smoothers[prefix]
        return None

    def _initialize_smoother(self, smoother_output=None, smooth_method=None, prefix=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if smoother_output is None:
            smoother_output = self.smoother_output
        if smooth_method is None:
            smooth_method = self.smooth_method
        (prefix, dtype, create_filter, create_statespace) = self._initialize_filter(prefix, **kwargs)
        create_smoother = create_filter or prefix not in self._kalman_smoothers
        if not create_smoother:
            kalman_smoother = self._kalman_smoothers[prefix]
            create_smoother = kalman_smoother.kfilter is not self._kalman_filters[prefix]
        if create_smoother:
            cls = self.prefix_kalman_smoother_map[prefix]
            self._kalman_smoothers[prefix] = cls(self._statespaces[prefix], self._kalman_filters[prefix], smoother_output, smooth_method)
        else:
            self._kalman_smoothers[prefix].set_smoother_output(smoother_output, False)
            self._kalman_smoothers[prefix].set_smooth_method(smooth_method)
        return (prefix, dtype, create_smoother, create_filter, create_statespace)

    def set_smoother_output(self, smoother_output=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Set the smoother output\n\n        The smoother can produce several types of results. The smoother output\n        variable controls which are calculated and returned.\n\n        Parameters\n        ----------\n        smoother_output : int, optional\n            Bitmask value to set the smoother output to. See notes for details.\n        **kwargs\n            Keyword arguments may be used to influence the smoother output by\n            setting individual boolean flags. See notes for details.\n\n        Notes\n        -----\n        The smoother output is defined by a collection of boolean flags, and\n        is internally stored as a bitmask. The methods available are:\n\n        SMOOTHER_STATE = 0x01\n            Calculate and return the smoothed states.\n        SMOOTHER_STATE_COV = 0x02\n            Calculate and return the smoothed state covariance matrices.\n        SMOOTHER_STATE_AUTOCOV = 0x10\n            Calculate and return the smoothed state lag-one autocovariance\n            matrices.\n        SMOOTHER_DISTURBANCE = 0x04\n            Calculate and return the smoothed state and observation\n            disturbances.\n        SMOOTHER_DISTURBANCE_COV = 0x08\n            Calculate and return the covariance matrices for the smoothed state\n            and observation disturbances.\n        SMOOTHER_ALL\n            Calculate and return all results.\n\n        If the bitmask is set directly via the `smoother_output` argument, then\n        the full method must be provided.\n\n        If keyword arguments are used to set individual boolean flags, then\n        the lowercase of the method must be used as an argument name, and the\n        value is the desired value of the boolean flag (True or False).\n\n        Note that the smoother output may also be specified by directly\n        modifying the class attributes which are defined similarly to the\n        keyword arguments.\n\n        The default smoother output is SMOOTHER_ALL.\n\n        If performance is a concern, only those results which are needed should\n        be specified as any results that are not specified will not be\n        calculated. For example, if the smoother output is set to only include\n        SMOOTHER_STATE, the smoother operates much more quickly than if all\n        output is required.\n\n        Examples\n        --------\n        >>> import statsmodels.tsa.statespace.kalman_smoother as ks\n        >>> mod = ks.KalmanSmoother(1,1)\n        >>> mod.smoother_output\n        15\n        >>> mod.set_smoother_output(smoother_output=0)\n        >>> mod.smoother_state = True\n        >>> mod.smoother_output\n        1\n        >>> mod.smoother_state\n        True\n        '
        if smoother_output is not None:
            self.smoother_output = smoother_output
        for name in KalmanSmoother.smoother_outputs:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def set_smooth_method(self, smooth_method=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the smoothing method\n\n        The smoothing method can be used to override the Kalman smoother\n        approach used. By default, the Kalman smoother used depends on the\n        Kalman filter method.\n\n        Parameters\n        ----------\n        smooth_method : int, optional\n            Bitmask value to set the filter method to. See notes for details.\n        **kwargs\n            Keyword arguments may be used to influence the filter method by\n            setting individual boolean flags. See notes for details.\n\n        Notes\n        -----\n        The smoothing method is defined by a collection of boolean flags, and\n        is internally stored as a bitmask. The methods available are:\n\n        SMOOTH_CONVENTIONAL = 0x01\n            Default Kalman smoother, as presented in Durbin and Koopman, 2012\n            chapter 4.\n        SMOOTH_CLASSICAL = 0x02\n            Classical Kalman smoother, as presented in Anderson and Moore, 1979\n            or Durbin and Koopman, 2012 chapter 4.6.1.\n        SMOOTH_ALTERNATIVE = 0x04\n            Modified Bryson-Frazier Kalman smoother method; this is identical\n            to the conventional method of Durbin and Koopman, 2012, except that\n            an additional intermediate step is included.\n        SMOOTH_UNIVARIATE = 0x08\n            Univariate Kalman smoother, as presented in Durbin and Koopman,\n            2012 chapter 6, except with modified Bryson-Frazier timing.\n\n        Practically speaking, these methods should all produce the same output\n        but different computational implications, numerical stability\n        implications, or internal timing assumptions.\n\n        Note that only the first method is available if using a Scipy version\n        older than 0.16.\n\n        If the bitmask is set directly via the `smooth_method` argument, then\n        the full method must be provided.\n\n        If keyword arguments are used to set individual boolean flags, then\n        the lowercase of the method must be used as an argument name, and the\n        value is the desired value of the boolean flag (True or False).\n\n        Note that the filter method may also be specified by directly modifying\n        the class attributes which are defined similarly to the keyword\n        arguments.\n\n        The default filtering method is SMOOTH_CONVENTIONAL.\n\n        Examples\n        --------\n        >>> mod = sm.tsa.statespace.SARIMAX(range(10))\n        >>> mod.smooth_method\n        1\n        >>> mod.filter_conventional\n        True\n        >>> mod.filter_univariate = True\n        >>> mod.smooth_method\n        17\n        >>> mod.set_smooth_method(filter_univariate=False,\n                                  filter_collapsed=True)\n        >>> mod.smooth_method\n        33\n        >>> mod.set_smooth_method(smooth_method=1)\n        >>> mod.filter_conventional\n        True\n        >>> mod.filter_univariate\n        False\n        >>> mod.filter_collapsed\n        False\n        >>> mod.filter_univariate = True\n        >>> mod.smooth_method\n        17\n        '
        if smooth_method is not None:
            self.smooth_method = smooth_method
        for name in KalmanSmoother.smooth_methods:
            if name in kwargs:
                setattr(self, name, kwargs[name])

    def _smooth(self, smoother_output=None, smooth_method=None, prefix=None, complex_step=False, results=None, **kwargs):
        if False:
            i = 10
            return i + 15
        (prefix, dtype, create_smoother, create_filter, create_statespace) = self._initialize_smoother(smoother_output, smooth_method, prefix=prefix, **kwargs)
        if create_filter or create_statespace:
            raise ValueError('Passed settings forced re-creation of the Kalman filter. Please run `_filter` before running `_smooth`.')
        smoother = self._kalman_smoothers[prefix]
        smoother()
        return smoother

    def smooth(self, smoother_output=None, smooth_method=None, results=None, run_filter=True, prefix=None, complex_step=False, update_representation=True, update_filter=True, update_smoother=True, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Apply the Kalman smoother to the statespace model.\n\n        Parameters\n        ----------\n        smoother_output : int, optional\n            Determines which Kalman smoother output calculate. Default is all\n            (including state, disturbances, and all covariances).\n        results : class or object, optional\n            If a class, then that class is instantiated and returned with the\n            result of both filtering and smoothing.\n            If an object, then that object is updated with the smoothing data.\n            If None, then a SmootherResults object is returned with both\n            filtering and smoothing results.\n        run_filter : bool, optional\n            Whether or not to run the Kalman filter prior to smoothing. Default\n            is True.\n        prefix : str\n            The prefix of the datatype. Usually only used internally.\n\n        Returns\n        -------\n        SmootherResults object\n        '
        kfilter = self._filter(**kwargs)
        results = self.results_class(self)
        if update_representation:
            results.update_representation(self)
        if update_filter:
            results.update_filter(kfilter)
        else:
            results.nobs_diffuse = kfilter.nobs_diffuse
        if smoother_output is None:
            smoother_output = self.smoother_output
        smoother = self._smooth(smoother_output, results=results, **kwargs)
        if update_smoother:
            results.update_smoother(smoother)
        return results

class SmootherResults(FilterResults):
    """
    Results from applying the Kalman smoother and/or filter to a state space
    model.

    Parameters
    ----------
    model : Representation
        A Statespace representation

    Attributes
    ----------
    nobs : int
        Number of observations.
    k_endog : int
        The dimension of the observation series.
    k_states : int
        The dimension of the unobserved state process.
    k_posdef : int
        The dimension of a guaranteed positive definite covariance matrix
        describing the shocks in the measurement equation.
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    shapes : dictionary of name:tuple
        A dictionary recording the shapes of each of the representation
        matrices as tuples.
    endog : ndarray
        The observation vector.
    design : ndarray
        The design matrix, :math:`Z`.
    obs_intercept : ndarray
        The intercept for the observation equation, :math:`d`.
    obs_cov : ndarray
        The covariance matrix for the observation equation :math:`H`.
    transition : ndarray
        The transition matrix, :math:`T`.
    state_intercept : ndarray
        The intercept for the transition equation, :math:`c`.
    selection : ndarray
        The selection matrix, :math:`R`.
    state_cov : ndarray
        The covariance matrix for the state equation :math:`Q`.
    missing : array of bool
        An array of the same size as `endog`, filled with boolean values that
        are True if the corresponding entry in `endog` is NaN and False
        otherwise.
    nmissing : array of int
        An array of size `nobs`, where the ith entry is the number (between 0
        and k_endog) of NaNs in the ith row of the `endog` array.
    time_invariant : bool
        Whether or not the representation matrices are time-invariant
    initialization : str
        Kalman filter initialization method.
    initial_state : array_like
        The state vector used to initialize the Kalamn filter.
    initial_state_cov : array_like
        The state covariance matrix used to initialize the Kalamn filter.
    filter_method : int
        Bitmask representing the Kalman filtering method
    inversion_method : int
        Bitmask representing the method used to invert the forecast error
        covariance matrix.
    stability_method : int
        Bitmask representing the methods used to promote numerical stability in
        the Kalman filter recursions.
    conserve_memory : int
        Bitmask representing the selected memory conservation method.
    tolerance : float
        The tolerance at which the Kalman filter determines convergence to
        steady-state.
    loglikelihood_burn : int
        The number of initial periods during which the loglikelihood is not
        recorded.
    converged : bool
        Whether or not the Kalman filter converged.
    period_converged : int
        The time period in which the Kalman filter converged.
    filtered_state : ndarray
        The filtered state vector at each time period.
    filtered_state_cov : ndarray
        The filtered state covariance matrix at each time period.
    predicted_state : ndarray
        The predicted state vector at each time period.
    predicted_state_cov : ndarray
        The predicted state covariance matrix at each time period.
    kalman_gain : ndarray
        The Kalman gain at each time period.
    forecasts : ndarray
        The one-step-ahead forecasts of observations at each time period.
    forecasts_error : ndarray
        The forecast errors at each time period.
    forecasts_error_cov : ndarray
        The forecast error covariance matrices at each time period.
    loglikelihood : ndarray
        The loglikelihood values at each time period.
    collapsed_forecasts : ndarray
        If filtering using collapsed observations, stores the one-step-ahead
        forecasts of collapsed observations at each time period.
    collapsed_forecasts_error : ndarray
        If filtering using collapsed observations, stores the one-step-ahead
        forecast errors of collapsed observations at each time period.
    collapsed_forecasts_error_cov : ndarray
        If filtering using collapsed observations, stores the one-step-ahead
        forecast error covariance matrices of collapsed observations at each
        time period.
    standardized_forecast_error : ndarray
        The standardized forecast errors
    smoother_output : int
        Bitmask representing the generated Kalman smoothing output
    scaled_smoothed_estimator : ndarray
        The scaled smoothed estimator at each time period.
    scaled_smoothed_estimator_cov : ndarray
        The scaled smoothed estimator covariance matrices at each time period.
    smoothing_error : ndarray
        The smoothing error covariance matrices at each time period.
    smoothed_state : ndarray
        The smoothed state at each time period.
    smoothed_state_cov : ndarray
        The smoothed state covariance matrices at each time period.
    smoothed_state_autocov : ndarray
        The smoothed state lago-one autocovariance matrices at each time
        period: :math:`Cov(\\alpha_{t+1}, \\alpha_t)`.
    smoothed_measurement_disturbance : ndarray
        The smoothed measurement at each time period.
    smoothed_state_disturbance : ndarray
        The smoothed state at each time period.
    smoothed_measurement_disturbance_cov : ndarray
        The smoothed measurement disturbance covariance matrices at each time
        period.
    smoothed_state_disturbance_cov : ndarray
        The smoothed state disturbance covariance matrices at each time period.
    """
    _smoother_attributes = ['smoother_output', 'scaled_smoothed_estimator', 'scaled_smoothed_estimator_cov', 'smoothing_error', 'smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov', 'smoothed_measurement_disturbance', 'smoothed_state_disturbance', 'smoothed_measurement_disturbance_cov', 'smoothed_state_disturbance_cov', 'innovations_transition']
    _smoother_options = KalmanSmoother.smoother_outputs
    _attributes = FilterResults._model_attributes + _smoother_attributes

    def update_representation(self, model, only_options=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the results to match a given model\n\n        Parameters\n        ----------\n        model : Representation\n            The model object from which to take the updated values.\n        only_options : bool, optional\n            If set to true, only the smoother and filter options are updated,\n            and the state space representation is not updated. Default is\n            False.\n\n        Notes\n        -----\n        This method is rarely required except for internal usage.\n        '
        super(SmootherResults, self).update_representation(model, only_options)
        for name in self._smoother_options:
            setattr(self, name, getattr(model, name, None))
        self._smoothed_forecasts = None
        self._smoothed_forecasts_error = None
        self._smoothed_forecasts_error_cov = None

    def update_smoother(self, smoother):
        if False:
            return 10
        '\n        Update the smoother results\n\n        Parameters\n        ----------\n        smoother : KalmanSmoother\n            The model object from which to take the updated values.\n\n        Notes\n        -----\n        This method is rarely required except for internal usage.\n        '
        attributes = []
        if self.smoother_state or self.smoother_disturbance:
            attributes.append('scaled_smoothed_estimator')
        if self.smoother_state_cov or self.smoother_disturbance_cov:
            attributes.append('scaled_smoothed_estimator_cov')
        if self.smoother_state:
            attributes.append('smoothed_state')
        if self.smoother_state_cov:
            attributes.append('smoothed_state_cov')
        if self.smoother_state_autocov:
            attributes.append('smoothed_state_autocov')
        if self.smoother_disturbance:
            attributes += ['smoothing_error', 'smoothed_measurement_disturbance', 'smoothed_state_disturbance']
        if self.smoother_disturbance_cov:
            attributes += ['smoothed_measurement_disturbance_cov', 'smoothed_state_disturbance_cov']
        has_missing = np.sum(self.nmissing) > 0
        for name in self._smoother_attributes:
            if name == 'smoother_output':
                pass
            elif name in attributes:
                if name in ['smoothing_error', 'smoothed_measurement_disturbance']:
                    vector = getattr(smoother, name, None)
                    if vector is not None and has_missing:
                        vector = np.array(reorder_missing_vector(vector, self.missing, prefix=self.prefix))
                    else:
                        vector = np.array(vector, copy=True)
                    setattr(self, name, vector)
                elif name == 'smoothed_measurement_disturbance_cov':
                    matrix = getattr(smoother, name, None)
                    if matrix is not None and has_missing:
                        matrix = reorder_missing_matrix(matrix, self.missing, reorder_rows=True, reorder_cols=True, prefix=self.prefix)
                        copy_index_matrix(self.obs_cov, matrix, self.missing, index_rows=True, index_cols=True, inplace=True, prefix=self.prefix)
                    else:
                        matrix = np.array(matrix, copy=True)
                    setattr(self, name, matrix)
                else:
                    setattr(self, name, np.array(getattr(smoother, name, None), copy=True))
            else:
                setattr(self, name, None)
        self.innovations_transition = np.array(smoother.innovations_transition, copy=True)
        self.scaled_smoothed_diffuse_estimator = None
        self.scaled_smoothed_diffuse1_estimator_cov = None
        self.scaled_smoothed_diffuse2_estimator_cov = None
        if self.nobs_diffuse > 0:
            self.scaled_smoothed_diffuse_estimator = np.array(smoother.scaled_smoothed_diffuse_estimator, copy=True)
            self.scaled_smoothed_diffuse1_estimator_cov = np.array(smoother.scaled_smoothed_diffuse1_estimator_cov, copy=True)
            self.scaled_smoothed_diffuse2_estimator_cov = np.array(smoother.scaled_smoothed_diffuse2_estimator_cov, copy=True)
        start = 1
        end = None
        if 'scaled_smoothed_estimator' in attributes:
            self.scaled_smoothed_estimator_presample = self.scaled_smoothed_estimator[:, 0]
            self.scaled_smoothed_estimator = self.scaled_smoothed_estimator[:, start:end]
        if 'scaled_smoothed_estimator_cov' in attributes:
            self.scaled_smoothed_estimator_cov_presample = self.scaled_smoothed_estimator_cov[:, :, 0]
            self.scaled_smoothed_estimator_cov = self.scaled_smoothed_estimator_cov[:, :, start:end]
        self._smoothed_forecasts = None
        self._smoothed_forecasts_error = None
        self._smoothed_forecasts_error_cov = None
        if self.filter_concentrated and self.model._scale is None:
            self.smoothed_state_cov *= self.scale
            self.smoothed_state_autocov *= self.scale
            self.smoothed_state_disturbance_cov *= self.scale
            self.smoothed_measurement_disturbance_cov *= self.scale
            self.scaled_smoothed_estimator_presample /= self.scale
            self.scaled_smoothed_estimator /= self.scale
            self.scaled_smoothed_estimator_cov_presample /= self.scale
            self.scaled_smoothed_estimator_cov /= self.scale
            self.smoothing_error /= self.scale
        self.__smoothed_state_autocovariance = {}

    def _smoothed_state_autocovariance(self, shift, start, end, extend_kwargs=None):
        if False:
            print('Hello World!')
        '\n        Compute "forward" autocovariances, Cov(t, t+j)\n\n        Parameters\n        ----------\n        shift : int\n            The number of period to shift forwards when computing the\n            autocovariance. This has the opposite sign as `lag` from the\n            `smoothed_state_autocovariance` method.\n        start : int, optional\n            The start of the interval (inclusive) of autocovariances to compute\n            and return.\n        end : int, optional\n            The end of the interval (exclusive) autocovariances to compute and\n            return. Note that since it is an exclusive endpoint, the returned\n            autocovariances do not include the value at this index.\n        extend_kwargs : dict, optional\n            Keyword arguments containing updated state space system matrices\n            for handling out-of-sample autocovariance computations in\n            time-varying state space models.\n\n        '
        if extend_kwargs is None:
            extend_kwargs = {}
        n = end - start
        if shift == 0:
            max_insample = self.nobs - shift
        else:
            max_insample = self.nobs - shift + 1
        n_postsample = max(0, end - max_insample)
        if shift != 0:
            L = self.innovations_transition
            P = self.predicted_state_cov
            N = self.scaled_smoothed_estimator_cov
        else:
            acov = self.smoothed_state_cov
        if n_postsample > 0:
            endog = np.zeros((n_postsample, self.k_endog)) * np.nan
            mod = self.model.extend(endog, start=self.nobs, **extend_kwargs)
            mod.initialize_known(self.predicted_state[..., self.nobs], self.predicted_state_cov[..., self.nobs])
            res = mod.smooth()
            if shift != 0:
                start_insample = max(0, start)
                L = np.concatenate((L[..., start_insample:], res.innovations_transition), axis=2)
                P = np.concatenate((P[..., start_insample:], res.predicted_state_cov[..., 1:]), axis=2)
                N = np.concatenate((N[..., start_insample:], res.scaled_smoothed_estimator_cov), axis=2)
                end -= start_insample
                start -= start_insample
            else:
                acov = np.concatenate((acov, res.predicted_state_cov), axis=2)
        if shift != 0:
            start_insample = max(0, start)
            LT = L[..., start_insample:end + shift - 1].T
            P = P[..., start_insample:end + shift].T
            N = N[..., start_insample:end + shift - 1].T
            tmpLT = np.eye(self.k_states)[None, :, :]
            length = P.shape[0] - shift
            for i in range(1, shift + 1):
                tmpLT = LT[shift - i:length + shift - i] @ tmpLT
            eye = np.eye(self.k_states)[None, ...]
            acov = np.zeros((n, self.k_states, self.k_states))
            acov[:start_insample - start] = np.nan
            acov[start_insample - start:] = P[:-shift] @ tmpLT @ (eye - N[shift - 1:] @ P[shift:])
        else:
            acov = acov.T[start:end]
        return acov

    def smoothed_state_autocovariance(self, lag=1, t=None, start=None, end=None, extend_kwargs=None):
        if False:
            while True:
                i = 10
        '\n        Compute state vector autocovariances, conditional on the full dataset\n\n        Computes:\n\n        .. math::\n\n            Cov(\\alpha_t - \\hat \\alpha_t, \\alpha_{t - j} - \\hat \\alpha_{t - j})\n\n        where the `lag` argument gives the value for :math:`j`. Thus when\n        the `lag` argument is positive, the autocovariance is between the\n        current and previous periods, while if `lag` is negative the\n        autocovariance is between the current and future periods.\n\n        Parameters\n        ----------\n        lag : int, optional\n            The number of period to shift when computing the autocovariance.\n            Default is 1.\n        t : int, optional\n            A specific period for which to compute and return the\n            autocovariance. Cannot be used in combination with `start` or\n            `end`. See the Returns section for details on how this\n            parameter affects what is what is returned.\n        start : int, optional\n            The start of the interval (inclusive) of autocovariances to compute\n            and return. Cannot be used in combination with the `t` argument.\n            See the Returns section for details on how this parameter affects\n            what is what is returned. Default is 0.\n        end : int, optional\n            The end of the interval (exclusive) autocovariances to compute and\n            return. Note that since it is an exclusive endpoint, the returned\n            autocovariances do not include the value at this index. Cannot be\n            used in combination with the `t` argument. See the Returns section\n            for details on how this parameter affects what is what is returned\n            and what the default value is.\n        extend_kwargs : dict, optional\n            Keyword arguments containing updated state space system matrices\n            for handling out-of-sample autocovariance computations in\n            time-varying state space models.\n\n        Returns\n        -------\n        acov : ndarray\n            Array of autocovariance matrices. If the argument `t` is not\n            provided, then it is shaped `(k_states, k_states, n)`, while if `t`\n            given then the third axis is dropped and the array is shaped\n            `(k_states, k_states)`.\n\n            The output under the default case differs somewhat based on the\n            state space model and the sign of the lag. To see how these cases\n            differ, denote the output at each time point as Cov(t, t-j). Then:\n\n            - If `lag > 0` (and the model is either time-varying or\n              time-invariant), then the returned array is shaped `(*, *, nobs)`\n              and each entry [:, :, t] contains Cov(t, t-j). However, the model\n              does not have enough information to compute autocovariances in\n              the pre-sample period, so that we cannot compute Cov(1, 1-lag),\n              Cov(2, 2-lag), ..., Cov(lag, 0). Thus the first `lag` entries\n              have all values set to NaN.\n\n            - If the model is time-invariant and `lag < -1` or if `lag` is\n              0 or -1, and the model is either time-invariant or time-varying,\n              then the returned array is shaped `(*, *, nobs)` and each\n              entry [:, :, t] contains Cov(t, t+j). Moreover, all entries are\n              available (i.e. there are no NaNs).\n\n            - If the model is time-varying and `lag < -1` and `extend_kwargs`\n              is not provided, then the returned array is shaped\n              `(*, *, nobs - lag + 1)`.\n\n            - However, if the model is time-varying and `lag < -1`, then\n              `extend_kwargs` can be provided with `lag - 1` additional\n              matrices so that the returned array is shaped `(*, *, nobs)` as\n              usual.\n\n            More generally, the dimension of the last axis will be\n            `start - end`.\n\n        Notes\n        -----\n        This method computes:\n\n        .. math::\n\n            Cov(\\alpha_t - \\hat \\alpha_t, \\alpha_{t - j} - \\hat \\alpha_{t - j})\n\n        where the `lag` argument determines the autocovariance order :math:`j`,\n        and `lag` is an integer (positive, zero, or negative). This method\n        cannot compute values associated with time points prior to the sample,\n        and so it returns a matrix of NaN values for these time points.\n        For example, if `start=0` and `lag=2`, then assuming the output is\n        assigned to the variable `acov`, we will have `acov[..., 0]` and\n        `acov[..., 1]` as matrices filled with NaN values.\n\n        Based only on the "current" results object (i.e. the Kalman smoother\n        applied to the sample), there is not enough information to compute\n        Cov(t, t+j) for the last `lag - 1` observations of the sample. However,\n        the values can be computed for these time points using the transition\n        equation of the state space representation, and so for time-invariant\n        state space models we do compute these values. For time-varying models,\n        this can also be done, but updated state space matrices for the\n        out-of-sample time points must be provided via the `extend_kwargs`\n        argument.\n\n        See [1]_, Chapter 4.7, for all details about how these autocovariances\n        are computed.\n\n        The `t` and `start`/`end` parameters compute and return only the\n        requested autocovariances. As a result, using these parameters is\n        recommended to reduce the computational burden, particularly if the\n        number of observations and/or the dimension of the state vector is\n        large.\n\n        References\n        ----------\n        .. [1] Durbin, James, and Siem Jan Koopman. 2012.\n               Time Series Analysis by State Space Methods: Second Edition.\n               Oxford University Press.\n        '
        cache_key = None
        if extend_kwargs is None or len(extend_kwargs) == 0:
            cache_key = (lag, t, start, end)
        if cache_key is not None and cache_key in self.__smoothed_state_autocovariance:
            return self.__smoothed_state_autocovariance[cache_key]
        forward_autocovariances = False
        if lag < 0:
            lag = -lag
            forward_autocovariances = True
        if t is not None and (start is not None or end is not None):
            raise ValueError('Cannot specify both `t` and `start` or `end`.')
        if t is not None:
            start = t
            end = t + 1
        if start is None:
            start = 0
        if end is None:
            if forward_autocovariances and lag > 1 and (extend_kwargs is None):
                end = self.nobs - lag + 1
            else:
                end = self.nobs
        if extend_kwargs is None:
            extend_kwargs = {}
        if start < 0 or end < 0:
            raise ValueError('Negative `t`, `start`, or `end` is not allowed.')
        if end < start:
            raise ValueError('`end` must be after `start`')
        if lag == 0 and self.smoothed_state_cov is None:
            raise RuntimeError('Cannot return smoothed state covariances if those values have not been computed by Kalman smoothing.')
        if lag == 0 and end <= self.nobs + 1:
            acov = self.smoothed_state_cov
            if end == self.nobs + 1:
                acov = np.concatenate((acov[..., start:], self.predicted_state_cov[..., -1:]), axis=2).T
            else:
                acov = acov.T[start:end]
        elif lag == 1 and self.smoothed_state_autocov is not None and (not forward_autocovariances) and (end <= self.nobs + 1):
            if start == 0:
                nans = np.zeros((self.k_states, self.k_states, lag)) * np.nan
                acov = np.concatenate((nans, self.smoothed_state_autocov[..., :end - 1]), axis=2)
            else:
                acov = self.smoothed_state_autocov[..., start - 1:end - 1]
            acov = acov.transpose(2, 0, 1)
        elif lag == 1 and self.smoothed_state_autocov is not None and forward_autocovariances and (end < self.nobs + 1):
            acov = self.smoothed_state_autocov.T[start:end]
        elif forward_autocovariances:
            acov = self._smoothed_state_autocovariance(lag, start, end, extend_kwargs=extend_kwargs)
        else:
            out = self._smoothed_state_autocovariance(lag, start - lag, end - lag, extend_kwargs=extend_kwargs)
            acov = out.transpose(0, 2, 1)
        if t is not None:
            acov = acov[0]
        else:
            acov = acov.transpose(1, 2, 0)
        if cache_key is not None:
            self.__smoothed_state_autocovariance[cache_key] = acov
        return acov

    def news(self, previous, t=None, start=None, end=None, revisions_details_start=True, design=None, state_index=None):
        if False:
            i = 10
            return i + 15
        '\n        Compute the news and impacts associated with a data release\n\n        Parameters\n        ----------\n        previous : SmootherResults\n            Prior results object relative to which to compute the news. This\n            results object must have identical state space representation for\n            the prior sample period so that the only difference is that this\n            results object has updates to the observed data.\n        t : int, optional\n            A specific period for which to compute the news. Cannot be used in\n            combination with `start` or `end`.\n        start : int, optional\n            The start of the interval (inclusive) of news to compute. Cannot be\n            used in combination with the `t` argument. Default is the last\n            period of the sample (`nobs - 1`).\n        end : int, optional\n            The end of the interval (exclusive) of news to compute. Note that\n            since it is an exclusive endpoint, the returned news do not include\n            the value at this index. Cannot be used in combination with the `t`\n            argument.\n        revisions_details_start : bool or int, optional\n            The period at which to beging computing the detailed impacts of\n            data revisions. Any revisions prior to this period will have their\n            impacts grouped together. If a negative integer, interpreted as\n            an offset from the end of the dataset. If set to True, detailed\n            impacts are computed for all revisions, while if set to False, all\n            revisions are grouped together. Default is False. Note that for\n            large models, setting this to be near the beginning of the sample\n            can cause this function to be slow.\n        design : array, optional\n            Design matrix for the period `t` in time-varying models. If this\n            model has a time-varying design matrix, and the argument `t` is out\n            of this model\'s sample, then a new design matrix for period `t`\n            must be provided. Unused otherwise.\n        state_index : array_like, optional\n            An optional index specifying a subset of states to use when\n            constructing the impacts of revisions and news. For example, if\n            `state_index=[0, 1]` is passed, then only the impacts to the\n            observed variables arising from the impacts to the first two\n            states will be returned.\n\n        Returns\n        -------\n        news_results : SimpleNamespace\n            News and impacts associated with a data release. Includes the\n            following attributes:\n\n            - `update_impacts`: update to forecasts of impacted variables from\n              the news. It is equivalent to E[y^i | post] - E[y^i | revision],\n              where y^i are the variables of interest. In [1]_, this is\n              described as "revision" in equation (17).\n            - `revision_detailed_impacts`: update to forecasts of variables\n              impacted variables from data revisions. It is\n              E[y^i | revision] - E[y^i | previous], and does not have a\n              specific notation in [1]_, since there for simplicity they assume\n              that there are no revisions.\n            - `news`: the unexpected component of the updated data. Denoted\n              I = y^u - E[y^u | previous], where y^u are the data points that\n              were newly incorporated in a data release (but not including\n              revisions to data points that already existed in the previous\n              release). In [1]_, this is described as "news" in equation (17).\n            - `revisions`: y^r(updated) - y^r(previous) for periods in\n              which detailed impacts were computed\n            - `revisions_all` : y^r(updated) - y^r(previous) for all revisions\n            - `gain`: the gain matrix associated with the "Kalman-like" update\n              from the news, E[y I\'] E[I I\']^{-1}. In [1]_, this can be found\n              in the equation For E[y_{k,t_k} \\mid I_{v+1}] in the middle of\n              page 17.\n            - `revision_weights` weights on observations for the smoothed\n              signal\n            - `update_forecasts`: forecasts of the updated periods used to\n              construct the news, E[y^u | previous].\n            - `update_realized`: realizations of the updated periods used to\n              construct the news, y^u.\n            - `revised`: revised observations of the periods that were revised\n              and for which detailed impacts were computed\n            - `revised`: revised observations of the periods that were revised\n            - `revised_prev`: previous observations of the periods that were\n              revised and for which detailed impacts were computed\n            - `revised_prev_all`: previous observations of the periods that\n              were revised and for which detailed impacts were computed\n            - `prev_impacted_forecasts`: previous forecast of the periods of\n              interest, E[y^i | previous].\n            - `post_impacted_forecasts`: forecast of the periods of interest\n              after taking into account both revisions and updates,\n              E[y^i | post].\n            - `revision_results`: results object that updates the `previous`\n              results to take into account data revisions.\n            - `revision_results`: results object associated with the revisions\n            - `revision_impacts`: total impacts from all revisions (both\n              grouped and detailed)\n            - `revisions_ix`: list of `(t, i)` positions of revisions in endog\n            - `revisions_details`: list of `(t, i)` positions of revisions to\n              endog for which details of impacts were computed\n            - `revisions_grouped`: list of `(t, i)` positions of revisions to\n              endog for which impacts were grouped\n            - `revisions_details_start`: period in which revision details start\n              to be computed\n            - `updates_ix`: list of `(t, i)` positions of updates to endog\n            - `state_index`: index of state variables used to compute impacts\n\n        Notes\n        -----\n        This method computes the effect of new data (e.g. from a new data\n        release) on smoothed forecasts produced by a state space model, as\n        described in [1]_. It also computes the effect of revised data on\n        smoothed forecasts.\n\n        References\n        ----------\n        .. [1] Bańbura, Marta and Modugno, Michele. 2010.\n               "Maximum likelihood estimation of factor models on data sets\n               with arbitrary pattern of missing data."\n               No 1189, Working Paper Series, European Central Bank.\n               https://EconPapers.repec.org/RePEc:ecb:ecbwps:20101189.\n        .. [2] Bańbura, Marta, and Michele Modugno.\n               "Maximum likelihood estimation of factor models on datasets with\n               arbitrary pattern of missing data."\n               Journal of Applied Econometrics 29, no. 1 (2014): 133-160.\n\n        '
        if t is not None and (start is not None or end is not None):
            raise ValueError('Cannot specify both `t` and `start` or `end`.')
        if t is not None:
            start = t
            end = t + 1
        if start is None:
            start = self.nobs - 1
        if end is None:
            end = self.nobs
        if start < 0 or end < 0:
            raise ValueError('Negative `t`, `start`, or `end` is not allowed.')
        if end <= start:
            raise ValueError('`end` must be after `start`')
        if self.smoothed_state_cov is None:
            raise ValueError('Cannot compute news without having applied the Kalman smoother first.')
        error_ss = 'This results object has %s and so it does not appear to by an extension of `previous`. Can only compute the news by comparing this results set to previous results objects.'
        if self.nobs < previous.nobs:
            raise ValueError(error_ss % 'fewer observations than `previous`')
        if not (self.k_endog == previous.k_endog and self.k_states == previous.k_states and (self.k_posdef == previous.k_posdef)):
            raise ValueError(error_ss % 'different state space dimensions than `previous`')
        for key in self.model.shapes.keys():
            if key == 'obs':
                continue
            tv = getattr(self, key).shape[-1] > 1
            tv_prev = getattr(previous, key).shape[-1] > 1
            if tv and (not tv_prev):
                raise ValueError(error_ss % f'time-varying {key} while `previous` does not')
            if not tv and tv_prev:
                raise ValueError(error_ss % f'time-invariant {key} while `previous` does not')
        if state_index is not None:
            state_index = np.atleast_1d(np.sort(np.array(state_index, dtype=int)))
        if end > self.nobs and (not self.model.time_invariant):
            raise RuntimeError('Cannot compute the impacts of news on periods outside of the sample in time-varying models.')
        extend_kwargs = {}
        for key in self.model.shapes.keys():
            if key == 'obs':
                continue
            mat = getattr(self, key)
            prev_mat = getattr(previous, key)
            if mat.shape[-1] > prev_mat.shape[-1]:
                extend_kwargs[key] = mat[..., prev_mat.shape[-1]:]
        (revisions_ix, updates_ix) = previous.model.diff_endog(self.endog.T)
        prev_impacted_forecasts = previous.predict(start=start, end=end, **extend_kwargs).smoothed_forecasts
        post_impacted_forecasts = self.predict(start=start, end=end).smoothed_forecasts
        if revisions_details_start is True:
            revisions_details_start = 0
        elif revisions_details_start is False:
            revisions_details_start = previous.nobs
        elif revisions_details_start < 0:
            revisions_details_start = previous.nobs + revisions_details_start
        revisions_grouped = []
        revisions_details = []
        if revisions_details_start > 0:
            for (s, i) in revisions_ix:
                if s < revisions_details_start:
                    revisions_grouped.append((s, i))
                else:
                    revisions_details.append((s, i))
        else:
            revisions_details = revisions_ix
        if len(revisions_ix) > 0:
            revisions_details_start = max(revisions_ix[0][0], revisions_details_start)
        revised_endog = None
        revised_all = None
        revised_prev_all = None
        revisions_all = None
        revised = None
        revised_prev = None
        revisions = None
        revision_weights = None
        revision_detailed_impacts = None
        revision_results = None
        revision_impacts = None
        if len(revisions_ix) > 0:
            (revised_j, revised_p) = zip(*revisions_ix)
            compute_j = np.arange(revised_j[0], revised_j[-1] + 1)
            revised_endog = self.endog[:, :previous.nobs].copy()
            revised_endog[previous.missing.astype(bool)] = np.nan
            revised_all = revised_endog.T[compute_j]
            revised_prev_all = previous.endog.T[compute_j]
            revisions_all = revised_all - revised_prev_all
            tmp_endog = revised_endog.T.copy()
            tmp_nobs = max(end, previous.nobs)
            oos_nobs = tmp_nobs - previous.nobs
            if oos_nobs > 0:
                tmp_endog = np.concatenate([tmp_endog, np.zeros((oos_nobs, self.k_endog)) * np.nan], axis=0)
            clone_kwargs = {}
            for key in self.model.shapes.keys():
                if key == 'obs':
                    continue
                mat = getattr(self, key)
                if mat.shape[-1] > 1:
                    clone_kwargs[key] = mat[..., :tmp_nobs]
            rev_mod = previous.model.clone(tmp_endog, **clone_kwargs)
            init = initialization.Initialization.from_results(self)
            rev_mod.initialize(init)
            revision_results = rev_mod.smooth()
            if len(revisions_details) > 0:
                compute_j = np.arange(revisions_details_start, revised_j[-1] + 1)
                offset = revisions_details_start - revised_j[0]
                revised = revised_all[offset:]
                revised_prev = revised_prev_all[offset:]
                revisions = revisions_all[offset:]
                compute_t = np.arange(start, end)
                (smoothed_state_weights, _, _) = tools._compute_smoothed_state_weights(rev_mod, compute_t=compute_t, compute_j=compute_j, compute_prior_weights=False, scale=previous.scale)
                ZT = rev_mod.design.T
                if ZT.shape[0] > 1:
                    ZT = ZT[compute_t]
                if state_index is not None:
                    ZT = ZT[:, state_index, :]
                    smoothed_state_weights = smoothed_state_weights[:, :, state_index]
                revision_weights = np.nansum(smoothed_state_weights[..., None] * ZT[:, None, :, None, :], axis=2).transpose(0, 1, 3, 2)
                (revised_j, revised_p) = zip(*[s for s in revisions_ix if s[0] >= revisions_details_start])
                ix_j = revised_j - revised_j[0]
                revision_weights = revision_weights.transpose(0, 2, 1, 3)[:, :, ix_j, revised_p]
                revisions = revisions[ix_j, revised_p]
                revision_detailed_impacts = revision_weights @ revisions
                revised = revised[ix_j, revised_p]
                revised_prev = revised_prev[ix_j, revised_p]
                if t is not None:
                    revision_weights = revision_weights[0]
                    revision_detailed_impacts = revision_detailed_impacts[0]
            revised_impact_forecasts = revision_results.smoothed_forecasts[..., start:end]
            if end > revision_results.nobs:
                predict_start = max(start, revision_results.nobs)
                p = revision_results.predict(start=predict_start, end=end, **extend_kwargs)
                revised_impact_forecasts = np.concatenate((revised_impact_forecasts, p.forecasts), axis=1)
            revision_impacts = (revised_impact_forecasts - prev_impacted_forecasts).T
            if t is not None:
                revision_impacts = revision_impacts[0]
        if len(revisions_ix) > 0:
            (revised_j, revised_p) = zip(*revisions_ix)
            ix_j = revised_j - revised_j[0]
            revisions_all = revisions_all[ix_j, revised_p]
            revised_all = revised_all[ix_j, revised_p]
            revised_prev_all = revised_prev_all[ix_j, revised_p]
        if len(updates_ix) > 0:
            (update_t, update_k) = zip(*updates_ix)
            update_start_t = np.min(update_t)
            update_end_t = np.max(update_t)
            if revision_results is None:
                forecasts = previous.predict(start=update_start_t, end=update_end_t + 1, **extend_kwargs).smoothed_forecasts.T
            else:
                forecasts = revision_results.predict(start=update_start_t, end=update_end_t + 1).smoothed_forecasts.T
            realized = self.endog.T[update_start_t:update_end_t + 1]
            forecasts_error = realized - forecasts
            ix_t = update_t - update_start_t
            update_realized = realized[ix_t, update_k]
            update_forecasts = forecasts[ix_t, update_k]
            update_forecasts_error = forecasts_error[ix_t, update_k]
            if self.design.shape[2] == 1:
                design = self.design[..., 0][None, ...]
            elif end <= self.nobs:
                design = self.design[..., start:end].transpose(2, 0, 1)
            elif design is None:
                raise ValueError('Model has time-varying design matrix, so an updated time-varying matrix for period `t` is required.')
            elif design.ndim == 2:
                design = design[None, ...]
            else:
                design = design.transpose(2, 0, 1)
            state_gain = previous.smoothed_state_gain(updates_ix, start=start, end=end, extend_kwargs=extend_kwargs)
            if state_index is not None:
                design = design[:, :, state_index]
                state_gain = state_gain[:, state_index]
            obs_gain = design @ state_gain
            update_impacts = obs_gain @ update_forecasts_error
            if t is not None:
                obs_gain = obs_gain[0]
                update_impacts = update_impacts[0]
        else:
            update_impacts = None
            update_forecasts = None
            update_realized = None
            update_forecasts_error = None
            obs_gain = None
        out = SimpleNamespace(update_impacts=update_impacts, revision_detailed_impacts=revision_detailed_impacts, news=update_forecasts_error, revisions=revisions, revisions_all=revisions_all, gain=obs_gain, revision_weights=revision_weights, update_forecasts=update_forecasts, update_realized=update_realized, revised=revised, revised_all=revised_all, revised_prev=revised_prev, revised_prev_all=revised_prev_all, prev_impacted_forecasts=prev_impacted_forecasts, post_impacted_forecasts=post_impacted_forecasts, revision_results=revision_results, revision_impacts=revision_impacts, revisions_ix=revisions_ix, revisions_details=revisions_details, revisions_grouped=revisions_grouped, revisions_details_start=revisions_details_start, updates_ix=updates_ix, state_index=state_index)
        return out

    def smoothed_state_gain(self, updates_ix, t=None, start=None, end=None, extend_kwargs=None):
        if False:
            i = 10
            return i + 15
        '\n        Cov(\\tilde \\alpha_{t}, I) Var(I, I)^{-1}\n\n        where I is a vector of forecast errors associated with\n        `update_indices`.\n\n        Parameters\n        ----------\n        updates_ix : list\n            List of indices `(t, i)`, where `t` denotes a zero-indexed time\n            location and `i` denotes a zero-indexed endog variable.\n        '
        if t is not None and (start is not None or end is not None):
            raise ValueError('Cannot specify both `t` and `start` or `end`.')
        if t is not None:
            start = t
            end = t + 1
        if start is None:
            start = self.nobs - 1
        if end is None:
            end = self.nobs
        if extend_kwargs is None:
            extend_kwargs = {}
        if start < 0 or end < 0:
            raise ValueError('Negative `t`, `start`, or `end` is not allowed.')
        if end <= start:
            raise ValueError('`end` must be after `start`')
        n_periods = end - start
        n_updates = len(updates_ix)

        def get_mat(which, t):
            if False:
                print('Hello World!')
            mat = getattr(self, which)
            if mat.shape[-1] > 1:
                if t < self.nobs:
                    out = mat[..., t]
                else:
                    if which not in extend_kwargs or extend_kwargs[which].shape[-1] <= t - self.nobs:
                        raise ValueError(f'Model has time-varying {which} matrix, so an updated time-varying matrix for the extension period is required.')
                    out = extend_kwargs[which][..., t - self.nobs]
            else:
                out = mat[..., 0]
            return out

        def get_cov_state_revision(t):
            if False:
                return 10
            tmp1 = np.zeros((self.k_states, n_updates))
            for i in range(n_updates):
                (t_i, k_i) = updates_ix[i]
                acov = self.smoothed_state_autocovariance(lag=t - t_i, t=t, extend_kwargs=extend_kwargs)
                Z_i = get_mat('design', t_i)
                tmp1[:, i:i + 1] = acov @ Z_i[k_i:k_i + 1].T
            return tmp1
        tmp1 = np.zeros((n_periods, self.k_states, n_updates))
        for s in range(start, end):
            tmp1[s - start] = get_cov_state_revision(s)
        tmp2 = np.zeros((n_updates, n_updates))
        for i in range(n_updates):
            (t_i, k_i) = updates_ix[i]
            for j in range(i + 1):
                (t_j, k_j) = updates_ix[j]
                Z_i = get_mat('design', t_i)
                Z_j = get_mat('design', t_j)
                acov = self.smoothed_state_autocovariance(lag=t_i - t_j, t=t_i, extend_kwargs=extend_kwargs)
                tmp2[i, j] = tmp2[j, i] = np.squeeze(Z_i[k_i:k_i + 1] @ acov @ Z_j[k_j:k_j + 1].T)
                if t_i == t_j:
                    H = get_mat('obs_cov', t_i)
                    if i == j:
                        tmp2[i, j] += H[k_i, k_j]
                    else:
                        tmp2[i, j] += H[k_i, k_j]
                        tmp2[j, i] += H[k_i, k_j]
        gain = tmp1 @ np.linalg.inv(tmp2)
        if t is not None:
            gain = gain[0]
        return gain

    def _get_smoothed_forecasts(self):
        if False:
            return 10
        if self._smoothed_forecasts is None:
            self._smoothed_forecasts = np.zeros(self.forecasts.shape, dtype=self.dtype)
            self._smoothed_forecasts_error = np.zeros(self.forecasts_error.shape, dtype=self.dtype)
            self._smoothed_forecasts_error_cov = np.zeros(self.forecasts_error_cov.shape, dtype=self.dtype)
            for t in range(self.nobs):
                design_t = 0 if self.design.shape[2] == 1 else t
                obs_cov_t = 0 if self.obs_cov.shape[2] == 1 else t
                obs_intercept_t = 0 if self.obs_intercept.shape[1] == 1 else t
                mask = ~self.missing[:, t].astype(bool)
                self._smoothed_forecasts[:, t] = np.dot(self.design[:, :, design_t], self.smoothed_state[:, t]) + self.obs_intercept[:, obs_intercept_t]
                if self.nmissing[t] > 0:
                    self._smoothed_forecasts_error[:, t] = np.nan
                self._smoothed_forecasts_error[mask, t] = self.endog[mask, t] - self._smoothed_forecasts[mask, t]
                self._smoothed_forecasts_error_cov[:, :, t] = np.dot(np.dot(self.design[:, :, design_t], self.smoothed_state_cov[:, :, t]), self.design[:, :, design_t].T) + self.obs_cov[:, :, obs_cov_t]
        return (self._smoothed_forecasts, self._smoothed_forecasts_error, self._smoothed_forecasts_error_cov)

    @property
    def smoothed_forecasts(self):
        if False:
            return 10
        return self._get_smoothed_forecasts()[0]

    @property
    def smoothed_forecasts_error(self):
        if False:
            print('Hello World!')
        return self._get_smoothed_forecasts()[1]

    @property
    def smoothed_forecasts_error_cov(self):
        if False:
            i = 10
            return i + 15
        return self._get_smoothed_forecasts()[2]

    def get_smoothed_decomposition(self, decomposition_of='smoothed_state', state_index=None):
        if False:
            print('Hello World!')
        '\n        Decompose smoothed output into contributions from observations\n\n        Parameters\n        ----------\n        decomposition_of : {"smoothed_state", "smoothed_signal"}\n            The object to perform a decomposition of. If it is set to\n            "smoothed_state", then the elements of the smoothed state vector\n            are decomposed into the contributions of each observation. If it\n            is set to "smoothed_signal", then the predictions of the\n            observation vector based on the smoothed state vector are\n            decomposed. Default is "smoothed_state".\n        state_index : array_like, optional\n            An optional index specifying a subset of states to use when\n            constructing the decomposition of the "smoothed_signal". For\n            example, if `state_index=[0, 1]` is passed, then only the\n            contributions of observed variables to the smoothed signal arising\n            from the first two states will be returned. Note that if not all\n            states are used, the contributions will not sum to the smoothed\n            signal. Default is to use all states.\n\n        Returns\n        -------\n        data_contributions : array\n            Contributions of observations to the decomposed object. If the\n            smoothed state is being decomposed, then `data_contributions` are\n            shaped `(nobs, k_states, nobs, k_endog)`, where the\n            `(t, m, j, p)`-th element is the contribution of the `p`-th\n            observation at time `j` to the `m`-th state at time `t`. If the\n            smoothed signal is being decomposed, then `data_contributions` are\n            shaped `(nobs, k_endog, nobs, k_endog)`, where the\n            `(t, k, j, p)`-th element is the contribution of the `p`-th\n            observation at time `j` to the smoothed prediction of the `k`-th\n            observation at time `t`.\n        obs_intercept_contributions : array\n            Contributions of the observation intercept to the decomposed\n            object. If the smoothed state is being decomposed, then\n            `obs_intercept_contributions` are shaped\n            `(nobs, k_states, nobs, k_endog)`, where the `(t, m, j, p)`-th\n            element is the contribution of the `p`-th observation intercept at\n            time `j` to the `m`-th state at time `t`. If the smoothed signal\n            is being decomposed, then `obs_intercept_contributions` are shaped\n            `(nobs, k_endog, nobs, k_endog)`, where the `(t, k, j, p)`-th\n            element is the contribution of the `p`-th observation at time `j`\n            to the smoothed prediction of the `k`-th observation at time `t`.\n        state_intercept_contributions : array\n            Contributions of the state intercept to the decomposed object. If\n            the smoothed state is being decomposed, then\n            `state_intercept_contributions` are shaped\n            `(nobs, k_states, nobs, k_states)`, where the `(t, m, j, l)`-th\n            element is the contribution of the `l`-th state intercept at\n            time `j` to the `m`-th state at time `t`. If the smoothed signal\n            is being decomposed, then `state_intercept_contributions` are\n            shaped `(nobs, k_endog, nobs, k_endog)`, where the\n            `(t, k, j, l)`-th element is the contribution of the `p`-th\n            observation at time `j` to the smoothed prediction of the `k`-th\n            observation at time `t`.\n        prior_contributions : array\n            Contributions of the prior to the decomposed object. If the\n            smoothed state is being decomposed, then `prior_contributions` are\n            shaped `(nobs, k_states, k_states)`, where the `(t, m, l)`-th\n            element is the contribution of the `l`-th element of the prior\n            mean to the `m`-th state at time `t`. If the smoothed signal is\n            being decomposed, then `prior_contributions` are shaped\n            `(nobs, k_endog, k_states)`, where the `(t, k, l)`-th\n            element is the contribution of the `l`-th element of the prior mean\n            to the smoothed prediction of the `k`-th observation at time `t`.\n\n        Notes\n        -----\n        Denote the smoothed state at time :math:`t` by :math:`\\alpha_t`. Then\n        the smoothed signal is :math:`Z_t \\alpha_t`, where :math:`Z_t` is the\n        design matrix operative at time :math:`t`.\n        '
        if decomposition_of not in ['smoothed_state', 'smoothed_signal']:
            raise ValueError('Invalid value for `decomposition_of`. Must be one of "smoothed_state" or "smoothed_signal".')
        (weights, state_intercept_weights, prior_weights) = tools._compute_smoothed_state_weights(self.model, compute_prior_weights=True, scale=self.scale)
        ZT = self.model.design.T
        dT = self.model.obs_intercept.T
        cT = self.model.state_intercept.T
        if decomposition_of == 'smoothed_signal' and state_index is not None:
            ZT = ZT[:, state_index, :]
            weights = weights[:, :, state_index]
            prior_weights = prior_weights[:, state_index, :]
        if decomposition_of == 'smoothed_signal':
            weights = np.nansum(weights[..., None] * ZT[:, None, :, None, :], axis=2).transpose(0, 1, 3, 2)
            state_intercept_weights = np.nansum(state_intercept_weights[..., None] * ZT[:, None, :, None, :], axis=2).transpose(0, 1, 3, 2)
            prior_weights = np.nansum(prior_weights[..., None] * ZT[:, :, None, :], axis=1).transpose(0, 2, 1)
        data_contributions = weights * self.model.endog.T[None, :, None, :]
        data_contributions = data_contributions.transpose(0, 2, 1, 3)
        obs_intercept_contributions = -weights * dT[None, :, None, :]
        obs_intercept_contributions = obs_intercept_contributions.transpose(0, 2, 1, 3)
        state_intercept_contributions = state_intercept_weights * cT[None, :, None, :]
        state_intercept_contributions = state_intercept_contributions.transpose(0, 2, 1, 3)
        prior_contributions = prior_weights * self.initial_state[None, None, :]
        return (data_contributions, obs_intercept_contributions, state_intercept_contributions, prior_contributions)
"""
See statsmodels.tsa.arima.model.ARIMA and statsmodels.tsa.SARIMAX.
"""
ARIMA_DEPRECATION_ERROR = '\nstatsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have\nbeen removed in favor of statsmodels.tsa.arima.model.ARIMA (note the .\nbetween arima and model) and statsmodels.tsa.SARIMAX.\n\nstatsmodels.tsa.arima.model.ARIMA makes use of the statespace framework and\nis both well tested and maintained. It also offers alternative specialized\nparameter estimators.\n'

class ARMA:
    """
    ARMA has been deprecated in favor of the new implementation

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA models with a variety of parameter estimators
    statsmodels.tsa.statespace.SARIMAX
        SARIMAX models estimated using MLE
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError(ARIMA_DEPRECATION_ERROR)

class ARIMA(ARMA):
    """
    ARIMA has been deprecated in favor of the new implementation

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA models with a variety of parameter estimators
    statsmodels.tsa.statespace.SARIMAX
        SARIMAX models estimated using MLE
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class ARMAResults:
    """
    ARMA has been deprecated in favor of the new implementation

    See Also
    --------
    statsmodels.tsa.arima.model.ARIMA
        ARIMA models with a variety of parameter estimators
    statsmodels.tsa.statespace.SARIMAX
        SARIMAX models estimated using MLE
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise NotImplementedError(ARIMA_DEPRECATION_ERROR)

class ARIMAResults(ARMAResults):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
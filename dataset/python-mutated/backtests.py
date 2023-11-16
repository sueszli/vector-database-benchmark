import numpy as np
import scipy.stats as ss
from scipy import linalg

class CampbellBacktesting:
    """
    This class implements the Haircut Sharpe Ratios and Profit Hurdles algorithms described in the following paper:
    `Campbell R. Harvey and Yan Liu, Backtesting, (Fall 2015). Journal of Portfolio Management,
    2015 <https://papers.ssrn.com/abstract_id=2345489>`_; The code is based on the code provided by the authors of the paper.

    The Haircut Sharpe Ratios algorithm lets the user adjust the observed Sharpe Ratios to take multiple testing into account
    and calculate the corresponding haircuts. The haircut is the percentage difference between the original Sharpe ratio
    and the new Sharpe ratio.

    The Profit Hurdle algorithm lets the user calculate the required mean return for a strategy at a given level of
    significance, taking multiple testing into account.
    """

    def __init__(self, simulations=2000):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the desired number of simulations to make in Haircut Sharpe Ratios or Profit Hurdle algorithms.\n\n        :param simulations: (int) Number of simulations\n        '
        self.simulations = simulations

    @staticmethod
    def _sample_random_multest(rho, n_trails, prob_zero_mean, lambd, n_simulations, annual_vol=0.15, n_obs=240):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates empirical p-value distributions.\n\n        The algorithm is described in the paper and is based on the model estimated by `Harvey, C.R., Y. Liu,\n        and H. Zhu., … and the Cross-section of Expected Returns. Review of Financial Studies, forthcoming 2015`,\n        referred to as the HLZ model.\n\n        It provides a set of simulated t-statistics based on the parameters recieved from the _parameter_calculation\n        method.\n\n        Researchers propose a structural model to capture trading strategies’ underlying distribution.\n        With probability p0 (prob_zero_mean), a strategy has a mean return of zero and therefore comes\n        from the null distribution. With probability 1 – p0, a strategy has a nonzero mean and therefore\n        comes from the alternative distribution - exponential.\n\n        :param rho: (float) Average correlation among returns\n        :param n_trails: (int) Total number of trials inside a simulation\n        :param prob_zero_mean: (float) Probability for a random factor to have a zero mean\n        :param lambd: (float) Average of monthly mean returns for true strategies\n        :param n_simulations: (int) Number of rows (simulations)\n        :param annual_vol: (float) HLZ assume that the innovations in returns follow a normal distribution with a mean\n                                   of zero and a standard deviation of ma = 15%\n        :param n_obs: (int) Number of observations of used for volatility estimation from HLZ\n        :return: (np.ndarray) Array with distributions calculated\n        '
        pass

    @staticmethod
    def _parameter_calculation(rho):
        if False:
            return 10
        '\n        Estimates the parameters used to generate the distributions in _sample_random_multest - the HLZ model.\n\n        Based on the work of HLZ, the pairwise correlation of returns is used to estimate the probability (prob_zero_mean),\n        total number of trials (n_simulations) and (lambd) - parameter of the exponential distribution. Levels and\n        parameters taken from the HLZ research.\n\n        :param rho: (float) Average correlation coefficient between strategy returns\n        :return: (np.array) Array of parameters\n        '
        pass

    @staticmethod
    def _annualized_sharpe_ratio(sharpe_ratio, sampling_frequency='A', rho=0, annualized=False, autocorr_adjusted=False):
        if False:
            i = 10
            return i + 15
        "\n        Calculate the equivalent annualized Sharpe ratio after taking the autocorrelation of returns into account.\n\n        Adjustments are based on the work of `Lo, A., The Statistics of Sharpe Ratios. Financial Analysts Journal,\n        58 (2002), pp. 36-52` and are described there in more detail.\n\n        :param sharpe_ratio: (float) Sharpe ratio of the strategy\n        :param sampling_frequency: (str) Sampling frequency of returns\n                                   ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]\n        :param rho: (float) Autocorrelation coefficient of returns at specified frequency\n        :param annualized: (bool) Flag if annualized, 'ind_an' = 1, otherwise = 0\n        :param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation\n        :return: (float) Adjusted annualized Sharpe ratio\n        "
        pass

    @staticmethod
    def _monthly_observations(num_obs, sampling_frequency):
        if False:
            while True:
                i = 10
        "\n        Calculates the number of monthly observations based on sampling frequency and number of observations.\n\n        :param num_obs: (int) Number of observations used for modelling\n        :param sampling_frequency: (str) Sampling frequency of returns\n                                   ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]\n        :return: (np.float64) Number of monthly observations\n        "
        pass

    @staticmethod
    def _holm_method_sharpe(all_p_values, num_mult_test, p_val):
        if False:
            print('Hello World!')
        '\n        Runs one cycle of the Holm method for the Haircut Shape ratio algorithm.\n\n        :param all_p_values: (np.array) Sorted p-values to adjust\n        :param num_mult_test: (int) Number of multiple tests allowed\n        :param p_val: (float) Significance level p-value\n        :return: (np.float64) P-value adjusted at a significant level\n        '
        pass

    @staticmethod
    def _bhy_method_sharpe(all_p_values, num_mult_test, p_val):
        if False:
            for i in range(10):
                print('nop')
        '\n        Runs one cycle of the BHY method for the Haircut Shape ratio algorithm.\n\n        :param all_p_values: (np.array) Sorted p-values to adjust\n        :param num_mult_test: (int) Number of multiple tests allowed\n        :param p_val: (float) Significance level p-value\n        :param c_constant: (float) Constant used in BHY method\n        :return: (np.float64) P-value adjusted at a significant level\n        '
        pass

    @staticmethod
    def _sharpe_ratio_haircut(p_val, monthly_obs, sr_annual):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the adjusted Sharpe ratio and the haircut based on the final p-value of the method.\n\n        :param p_val: (float) Adjusted p-value of the method\n        :param monthly_obs: (int) Number of monthly observations\n        :param sr_annual: (float) Annualized Sharpe ratio to compare to\n        :return: (np.array) Elements (Adjusted annual Sharpe ratio, Haircut percentage)\n        '
        pass

    @staticmethod
    def _holm_method_returns(p_values_simulation, num_mult_test, alpha_sig):
        if False:
            print('Hello World!')
        '\n        Runs one cycle of the Holm method for the Profit Hurdle algorithm.\n\n        :param p_values_simulation: (np.array) Sorted p-values to adjust\n        :param num_mult_test: (int) Number of multiple tests allowed\n        :param alpha_sig: (float) Significance level (e.g., 5%)\n        :return: (np.float64) P-value adjusted at a significant level\n        '
        pass

    @staticmethod
    def _bhy_method_returns(p_values_simulation, num_mult_test, alpha_sig):
        if False:
            print('Hello World!')
        '\n        Runs one cycle of the BHY method for the Profit Hurdle algorithm.\n\n        :param p_values_simulation: (np.array) Sorted p-values to adjust\n        :param num_mult_test: (int) Number of multiple tests allowed\n        :param alpha_sig: (float) Significance level (e.g., 5%)\n        :return: (np.float64) P-value adjusted at a significant level\n        '
        pass

    def haircut_sharpe_ratios(self, sampling_frequency, num_obs, sharpe_ratio, annualized, autocorr_adjusted, rho_a, num_mult_test, rho):
        if False:
            i = 10
            return i + 15
        "\n        Calculates the adjusted Sharpe ratio due to testing multiplicity.\n\n        This algorithm lets the user calculate Sharpe ratio adjustments and the corresponding haircuts based on\n        the key parameters of returns from the strategy. The adjustment methods are Bonferroni, Holm,\n        BHY (Benjamini, Hochberg and Yekutieli) and the Average of them. The algorithm calculates adjusted p-value,\n        adjusted Sharpe ratio and the haircut.\n\n        The haircut is the percentage difference between the original Sharpe ratio and the new Sharpe ratio.\n\n        :param sampling_frequency: (str) Sampling frequency ['D','W','M','Q','A'] of returns\n        :param num_obs: (int) Number of returns in the frequency specified in the previous step\n        :param sharpe_ratio: (float) Sharpe ratio of the strategy. Either annualized or in the frequency specified in the previous step\n        :param annualized: (bool) Flag if Sharpe ratio is annualized\n        :param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation\n        :param rho_a: (float) Autocorrelation coefficient of returns at the specified frequency (if the Sharpe ratio\n                              wasn't corrected)\n        :param num_mult_test: (int) Number of other strategies tested (multiple tests)\n        :param rho: (float) Average correlation among returns of strategies tested\n        :return: (np.ndarray) Array with adjuted p-value, adjusted Sharpe ratio, and haircut as rows\n                              for Bonferroni, Holm, BHY and average adjustment as columns\n        "
        pass

    def profit_hurdle(self, num_mult_test, num_obs, alpha_sig, vol_anu, rho):
        if False:
            return 10
        "\n        Calculates the required mean monthly return for a strategy at a given level of significance.\n\n        This algorithm uses four adjustment methods - Bonferroni, Holm, BHY (Benjamini, Hochberg and Yekutieli)\n        and the Average of them. The result is the Minimum Average Monthly Return for the strategy to be significant\n        at a given significance level, taking into account multiple testing.\n\n        This function doesn't allow for any autocorrelation in the strategy returns.\n\n        :param num_mult_test: (int) Number of tests in multiple testing allowed (number of other strategies tested)\n        :param num_obs: (int) Number of monthly observations for a strategy\n        :param alpha_sig: (float) Significance level (e.g., 5%)\n        :param vol_anu: (float) Annual volatility of returns(e.g., 0.05 or 5%)\n        :param rho: (float) Average correlation among returns of strategies tested\n        :return: (np.ndarray) Minimum Average Monthly Returns for\n                              [Independent tests, Bonferroni, Holm, BHY and Average for Multiple tests]\n        "
        pass
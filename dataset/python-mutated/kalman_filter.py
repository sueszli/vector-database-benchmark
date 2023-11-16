import numpy as np
import scipy.linalg
"\nTable for the 0.95 quantile of the chi-square distribution with N degrees of\nfreedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv\nfunction and used as Mahalanobis gating threshold.\n"
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.07, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        if False:
            return 10
        (ndim, dt) = (4, 1.0)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        if False:
            while True:
                i = 10
        'Create track from unassociated measurement.\n\n        Parameters\n        ----------\n        measurement : ndarray\n            Bounding box coordinates (x, y, a, h) with center position (x, y),\n            aspect ratio a, and height h.\n\n        Returns\n        -------\n        (ndarray, ndarray)\n            Returns the mean vector (8 dimensional) and covariance matrix (8x8\n            dimensional) of the new track. Unobserved velocities are initialized\n            to 0 mean.\n\n        '
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [2 * self._std_weight_position * measurement[3], 2 * self._std_weight_position * measurement[3], 0.01, 2 * self._std_weight_position * measurement[3], 10 * self._std_weight_velocity * measurement[3], 10 * self._std_weight_velocity * measurement[3], 1e-05, 10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return (mean, covariance)

    def predict(self, mean, covariance):
        if False:
            i = 10
            return i + 15
        'Run Kalman filter prediction step.\n\n        Parameters\n        ----------\n        mean : ndarray\n            The 8 dimensional mean vector of the object state at the previous\n            time step.\n        covariance : ndarray\n            The 8x8 dimensional covariance matrix of the object state at the\n            previous time step.\n\n        Returns\n        -------\n        (ndarray, ndarray)\n            Returns the mean vector and covariance matrix of the predicted\n            state. Unobserved velocities are initialized to 0 mean.\n\n        '
        std_pos = [self._std_weight_position * mean[3], self._std_weight_position * mean[3], 0.01, self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3], self._std_weight_velocity * mean[3], 1e-05, self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return (mean, covariance)

    def project(self, mean, covariance):
        if False:
            i = 10
            return i + 15
        "Project state distribution to measurement space.\n\n        Parameters\n        ----------\n        mean : ndarray\n            The state's mean vector (8 dimensional array).\n        covariance : ndarray\n            The state's covariance matrix (8x8 dimensional).\n\n        Returns\n        -------\n        (ndarray, ndarray)\n            Returns the projected mean and covariance matrix of the given state\n            estimate.\n\n        "
        std = [self._std_weight_position * mean[3], self._std_weight_position * mean[3], 0.1, self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return (mean, covariance + innovation_cov)

    def multi_predict(self, mean, covariance):
        if False:
            while True:
                i = 10
        'Run Kalman filter prediction step (Vectorized version).\n        Parameters\n        ----------\n        mean : ndarray\n            The Nx8 dimensional mean matrix of the object states at the previous\n            time step.\n        covariance : ndarray\n            The Nx8x8 dimensional covariance matrics of the object states at the\n            previous time step.\n        Returns\n        -------\n        (ndarray, ndarray)\n            Returns the mean vector and covariance matrix of the predicted\n            state. Unobserved velocities are initialized to 0 mean.\n        '
        std_pos = [self._std_weight_position * mean[:, 3], self._std_weight_position * mean[:, 3], 0.01 * np.ones_like(mean[:, 3]), self._std_weight_position * mean[:, 3]]
        std_vel = [self._std_weight_velocity * mean[:, 3], self._std_weight_velocity * mean[:, 3], 1e-05 * np.ones_like(mean[:, 3]), self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return (mean, covariance)

    def update(self, mean, covariance, measurement):
        if False:
            while True:
                i = 10
        "Run Kalman filter correction step.\n\n        Parameters\n        ----------\n        mean : ndarray\n            The predicted state's mean vector (8 dimensional).\n        covariance : ndarray\n            The state's covariance matrix (8x8 dimensional).\n        measurement : ndarray\n            The 4 dimensional measurement vector (x, y, a, h), where (x, y)\n            is the center position, a the aspect ratio, and h the height of the\n            bounding box.\n\n        Returns\n        -------\n        (ndarray, ndarray)\n            Returns the measurement-corrected state distribution.\n\n        "
        (projected_mean, projected_cov) = self.project(mean, covariance)
        (chol_factor, lower) = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return (new_mean, new_covariance)

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        if False:
            for i in range(10):
                print('nop')
        'Compute gating distance between state distribution and measurements.\n        A suitable distance threshold can be obtained from `chi2inv95`. If\n        `only_position` is False, the chi-square distribution has 4 degrees of\n        freedom, otherwise 2.\n        Parameters\n        ----------\n        mean : ndarray\n            Mean vector over the state distribution (8 dimensional).\n        covariance : ndarray\n            Covariance of the state distribution (8x8 dimensional).\n        measurements : ndarray\n            An Nx4 dimensional matrix of N measurements, each in\n            format (x, y, a, h) where (x, y) is the bounding box center\n            position, a the aspect ratio, and h the height.\n        only_position : Optional[bool]\n            If True, distance computation is done with respect to the bounding\n            box center position only.\n        Returns\n        -------\n        ndarray\n            Returns an array of length N, where the i-th element contains the\n            squared Mahalanobis distance between (mean, covariance) and\n            `measurements[i]`.\n        '
        (mean, covariance) = self.project(mean, covariance)
        if only_position:
            (mean, covariance) = (mean[:2], covariance[:2, :2])
            measurements = measurements[:, :2]
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
#include "WSrSeqUKF.h"
#include "Tools/Math/General.h"
#include "IKFModel.h"
#include <sstream>

WSrSeqUKF::WSrSeqUKF(IKFModel *model): IWeightedKalmanFilter(model), m_estimate(model->totalStates()), m_unscented_transform(model->totalStates())
{
    init();
}

WSrSeqUKF::WSrSeqUKF(const WSrSeqUKF& source): IWeightedKalmanFilter(source.m_model), m_estimate(source.m_estimate), m_unscented_transform(source.m_unscented_transform)
{
    init();
    m_filter_weight = source.m_filter_weight;
}

WSrSeqUKF::~WSrSeqUKF()
{
}

void WSrSeqUKF::init()
{
    initialiseEstimate(m_estimate);
    m_weighting_enabled = false;
    m_filter_weight = 1.f;

    m_sqrt_covariance_weights = Matrix(m_unscented_transform.covarianceWeights());
    for(unsigned int m = 0; m < m_sqrt_covariance_weights.getm(); ++m)
    {
        for(unsigned int n = 0; n < m_sqrt_covariance_weights.getm(); ++n)
        {
            m_sqrt_covariance_weights[m][n] = sqrt(m_sqrt_covariance_weights[m][n]);
        }
    }
}

/*!
 * @brief Calculates the sigma points from the current mean an covariance of the filter.
 * @return The sigma points that describe the current mean and covariance.
 */
Matrix WSrSeqUKF::GenerateSqrtSigmaPoints() const
{
    const unsigned int numPoints = m_unscented_transform.totalSigmaPoints();
    const Matrix current_mean = m_estimate.mean();
    const unsigned int num_states = m_estimate.totalStates();
    Matrix points(num_states, numPoints, false);

    points.setCol(0, current_mean); // First sigma point is the current mean with no deviation
    Matrix sqtCovariance = cholesky(m_unscented_transform.covarianceSigmaWeight() * m_estimate.covariance());
    Matrix deviation;

    for(unsigned int i = 1; i < num_states + 1; i++){
        int negIndex = i+num_states;
        deviation = sqtCovariance.getCol(i - 1);        // Get deviation from weighted covariance
        points.setCol(i, (current_mean + deviation));         // Add mean + deviation
        points.setCol(negIndex, (current_mean - deviation));  // Add mean - deviation
    }
    return points;
}

/*!
 * @brief Performs the time update of the filter.
 * @param deltaT The time that has passed since the previous update.
 * @param measurement The measurement/s (if any) that can be used to measure a change in the system.
 * @param linearProcessNoise The linear process noise that will be added.
 * @return True if the time update was performed successfully. False if it was not.
 */
bool WSrSeqUKF::timeUpdate(double delta_t, const Matrix& measurement, const Matrix& process_noise, const Matrix& measurement_noise)
{
    const unsigned int total_points = m_unscented_transform.totalSigmaPoints();

    // Calculate the current sigma points, and write to member variable.
    m_sigma_points = GenerateSqrtSigmaPoints();

    Matrix currentPoint; // temporary storage.

    // update each sigma point.
    for (unsigned int i = 0; i < total_points; ++i)
    {
        currentPoint = m_sigma_points.getCol(i);    // Get the sigma point.
        m_sigma_points.setCol(i, m_model->processEquation(currentPoint, delta_t, measurement));   // Write the propagated version of it.
    }

    // Calculate the new mean and covariance values.
    Matrix predictedMean = m_unscented_transform.CalculateMeanFromSigmas(m_sigma_points);
    Matrix predictedCovariance = m_unscented_transform.CalculateCovarianceFromSigmas(m_sigma_points, predictedMean) + process_noise;

    // Set the new mean and covariance values.
    MultivariateGaussian new_estimate = m_estimate;
    new_estimate.setMean(predictedMean);
    new_estimate.setCovariance(predictedCovariance);
    initialiseEstimate(new_estimate);

    return true;
}

/*!
 * @brief Performs the measurement update of the filter.
 * @param measurement The measurement to be used for the update.
 * @param measurementNoise The linear measurement noise that will be added.
 * @param measurementArgs Any additional information about the measurement, if required.
 * @return True if the measurement update was performed successfully. False if it was not.
 */
bool WSrSeqUKF::measurementUpdate(const Matrix& measurement, const Matrix& noise, const Matrix& args, unsigned int type)
{
    const unsigned int total_points = m_unscented_transform.totalSigmaPoints();
    const unsigned int totalMeasurements = measurement.getm();
    Matrix current_point; // temporary storage.

    Matrix Yprop(totalMeasurements, total_points);

    // First step is to calculate the expected measurmenent for each sigma point.
    for (unsigned int i = 0; i < total_points; ++i)
    {
        current_point = m_sigma_points.getCol(i);    // Get the sigma point.
        Yprop.setCol(i, m_model->measurementEquation(current_point, args, type));
    }

    // Now calculate the mean of these measurement sigmas.
    Matrix Ymean = m_unscented_transform.CalculateMeanFromSigmas(Yprop);

    Matrix Y(totalMeasurements, total_points,false);

    // Calculate the Y vector.
    for(unsigned int i = 0; i < total_points; ++i)
    {
        double weight = m_sqrt_covariance_weights[0][i];
        Y.setCol(i, weight*(Yprop.getCol(i) - Ymean));
    }

    // Calculate the new C and d values.
    Matrix Yaug = Y;
    Yaug.setCol(0, mathGeneral::sign(m_covariance_weights[0][0]) * Yaug.getCol(0));
    Matrix Ytransp = Yaug.transp();

    const Matrix innovation = measurement - Ymean;

    // Check for outlier, if outlier return without updating estimate.
    if(evaluateMeasurement(innovation, Y * Ytransp, noise) == false) // Y * Y^T is the estimate variance, by definition.
        return false;

    m_C = m_C - m_C * Ytransp * InverseMatrix(noise + Y*m_C*Ytransp) * Y * m_C;
    m_d = m_d + Ytransp * InverseMatrix(noise) * innovation;

    // Update mean and covariance.
    Matrix updated_mean = m_sigma_mean + m_X * m_C * m_d;
    Matrix updated_covariance = m_X * m_C * m_X.transp();

    m_estimate.setMean(updated_mean);
    m_estimate.setCovariance(updated_covariance);
    return true;
}

void WSrSeqUKF::initialiseEstimate(const MultivariateGaussian& estimate)
{
    // This is more complicated than you might expect because of all of the buffered values that
    // must be kept up to date.
    const unsigned int total_points = m_unscented_transform.totalSigmaPoints();
    const unsigned int num_states = m_estimate.totalStates();

    // Assign the estimate.
    m_estimate = estimate;

    // Redraw sigma points to cover this new estimate.
    m_sigma_mean = estimate.mean();
    m_sigma_points = GenerateSqrtSigmaPoints();

    // Initialise the variables for sequential measurement updates.
    m_C = Matrix(total_points, total_points, true);
    m_d = Matrix(total_points, 1, false);

    // Calculate X vector.
    m_X = Matrix(num_states, total_points, false);
    for(unsigned int i = 0; i < total_points; ++i)
    {
        double weight = m_sqrt_covariance_weights[0][i];
        m_X.setCol(i, weight*(m_sigma_points.getCol(i) - m_sigma_mean));
    }

    return;
}

bool WSrSeqUKF::evaluateMeasurement(const Matrix& innovation, const Matrix& estimate_variance, const Matrix& measurement_variance)
{
    if(!m_outlier_filtering_enabled and !m_weighting_enabled) return true;

    Matrix innov_transp = innovation.transp();
    Matrix sum_variance = estimate_variance + measurement_variance;

    if(m_outlier_filtering_enabled)
    {
        float innovation_2 = convDble(innov_transp * InverseMatrix(sum_variance) * innovation);
        if(m_outlier_threshold > 0 and innovation_2 > m_outlier_threshold)
            return false;
    }

    if(m_weighting_enabled)
    {
        m_filter_weight *= 1 / ( 1 + convDble(innov_transp * InverseMatrix(measurement_variance) * innovation));
    }

    return true;
}

std::string WSrSeqUKF::summary(bool detailed) const
{
    std::stringstream str_strm;
    str_strm << "ID: " << m_id <<  " Weight: " << m_filter_weight << std::endl;
    str_strm << m_estimate.string() << std::endl;
    return str_strm.str();
}

std::ostream& WSrSeqUKF::writeStreamBinary (std::ostream& output) const
{
    m_model->writeStreamBinary(output);
    m_unscented_transform.writeStreamBinary(output);
    m_estimate.writeStreamBinary(output);
    return output;
}

std::istream& WSrSeqUKF::readStreamBinary (std::istream& input)
{
    m_model->readStreamBinary(input);
    m_unscented_transform.readStreamBinary(input);
    init();
    MultivariateGaussian temp;
    temp.readStreamBinary(input);
    initialiseEstimate(temp);
    input.read((char*)&m_weighting_enabled, sizeof(m_weighting_enabled));
    input.read((char*)&m_filter_weight, sizeof(m_filter_weight));
    return input;
}

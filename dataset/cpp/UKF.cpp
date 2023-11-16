#include "UKF.h"
#include "debug.h"

using namespace std;


UKF::UKF(): m_numStates(0)
{
}

UKF::UKF(unsigned int numStates): m_numStates(numStates)
{
    m_mean = Matrix(m_numStates,1,false);
    m_covariance = Matrix(m_numStates,m_numStates,true);
    CalculateSigmaWeights();
}

UKF::UKF(const UKF& source)
{
    m_numStates = source.m_numStates;
    m_mean = source.m_mean;
    m_covariance = source.m_covariance;
    CalculateSigmaWeights();
}

UKF::~UKF()
{
}

Matrix UKF::CalculateMeanFromSigmas(const Matrix& sigmaPoints) const
{
    //unsigned int numPoints = sigmaPoints.getn();
    Matrix mean(sigmaPoints.getm(),1,false);
    mean = sigmaPoints * m_sigmaWeights.transp();
    return mean;
}

Matrix UKF::CalculateCovarianceFromSigmas(const Matrix& sigmaPoints, const Matrix& mean) const
{
    unsigned int numPoints = sigmaPoints.getn();
    Matrix covariance(m_numStates,m_numStates, false);
    Matrix diff;
    for(unsigned int i = 0; i < numPoints; ++i)
    {
        diff = sigmaPoints.getCol(i) - mean;
        covariance = covariance + m_sigmaWeights[0][i]*diff*diff.transp();
    }
    return covariance;
}

void UKF::CalculateSigmaWeights(float kappa)
{
    unsigned int numPoints = 2*m_numStates + 1;
    m_sigmaWeights = Matrix(1,numPoints, false);
    m_sqrtSigmaWeights = Matrix(1,numPoints, false);

    double meanWeight = kappa/(m_numStates+kappa);
    double outerWeight = (1.0-meanWeight)/(2*m_numStates);

    // First weight
    m_sigmaWeights[0][0] = meanWeight;
    m_sqrtSigmaWeights[0][0] = sqrt(meanWeight);
    // The rest
    for(unsigned int i = 1; i < numPoints; i++)
    {
        m_sigmaWeights[0][i] = outerWeight;
        m_sqrtSigmaWeights[0][i] = sqrt(outerWeight);
    }
}

Matrix UKF::GenerateSigmaPoints() const
{
    int numberOfSigmaPoints = 2*m_numStates+1;
    Matrix sigmaPoints(m_mean.getm(), numberOfSigmaPoints, false);

    sigmaPoints.setCol(0,m_mean); // First sigma point is the current mean with no deviation
    Matrix deviation;
    Matrix sqtCovariance = cholesky(m_numStates / (1-m_sigmaWeights[0][0]) * m_covariance);

    for(unsigned int i = 1; i < m_numStates + 1; i++){
        int negIndex = i+m_numStates;
        deviation = sqtCovariance.getCol(i - 1);        // Get weighted deviation
        sigmaPoints.setCol(i, (m_mean + deviation));                // Add mean + deviation
        sigmaPoints.setCol(negIndex, (m_mean - deviation));  // Add mean - deviation
    }
    return sigmaPoints;
}

double UKF::getMean(int stateId) const
{
    return m_mean[stateId][0];
}

double UKF::calculateSd(int stateId) const
{
    return sqrt(m_covariance[stateId][stateId]);
}

bool UKF::setState(Matrix mean, Matrix covariance)
{
    if( (mean.getm() == covariance.getm()) && (mean.getm() == covariance.getn()) )
    {
        m_numStates = mean.getm();
        m_mean = mean;
        m_covariance = covariance;
        CalculateSigmaWeights();
        return true;
    }
    else
    {
        return false;
    }
}

bool  UKF::timeUpdate(const Matrix& updatedSigmaPoints, const Matrix& processNoise)
{
    m_mean = CalculateMeanFromSigmas(updatedSigmaPoints);
    // Update covariance assuming additive process noise.
    m_covariance = CalculateCovarianceFromSigmas(updatedSigmaPoints, m_mean) + processNoise;
    return true;
}

bool UKF::measurementUpdate(const Matrix& measurement, const Matrix& measurementNoise, const Matrix& predictedMeasurementSigmas, const Matrix& stateEstimateSigmas)
{
    const int numMeasurements = measurement.getm();
    const int numberOfSigmaPoints = stateEstimateSigmas.getn();

    // Find mean of predicted measurement
    Matrix predictedMeasurement = CalculateMeanFromSigmas(predictedMeasurementSigmas);

    //Matrix Pyy(numMeasurements,numMeasurements,false);
    Matrix Pyy(measurementNoise);
    Matrix Pxy(stateEstimateSigmas.getm(),numMeasurements,false);

    Matrix temp;
    for(int i =0; i < numberOfSigmaPoints; i++)
    {
        // store difference between prediction and measurment.
        temp = predictedMeasurementSigmas.getCol(i) - predictedMeasurement;
        // Innovation covariance - Add Measurement noise
        Pyy = Pyy + m_sigmaWeights[0][i]*temp * temp.transp();
        // Cross correlation matrix
        Pxy = Pxy + m_sigmaWeights[0][i]*(stateEstimateSigmas.getCol(i) - m_mean) * temp.transp();
    }
    Matrix K;
    if(numMeasurements == 2)
    {
        K = Pxy * Invert22(Pyy);
    }
    else
    {
        K = Pxy * InverseMatrix(Pyy);
    }

    m_mean  = m_mean + K * (measurement - predictedMeasurement);
    m_covariance = m_covariance - K*Pyy*K.transp();

    // Alternate calculation
    //m_covariance = m_covariance - Pxy*Pyy*Pxy.transp();
\
    // Stolen from last years code... does not all seem right for this iplementation.
    //m_covariance = HT(horzcat(stateEstimateSigmas-m_mean*m_sigmaWeights - K*predictedMeasurementSigmas +
    //                          K*predictedMeasurement*m_sigmaWeights,K*measurementNoise));
    return true;
}

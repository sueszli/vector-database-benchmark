#include "MultivariateGaussian.h"
#include <assert.h>
#include <sstream>
#include <iostream>

/*! @brief Default constructor

     Creates a new moment with zero states. This is a null representation.

 */
MultivariateGaussian::MultivariateGaussian(): m_numStates(0)
{
    m_mean = Matrix(m_numStates,1,false);
    m_covariance = Matrix(m_numStates, m_numStates, false);
}

/*! @brief Size specified constructor

     Creates a new moment desribing the number of states specified.

    @param numStates the number fo states described by the moment.
 */
MultivariateGaussian::MultivariateGaussian(unsigned int numStates): m_numStates(numStates)
{
    m_mean = Matrix(m_numStates,1,false);
    m_covariance = Matrix(m_numStates, m_numStates, false);
}

MultivariateGaussian::MultivariateGaussian(const MultivariateGaussian& source): m_numStates(source.m_numStates)
{
    m_mean = source.m_mean;
    m_covariance = source.m_covariance;
}

MultivariateGaussian::MultivariateGaussian(const Matrix& mean, const Matrix& covariance)
{
    const unsigned int size = mean.getm();
    assert(size == covariance.getn());
    assert(size == covariance.getm());

    m_numStates = size;
    setMean(mean);
    setCovariance(covariance);
    return;
}

MultivariateGaussian& MultivariateGaussian::operator=(const MultivariateGaussian& source)
{
    if (this != &source) // protect against invalid self-assignment
    {
        m_numStates = source.m_numStates;
        setMean(source.mean());
        setCovariance(source.covariance());
    }
    // by convention, always return *this
    return *this;
}

bool MultivariateGaussian::operator ==(const MultivariateGaussian& b) const
{
    if( m_numStates != b.m_numStates)
    {
        return false;
    }
    if( m_mean != b.m_mean)
    {
        return false;
    }
    if( m_covariance != b.m_covariance)
    {
        return false;
    }
    return true;
}

/*! @brief Returns the mean of the specified state.

    @param stateNumber the number of the desired state.
 */
float MultivariateGaussian::mean(unsigned int stateNumber) const
{
    if(stateNumber < m_numStates)
        return m_mean[stateNumber][0];
    return 0.0f;
}

/*! @brief Returns the mean of the moment.
 */
Matrix MultivariateGaussian::mean() const
{
    return m_mean;
}

/*! @brief Returns the covariance of the moment
 */
Matrix MultivariateGaussian::covariance() const
{
    return m_covariance;
}

float MultivariateGaussian::covariance(unsigned int row, unsigned int col) const
{
    if( (row < m_numStates) and (col < m_numStates) )
        return m_covariance[row][col];
    return 0.0f;
}

/*! @brief Returns the standard deviation of the specified state.

    @param stateNumber the number of the desired state.
 */
float MultivariateGaussian::sd(unsigned int stateNumber) const
{
    return sqrt(variance(stateNumber));
}

/*! @brief Returns the variance of the specified state.

    @param stateNumber the number of the desired state.
 */
float MultivariateGaussian::variance(unsigned int stateNumber) const
{
    if(stateNumber < m_numStates)
        return m_covariance[stateNumber][stateNumber];
    return 0.0f;
}

/*! @brief Sets the mean of the moment to the given value/s.

    The dimensions of the new mean matrix must match the number of states for moment.

    @param newMean the new value for the mean.
 */
void MultivariateGaussian::setMean(const Matrix& newMean)
{
    bool isCorrectSize = ((unsigned int)newMean.getm() == m_numStates) && (newMean.getn() == 1);
    assert(isCorrectSize);
    assert(newMean.isValid());
    if(isCorrectSize and newMean.isValid())
    {
        m_mean = newMean;
    }
    return;
}

/*! @brief Sets the covariance of the moment to the given value/s.

    The dimensions of the new covariance matrix must match the number of states for moment.

    @param newCovariance the new value for the covariance.
 */
void MultivariateGaussian::setCovariance(const Matrix& newCovariance)
{
    bool isCorrectSize = (newCovariance.getm() == m_numStates) && (newCovariance.getn() == m_numStates);
    assert(isCorrectSize);
    assert(newCovariance.isValid());
    if(isCorrectSize and newCovariance.isValid())
    {
        m_covariance = newCovariance;
    }
    return;
}

/*! @brief Determines if the moment is null. this is the case if it contains zero states.
 */
bool MultivariateGaussian::isNull() const
{
    return (m_numStates < 1);
}

/*! @brief Return the current moment as a human-readable string for display.
 */
std::string MultivariateGaussian::string() const
{
    std::stringstream result;
    result << "Mean: " << mean().transp();
    result << "Covariance: " << std::endl;
    result << covariance();
    return result.str();
}

template <class T>
const char * c_cast(const T& input)
{
    return reinterpret_cast<const char*>(&input);
}

void MultivariateGaussian::writeData(std::ostream& output) const
{
    char header[] = {"m"};
    output.write(header,1);
    float temp;
    for(unsigned int i = 0; i < m_numStates; i++)
    {
        temp = mean(i);
        output.write(c_cast(temp), sizeof(temp));
    }
    for (int r = 0; r < m_covariance.getm(); r++)
    {
        for (int c = 0; c < m_covariance.getn(); c++)
        {
            temp = m_covariance[r][c];
            output.write(c_cast(temp), sizeof(temp));
        }
    }
    return;
}

std::ostream& MultivariateGaussian::writeStreamBinary (std::ostream& output) const
{
    output.write(reinterpret_cast<const char*>(&m_numStates), sizeof(m_numStates));
    WriteMatrix(output, m_mean);
    WriteMatrix(output, m_covariance);
    return output;
}

std::istream& MultivariateGaussian::readStreamBinary (std::istream& input)
{
    input.read(reinterpret_cast<char*>(&m_numStates), sizeof(m_numStates));
    m_mean = ReadMatrix(input);
    m_covariance = ReadMatrix(input);
    return input;
}

std::ostream& operator<< (std::ostream& output, const MultivariateGaussian& p_moment)
{
    return p_moment.writeStreamBinary(output);
}

std::istream& operator>> (std::istream& input, MultivariateGaussian& p_moment)
{
    return p_moment.readStreamBinary(input);
}

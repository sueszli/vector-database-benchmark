/*! @file OdometryEstimator.cpp
 @brief implementation of a class used to estimate the robots odometry
 
 @author Steven Nicklin
 
 Copyright (c) 2011 Steven Nicklin
 
 This file is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with NUbot.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "OdometryEstimator.h"
#include "nubotdataconfig.h"
#include "debug.h"

/*! @brief Constructor for OdometryEstimator class.
 */
OdometryEstimator::OdometryEstimator()
{
    m_logging_enabled = false;

    // Tuning variables
    m_minimum_support_foot_pressure = 1.0; // in Newtons. 0.75 initial (robot 214)
    m_turn_multiplier = 1.2f;   // Turn Gripping factor
    m_x_multiplier = 1.0f;      // X Gripping factor
    m_y_multiplier = 1.0f;      // Y Gripping factor

    // Variables used in calculation of odometry values.
    m_support_leg = none;
    m_left_foot_position.resize(6,0.0f);
    m_right_foot_position.resize(6,0.0f);
    if(m_logging_enabled)
    {
        std::string filename = DATA_DIR + std::string("odometryLog.csv");
        IntialiseFile(filename);
    }
}

/*! @brief Destructor for OdometryEstimator class.
 */
OdometryEstimator::~OdometryEstimator()
{
    m_odometry_log.close();
}

/*! @brief Opens and initialises the odometry log file.
    @param fileName The path of the log file to initialise.
 */
void OdometryEstimator::IntialiseFile(std::string filename)
{
    m_odometry_log.open(filename.c_str(), std::fstream::out | std::fstream::trunc);
    if(m_odometry_log.is_open())
    {
        m_odometry_log.clear();
        m_odometry_log << "GPS X, GPS Y, Compass,";
        m_odometry_log << "L Odom X, L Odom Y, L Odom Turn,";
        m_odometry_log << "R Odom X, R Odom Y, R Odom Turn,";
        m_odometry_log << "L Force Sum, R Force Sum, Left Z, Right Z, Curr Support Leg";
        m_odometry_log << std::endl;
        m_logging_enabled = true;
        debug << "Odometry log created sucessfully." << std::endl;
    }
    else
    {
        m_logging_enabled = false;
        debug << "Odometry log creation failed." << std::endl;
    }
}

/*! @brief Writes data to the odometry log file.
    @param gps External position data.
    @param compass External heading data.
    @param leftOdom Left foot based odometry data.
    @param rightOdom Right foot based odometry data.
    @param forceLeft The force in newtons of the left foot.
    @param forceRight The force in newtons of the right foot.
    @param supportLeg The id of the current support leg (0=none, 1=left, 2=right)
 */
void OdometryEstimator::WriteLogData(std::vector<float>& gps, float compass,
                                    const std::vector<float>& leftOdom, const std::vector<float>& rightOdom,
                                    float forceLeft, float forceRight, float leftZ, float rightZ, OdometryEstimator::LegIdentifier supportLeg)
{
    if(m_odometry_log.is_open() and m_odometry_log.good())
    {
        if(gps.size() > 0)
        {
            m_odometry_log << gps[0] << "," << gps[1] << "," << compass;
        }
        else
        {
            m_odometry_log << "0,0,0";
        }
        for(unsigned int n=0; n<leftOdom.size(); n++)
            m_odometry_log << "," << leftOdom[n];
        for(unsigned int n=0; n<rightOdom.size(); n++)
            m_odometry_log << "," << rightOdom[n];
        m_odometry_log << "," << forceLeft << "," << forceRight;
        m_odometry_log << "," << leftZ << "," << rightZ << "," << supportLeg;
        m_odometry_log << std::endl;
    }
}

/*! @brief Determines the current support leg based on the force applied to the feet.
    @param forceLeft The force in newtons of the left foot.
    @param forceRight The force in newtons of the right foot.
 */
OdometryEstimator::LegIdentifier OdometryEstimator::SelectSupportLegTouch(float forceLeft, float forceRight)
{
    LegIdentifier currSupport = m_support_leg;
    float min_force = m_minimum_support_foot_pressure;

    if(currSupport == left)
    {
        if(forceLeft < min_force and forceLeft < forceRight)
        {
                currSupport = right;
        }
    }
    else if(currSupport == right)
    {
        if(forceRight < min_force and forceRight < forceLeft)
        {
                currSupport = left;
        }
    }
    else
    {
        if(forceRight < forceLeft)
        {
                currSupport = left;
        }
        else
        {
                currSupport = right;
        }
    }
    m_support_leg = currSupport;
    return currSupport;
}

/*! @brief Determines the current support leg based on kinematic based height of each foot.
    @param left_z The position of the left foot in respect to the torso z-axis.
    @param right_z The position of the left foot in respect to the torso z-axis.
 */
OdometryEstimator::LegIdentifier OdometryEstimator::SelectSupportLegKinematic(float left_z, float right_z)
{
    LegIdentifier result;
    if(right_z < left_z)
    {
        result = right;
    }
    else
    {
        result = left;
    }
    return result;
}

/*! @brief Calculate the odometry estimate from the previous estimation
    @param leftPos Left foot positional data.
    @param rightPos Right foot positional data.
    @param forceLeft The force in newtons of the left foot.
    @param forceRight The force in newtons of the right foot.
    @param gps External position data.
    @param compass External heading data.
 */
std::vector<float> OdometryEstimator::CalculateNextStep(const std::vector<float>& leftPos, const std::vector<float>& rightPos,
                                                        float forceLeft, float forceRight, std::vector<float>& gps, float compass)
{
    std::vector<float> result;
    LegIdentifier currSupport;

    float leftZ = leftPos[2];
    float rightZ = rightPos[2];

    if(forceLeft != -1.0f or forceRight != -1.0f)
    {
        currSupport = SelectSupportLegTouch(forceLeft, forceRight);
    }
    else
    {
        currSupport = SelectSupportLegKinematic(leftZ, rightZ);
    }

    // Calculate left foot motion
    std::vector<float> LOdom(3);
    LOdom[0] = m_x_multiplier * (m_left_foot_position[0] - leftPos[0]);
    LOdom[1] = m_y_multiplier * (m_left_foot_position[1] - leftPos[1]);
    LOdom[2] = m_turn_multiplier * (m_left_foot_position[5] - leftPos[5]);

    // Calculate right foot motion
    std::vector<float> ROdom(3);
    ROdom[0] = m_x_multiplier * (m_right_foot_position[0] - rightPos[0]);
    ROdom[1] = m_y_multiplier * (m_right_foot_position[1] - rightPos[1]);
    ROdom[2] = m_turn_multiplier * (m_right_foot_position[5] - rightPos[5]);

    // Set results based on the support foot.
    if(currSupport == left)
        result = LOdom;
    else if(currSupport == right)
        result = ROdom;
    else
        result.resize(3,0.0f);

    // If external position data is available write to log.
    if(m_logging_enabled)
    {
        WriteLogData(gps, compass, LOdom, ROdom, forceLeft, forceRight, leftZ, rightZ, currSupport);
    }

    // Save historical data
    m_left_foot_position = leftPos;
    m_right_foot_position = rightPos;
    return result;
}

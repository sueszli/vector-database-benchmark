/*! @file NAOKick.cpp
    @brief Implementation of NAOKick class

    @author Jed Rietveld
 
 Copyright (c) 2010 Jed Rietveld
 
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

#include "NAOKick.h"
#include "NUPlatform/NUPlatform.h"
#include "Infrastructure/NUSensorsData/NUSensorsData.h"
#include "Infrastructure/FieldObjects/FieldObjects.h"
#include "Infrastructure/Jobs/MotionJobs/KickJob.h"
#include "Infrastructure/Jobs/MotionJobs/WalkJob.h"
#include "Motion/Tools/MotionCurves.h"

#include "motionconfig.h"
#include "debugverbositynumotion.h"
#include "Autoconfig/targetconfig.h"

#include "Tools/Math/General.h"
#include "Tools/Math/StlVector.h"
using namespace mathGeneral;

//#if DEBUG_LOCALISATION_VERBOSITY > 0



NAOKick::NAOKick(NUWalk* walk, NUSensorsData* data, NUActionatorsData* actions) : NUKick(walk, data, actions)
{
    m_kinematicModel = new Kinematics();
    m_kinematicModel->LoadModel();
    pose = DO_NOTHING;
    m_kicking_leg = noLeg;

    m_stateCommandGiven = false;
    m_estimatedStateCompleteTime = 0.0;
    m_currentTimestamp = 0;
    m_previousTimestamp = 0;
    loadKickParameters();

    m_pauseState = false;
    m_variableGainValue = 0.01;
    m_armCommandSent = false;
    m_kickWait = false;
}

/*! @brief Destructor for motion module
 */
NAOKick::~NAOKick()
{
    kill();
    delete m_kinematicModel;
}

void NAOKick::loadKickParameters()
{
    m_defaultMotorGain = 75.0f;                 // Default to 75% gain.
    m_defaultArmMotorGain = 35.0f;
    m_leftLegInitialPose.resize(6,0.0f);
    m_leftLegInitialPose[3] = 0.5;
    m_leftLegInitialPose[1] = - m_leftLegInitialPose[3] / 2.0f;
    m_leftLegInitialPose[5] = - m_leftLegInitialPose[3] / 2.0f;
    m_rightLegInitialPose.assign(m_leftLegInitialPose.begin(), m_leftLegInitialPose.end());

    jointLimit hyp = jointLimit(-1.145303, 0.740810);

    m_leftLegLimits.reserve(6);
    // Hip Roll
    m_leftLegLimits.push_back(jointLimit(-0.379472, 0.790477));
    // Hip Pitch
    m_leftLegLimits.push_back(jointLimit(-1.773912, 0.484090));
    // Hip Yaw - Pitch
    m_leftLegLimits.push_back(hyp);
    // Knee Pitch
    m_leftLegLimits.push_back(jointLimit(-0.092346, 2.112528));
    // Ankle Roll
    m_leftLegLimits.push_back(jointLimit(-0.769001, 0.397880));
    // Ankle Pitch
    m_leftLegLimits.push_back(jointLimit(-1.189516, 0.922747));

    m_rightLegLimits.reserve(6);
    // Hip Roll
    m_rightLegLimits.push_back(jointLimit(-0.738321, 0.414754));
    // Hip Pitch
    m_rightLegLimits.push_back(jointLimit(-1.772308, 0.485624));
    // Hip Yaw - Pitch
    m_rightLegLimits.push_back(hyp);
    // Knee Pitch
    m_rightLegLimits.push_back(jointLimit(-0.103083, 2.120198));
    // Ankle Roll
    m_rightLegLimits.push_back(jointLimit(-0.388676, 0.785875));
    // Ankle Pitch
    m_rightLegLimits.push_back(jointLimit(-1.186448, 0.932056));

    const float footWidth = 9.5f;
    float footInnerWidth = 4.5f;
    m_footWidth = footWidth;
    m_ballRadius = 3.5f;
    const float yReachFwd = 2.0f;
    const float yReachSide = 20.0f;
    const float xMin = 9.0f - 1;
    const float xReachFwd = xMin + 10.0f;
    const float xReachSide = 30.0f;
    float yMin = footInnerWidth + 2.0f;
    float yMax = yMin + footWidth - 2.0;

    m_intialWeightShiftPercentage = 0.9f;
#ifdef TARGET_IS_NAOWEBOTS
    m_intialWeightShiftPercentage = 0.9;
    yMin = footInnerWidth;
#endif

    // These boxes are relative to the supporting foot at the ankle's attachment point
    LeftFootForwardKickableArea = Rectangle(xMin, xReachFwd, (yMin), yMax);
    RightFootForwardKickableArea = Rectangle(xMin, xReachFwd, -(yMax), -(yMin));

    LeftFootRightKickableArea = Rectangle(xMin, xReachSide, footWidth/2.0, yReachSide);
    //LeftFootLeftKickableArea = Rectangle(xMin, xReachSide, 2.0f*footWidth, 3.0f/2.0f*footWidth + yReachSide);
    LeftFootLeftKickableArea = Rectangle();

    RightFootLeftKickableArea = Rectangle(xMin, xReachSide, -footWidth/2.0, -yReachSide);
    //RightFootRightKickableArea = Rectangle(xMin, xReachSide, -2.0f*footWidth, -3.0f/2.0f*footWidth - yReachSide);
    RightFootRightKickableArea = Rectangle();
}

std::string NAOKick::toString(swingDirection_t theSwingDirection)
{
    std::string result;
    switch(theSwingDirection)
    {
    case ForwardSwing:
        result = "Forward";
        break;
    case LeftSwing:
        result = "Left";
        break;
    case RightSwing:
        result = "Right";
        break;
    default:
        result = "None";
        break;
    }
    return result;
}

std::string NAOKick::toString(poseType_t thePose)
{
    std::string result;
    switch(thePose)
    {
    case DO_NOTHING:
        result = "DO_NOTHING";
        break;
    case LIFT_LEG:
        result = "LIFT_LEG";
        break;
    case ADJUST_YAW:
        result = "ADJUST_YAW";
        break;
    case SET_LEG:
        result = "SET_LEG";
        break;
    case POISE_LEG:
        result = "POISE_LEG";
        break;
    case SWING:
        result = "SWING";
        break;
    case RETRACT:
        result = "RETRACT";
        break;
    case REALIGN_LEGS:
        result = "REALIGN_LEGS";
        break;
    case UNSHIFT_LEG:
        result = "UNSHIFT_LEG";
        break;
    case ALIGN_BALL:
        result = "ALIGN_BALL";
        break;
    case ALIGN_SIDE:
        result = "ALIGN_SIDE";
        break;
    case EXTEND_SIDE:
        result = "EXTEND_SIDE";
        break;
    case RESET:
        result = "RESET";
        break;
    case NO_KICK:
        result = "NO_KICK";
        break;
    case PRE_KICK:
        result = "PRE_KICK";
        break;
    case POST_KICK:
        result = "POST_KICK";
        break;
    case TRANSFER_TO_SUPPORT:
        result = "TRANSFER_TO_SUPPORT";
        break;
    default:
        result = "Unknown";
        break;
    }
    return result;
}

///*! @brief Returns true if the kick is using the head */
//bool NAOKick::isUsingHead()
//{
//    bool usingHead;
//    switch (pose)
//    {
//        case PRE_KICK:
//        case DO_NOTHING:
//        case POST_KICK:
//        case TRANSFER_TO_SUPPORT:
//        case UNSHIFT_LEG:
//            usingHead = false;
//            break;
//        default:
//            usingHead = true;
//            break;
//    }
//    return usingHead;
//}

/*! @brief Kills the kick module
 */
void NAOKick::kill()
{
    #if DEBUG_NUMOTION_VERBOSITY > 3
    debug << "Kick kill called." << endl;
    #endif
    pose = DO_NOTHING;
    m_stateCommandGiven = false;
    m_estimatedStateCompleteTime = m_data->CurrentTime;
    m_kick_enabled = false;
    m_kick_ready = false;
}

void NAOKick::stopLegs()
{   // if another module wants to use the legs, then we should stop
    #if DEBUG_NUMOTION_VERBOSITY > 3
    debug << "Kick stop called." << endl;
    #endif
    // Chose the state that can be transitioned to allowing kick to finish as soon as possible.
    switch(pose)
    {
        case PRE_KICK:
            pose = POST_KICK;
            m_kicking_leg = noLeg;
            break;
        case TRANSFER_TO_SUPPORT:
            pose = UNSHIFT_LEG;
            break;
            /*// Don't want to do these since we sometimes occlude the ball while kicking.
        case LIFT_LEG:
            pose = REALIGN_LEGS;
            break;
        case POISE_LEG:
            pose = RETRACT;
        case SWING:
            pose = RETRACT;
            break;
            */
        default:
            pose = pose;
    }
}

void NAOKick::kickToPoint(const vector<float>& position, const vector<float>& target)
{
    // Evaluate the kicking target to start a new kick, or change a kick in progress.
	m_ball_x = position[0];
	m_ball_y = position[1];
	
	m_target_x = target[0];
	m_target_y = target[1];

    #if DEBUG_NUMOTION_VERBOSITY > 4
        debug << "void NAOKick::kickToPoint( (" << position[0] << "," << position[1] << "),(" << target[0] << "," << target[1] << ") )" << endl;
        debug << "current pose = " << toString(pose) << endl;
    #endif

    if(!isActive())
    {
        #if DEBUG_NUMOTION_VERBOSITY > 4
        debug << "NAOKick::Choosing leg." << endl;
        #endif
        m_kick_ready = chooseLeg();

    }
    else
    {
        if(kickAbortCondition())
        {
            #if DEBUG_NUMOTION_VERBOSITY > 4
            debug << "NAOKick::Aborting kick." << endl;
            stop();
            #endif
        }
    }
}

void NAOKick::doKick()
{
    #if DEBUG_NUMOTION_VERBOSITY > 4
        debug << "NAOKick::doKick()" << endl;
    #endif
    bool done = false;
    float balanceYoffset = 0.0f;
    float balanceXoffset = 3.0f;
    KickingLeg supportLeg = noLeg;
    if(m_kicking_leg == rightLeg)
    {
        supportLeg = leftLeg;
    }
    else if(m_kicking_leg == leftLeg)
    {
        supportLeg = rightLeg;
    }
    else
    {
        pose = POST_KICK;
    }

    //currently must be at zero position
    double kickAngle = atan2(m_target_y-m_ball_y, m_target_x-m_ball_x);
    double kickDistance = sqrt(pow(m_target_y-m_ball_y,2) + pow(m_target_x-m_ball_x,2));

    switch(pose)
    {
        case DO_NOTHING:
        {
                break;
        }
        case PRE_KICK:
        {
            done = doPreKick();
            if(done && !m_pauseState)
            {
                #if DEBUG_NUMOTION_VERBOSITY > 3
                cout << "Pre kick complete!" << endl;
                debug << "Pre kick complete!" << endl;
                #endif
                pose = TRANSFER_TO_SUPPORT;
            }
            break;
            }
        case POST_KICK:
        {
            done = doPostKick();
            if(done && !m_pauseState)
            {
                #if DEBUG_NUMOTION_VERBOSITY > 3
                cout << "Post kick complete!" << endl;
                debug << "Post kick complete!" << endl;
                #endif
                pose = DO_NOTHING;
            }
            break;
        }
        case TRANSFER_TO_SUPPORT:
        {
            // Shift the weight of the robot to the support leg.
            //done = ShiftWeightToFoot(supportLeg,1.0f,0.01, 1500);
            done = ShiftWeightToFootClosedLoop(supportLeg, m_intialWeightShiftPercentage, 0.3);
            if(done && !m_pauseState)
            {
                #if DEBUG_NUMOTION_VERBOSITY > 3
                cout << "Weight now on support foot!" << endl;
                debug << "Weight now on support foot!" << endl;
                #endif
                pose = LIFT_LEG;
            }
            break;
        }

        case LIFT_LEG:
        {
            done = LiftKickingLeg(m_kicking_leg, 1.5f);
            BalanceCoP(supportLeg);
            if(m_swingDirection == ForwardSwing)
            {
                if(!m_armCommandSent)
                {
                    MoveArmsToKickPose(m_kicking_leg, 0.7f);
                    m_armCommandSent = true;
                }
            }
            if(done && !m_pauseState)
            {
                #if DEBUG_NUMOTION_VERBOSITY > 3
                cout << "Leg is now lifted!" << endl;
                debug << "Leg is now lifted!" << endl;
                #endif
                if(m_swingDirection == ForwardSwing)
                {
                    m_armCommandSent = false;
                    pose = POISE_LEG;
                }
                else
                {
                    pose = ALIGN_SIDE;
                }
            }
            break;
        }

        case ADJUST_YAW:
        {
            break;
        }

        case SET_LEG:
        {
            break;
        }

        case POISE_LEG:
        {
            done = doPoise(m_kicking_leg, 0.5, 1.5f);
            if(m_kicking_leg == leftLeg)
            {
                BalanceCoP(supportLeg,balanceXoffset,balanceYoffset);
            }
            else
            {
                BalanceCoP(supportLeg,balanceXoffset,balanceYoffset);
            }

            if(done && !m_pauseState)
            {
                #if DEBUG_NUMOTION_VERBOSITY > 3
                cout << "Leg is now poised!" << endl;
                debug << "Leg is now poised!" << endl;
                #endif
                pose = SWING;
            }
            break;
        }

        case ALIGN_BALL:
        {
            done = AlignYposition(m_kicking_leg, 0.01, m_ball_y);
            BalanceCoP(supportLeg);
            if(done && !m_pauseState)
            {
                #if DEBUG_NUMOTION_VERBOSITY > 3
                cout << "Ball is now aligned!" << endl;
                debug << "Ball is now aligned!" << endl;
                #endif
                pose = SWING;
            }
           break;
        }

        case ALIGN_SIDE:
        {
            float yTarget;
            if(m_swingDirection == LeftSwing)
            {
                yTarget = m_ball_y - (m_footWidth/2.0f + m_ballRadius);
            }
            else
            {
                yTarget = m_ball_y + (m_footWidth/2.0f + m_ballRadius);
            }
            done = AlignYposition(m_kicking_leg, 0.01, yTarget);
            BalanceCoP(supportLeg,balanceXoffset,balanceYoffset);
            if(done && !m_pauseState)
            {
                #if DEBUG_NUMOTION_VERBOSITY > 3
                cout << "Kicking width is now aligned!" << endl;
                debug << "Kicking width is now aligned!" << endl;
                #endif
                pose = EXTEND_SIDE;
            }
           break;
        }

        case EXTEND_SIDE:
        {
            done = AlignXposition(m_kicking_leg, 0.01, m_ball_x);
            BalanceCoP(supportLeg, balanceXoffset, balanceYoffset);
            if(done && !m_pauseState)
            {
                #if DEBUG_NUMOTION_VERBOSITY > 3
                cout << "Kicking depth is now aligned!" << endl;
                debug << "Kicking depth is now aligned!" << endl;
                #endif
                pose = SWING;
            }
           break;
        }


        case SWING:
        {
                if(m_swingDirection == ForwardSwing)
                {
                    float kickSpeed = CalculateForwardSwingSpeed(kickDistance);
                    #if DEBUG_NUMOTION_VERBOSITY > 4
                    debug << "Kicking Distance: " << kickDistance << endl;
                    debug << "Swinging at speed: " << kickSpeed << endl;
                    #endif
                    done = SwingLegForward(m_kicking_leg, kickSpeed);
                    if(!m_armCommandSent)
                    {
                        MoveArmsToKickPose(supportLeg, kickSpeed);
                        m_armCommandSent = true;
                    }
                }
                else if( (m_swingDirection == LeftSwing) || (m_swingDirection == RightSwing))
                {
                    done = SwingLegSideward(m_kicking_leg, CalculateSidewardSwingSpeed(kickDistance));
                }

                BalanceCoP(supportLeg, balanceXoffset, balanceYoffset);
                if(done && !m_pauseState)
                {
                    m_armCommandSent = false;
                    #if DEBUG_NUMOTION_VERBOSITY > 3
                    cout << "Swing completed!" << endl;
                    debug << "Swing completed!" << endl;
                    #endif
                    pose = RETRACT;
                }
                break;
        }
        case RETRACT:
            {
                done = LiftKickingLeg(m_kicking_leg, 1.5f);
                //BalanceCoP(supportLeg,balanceXoffset,balanceYoffset);
                if(done && !m_pauseState)
                {
                    #if DEBUG_NUMOTION_VERBOSITY > 3
                    cout << "Leg Retracted!" << endl;
                    debug << "Leg Retracted!" << endl;
                    #endif
                    pose = REALIGN_LEGS;
                }
                break;
            }
        case REALIGN_LEGS:
            {
                done = LowerLeg(m_kicking_leg, 0.7f);
                //BalanceCoP(supportLeg,balanceXoffset,balanceYoffset);
                if(done && !m_pauseState)
                {
                    #if DEBUG_NUMOTION_VERBOSITY > 3
                    cout << "Legs Aligned!" << endl;
                    debug << "Legs Aligned!" << endl;
                    #endif
                    pose = UNSHIFT_LEG;
                }
                break;
            }
        case UNSHIFT_LEG:
        {
                //done = ShiftWeightToFoot(m_kicking_leg,0.5f,0.01f, 500.0f);
                done = ShiftWeightToFootClosedLoop(supportLeg, 0.5f, 0.3);
                if(done && !m_pauseState)
                {
                    #if DEBUG_NUMOTION_VERBOSITY > 3
                    cout << "Weight Unshifted!" << endl;
                    debug << "Weight Unshifted!" << endl;
                    #endif
                    pose = POST_KICK;
                }
                break;
        }

        case RESET:
        {
                break;
        }

        case NO_KICK:
        {
                break;
        }

        default:
        {
                pose = NO_KICK;
                break;
        }
    }
}

bool NAOKick::doPreKick()
{
    #if DEBUG_NUMOTION_VERBOSITY > 3
    debug << "Pre - Kick" << endl;
    #endif
    bool validData = true;
    vector<float>leftJoints;
    vector<float>rightJoints;
    validData = validData && m_data->getPosition(NUSensorsData::LLeg,leftJoints);
    validData = validData && m_data->getPosition(NUSensorsData::RLeg,rightJoints);
    validData = validData && (leftJoints.size() >= 6) && (rightJoints.size() >= 6);

    vector<float> leftArmJoints;
    vector<float> rightArmJoints;
    validData = validData && m_data->getPosition(NUSensorsData::LArm,leftArmJoints);
    validData = validData && m_data->getPosition(NUSensorsData::RArm,rightArmJoints);
    
    if (not validData)
        return false;
    
    static double endWaitTime = 0;
    if (m_kickWait)
    {
        endWaitTime = m_data->CurrentTime + 1200;
        m_kickWait = false;
        debug << "Pre - Kick. Updated endWaitTime: " << endWaitTime << endl;
    }

    if(!m_stateCommandGiven and m_data->CurrentTime - endWaitTime > 0)
    {
        vector<float> armpos (4, 0.0f);
        armpos[1] = PI/2.0f;
        float maxSpeed = 0.7;
        double leftTime,rightTime;
        leftTime = MoveLimbToPositionWithSpeed(NUActionatorsData::LLeg, leftJoints, m_leftLegInitialPose, maxSpeed , 75.0, 1);
        rightTime = MoveLimbToPositionWithSpeed(NUActionatorsData::RLeg, rightJoints, m_rightLegInitialPose, maxSpeed , 75.0, 1);
        armpos[0] = PI/8.0;
        armpos[3] = -PI/2.0f;
        MoveLimbToPositionWithSpeed(NUActionatorsData::LArm, leftArmJoints, armpos, 2*maxSpeed , m_defaultArmMotorGain);
        armpos[0] = -PI/8.0;
        armpos[3] = PI/2.0f;
        MoveLimbToPositionWithSpeed(NUActionatorsData::RArm, rightArmJoints, armpos, 2*maxSpeed , m_defaultArmMotorGain);
        m_stateCommandGiven = true;
        m_estimatedStateCompleteTime = max(rightTime,leftTime);
        debug << "Moving to Initial Position: Estimated Completion Time = " << m_estimatedStateCompleteTime << endl;
    }
    
    if(m_stateCommandGiven)
    {
        if((allEqual(m_leftLegInitialPose, leftJoints, 0.05f) && allEqual(m_leftLegInitialPose, rightJoints, 0.05f)) || (m_data->CurrentTime - m_estimatedStateCompleteTime > 200.0))
        {
            m_stateCommandGiven = false;
            return true;
        }
    }
    return false;
}

bool NAOKick::doPostKick()
{
    #if DEBUG_NUMOTION_VERBOSITY > 4
    debug << "NAOKick::doPostKick";
    #endif
    if(m_stateCommandGiven)
    {
        #if DEBUG_NUMOTION_VERBOSITY > 4
        debug << "- Waiting for action to complete: ";
        debug << m_data->CurrentTime << " / " << m_estimatedStateCompleteTime << endl;
        #endif
    }
    bool done;
    if(m_stateCommandGiven && (m_data->CurrentTime > m_estimatedStateCompleteTime))
        m_stateCommandGiven = false;
    done = !m_stateCommandGiven;
    if(done)
    {
        m_kicking_leg = noLeg;
        m_kick_enabled = false;
        m_kick_ready = false;
    }
    return done;
}

bool NAOKick::ShiftWeightToFoot(KickingLeg targetLeg, float targetWeightPercentage, float speed, float time)
{
    const float maxShiftSpeed = speed * SpeedMultiplier();
    const float reqiredAccuracy = 0.1; // 10%
    const float propGain = 0.05f;
    bool validData = true;
    bool leftContact, rightContact;
    validData = validData && m_data->getContact(NUSensorsData::LLeg, leftContact);
    validData = validData && m_data->getContact(NUSensorsData::RLeg, rightContact);
    if(!validData) return false;
    vector<float> lcop, rcop;
    float lcopx(0.0f),lcopy(0.0f),rcopx(0.0f),rcopy(0.0f);
    float lforce(0.0f),rforce(0.0f);

    if(leftContact)
    {
        validData = validData && m_data->getCoP(NUSensorsData::LLeg, lcop);
        validData = validData && m_data->getForce(NUSensorsData::LLeg, lforce);
    }
    if(rightContact)
    {
        validData = validData && m_data->getCoP(NUSensorsData::RLeg, rcop);
        validData = validData && m_data->getForce(NUSensorsData::RLeg, rforce);
    }



    vector<float>leftJoints;
    vector<float>rightJoints;
    validData = validData && m_data->getTarget(NUSensorsData::LLeg,leftJoints);
    validData = validData && m_data->getTarget(NUSensorsData::RLeg,rightJoints);
    validData = validData && (leftJoints.size() >= 6) && (rightJoints.size() >= 6);

    if(validData)
    {
        float weightPercentage;
        float weightError = 0;
        bool recentImpact(false);
        float newHipPos(0.0f);
        if(targetLeg == rightLeg)
        {
            weightPercentage = rforce / (rforce + lforce);
            weightError = targetWeightPercentage - weightPercentage;
            newHipPos = rightJoints[0] + crop(weightError*propGain, -maxShiftSpeed, maxShiftSpeed);
        }
        else if(targetLeg == leftLeg)
        {
            weightPercentage = lforce / (rforce + lforce);
            weightError = weightPercentage - targetWeightPercentage;
            newHipPos = leftJoints[0] + crop(weightError*propGain, -maxShiftSpeed, maxShiftSpeed);
        }
        if(!m_stateCommandGiven)
        {
            m_stateCommandGiven = true;
            m_estimatedStateCompleteTime = m_data->CurrentTime + time;
        }
        if(fabs(weightError) > reqiredAccuracy)
        {
            leftJoints[0] = newHipPos;
            rightJoints[0] = newHipPos;
            leftJoints[4] = -newHipPos;
            rightJoints[4] = -newHipPos;
            m_actions->add(NUActionatorsData::LLeg, Platform->getTime(), leftJoints, m_defaultMotorGain);
            m_actions->add(NUActionatorsData::RLeg, Platform->getTime(), rightJoints, m_defaultMotorGain);
        }
        else
        {
            BalanceCoP(targetLeg);
        }
        //float elapsedTime = m_data->CurrentTime - (m_estimatedStateCompleteTime - time);
        if(m_stateCommandGiven && !recentImpact && (m_estimatedStateCompleteTime < m_data->CurrentTime))
        {
            m_stateCommandGiven = false;
            return true;
        }
    }
    return false;
}


bool NAOKick::ShiftWeightToFootClosedLoop(KickingLeg p_targetLeg, float targetWeightPercentage, float speed)
{
    bool validData = true;
    NUData::id_t targetLeg;
    NUData::id_t otherLeg;
    float targetDisplacement;
    if(p_targetLeg == rightLeg)
    {
        targetLeg = NUData::RLeg;
        otherLeg = NUData::LLeg;
        targetDisplacement = 2 * (0.5 - targetWeightPercentage) * 5.0f;
    }
    else if(p_targetLeg == leftLeg)
    {
        targetLeg = NUData::LLeg;
        otherLeg = NUData::RLeg;
        targetDisplacement = 2 * (targetWeightPercentage - 0.5) * 5.0f;
    }
    else return true;

    vector<float> targetLegPositions;
    vector<float> otherLegPositions;
    validData = validData && m_data->getTarget(targetLeg, targetLegPositions);
    validData = validData && m_data->getTarget(otherLeg, otherLegPositions);

    validData = validData && (targetLegPositions.size() >= 6) && (otherLegPositions.size() >= 6);

    static vector<float> legPositionTargets;

    if(validData)
    {
        if(!m_stateCommandGiven)
        {
            const float thigh_length = 10.0f;
            const float tibia_length = 10.0f;
            float targetLegLength = sqrt( pow(thigh_length,2) + pow(tibia_length,2) - 2*thigh_length*tibia_length*cos(mathGeneral::PI/2.0f - targetLegPositions[3]));
            float otherLegLength = sqrt( pow(thigh_length,2) + pow(tibia_length,2) - 2*thigh_length*tibia_length*cos(mathGeneral::PI/2.0f - otherLegPositions[3]));
            float targetLength;
            if(targetLegLength < otherLegLength)
            {
                targetLength = targetLegLength;
                legPositionTargets = targetLegPositions;
            }
            else
            {
                targetLength = otherLegLength;
                legPositionTargets = otherLegPositions;
            }

            debug << "targetDisplacement = " << targetDisplacement << endl;
            debug << "targetLength = " << targetLength << endl;
            float targetAnkleRoll = asin(targetDisplacement / targetLength);
            debug << "targetAnkleRoll = " << targetAnkleRoll << endl;
            legPositionTargets[0] = -targetAnkleRoll;
            legPositionTargets[4] = targetAnkleRoll;
            m_estimatedStateCompleteTime = MoveLegsToPositionWithSpeed(legPositionTargets,speed,m_defaultMotorGain,1.0);

            m_stateCommandGiven = true;
            #if DEBUG_NUMOTION_VERBOSITY > 4
                debug << "NAOKick::ShiftWeightToFootClosedLoop - Sending move command ";
                debug << "Estimated completion time = " << m_estimatedStateCompleteTime << endl;
            #endif
        }

        if((allEqual(targetLegPositions, legPositionTargets, 0.05f) && allEqual(otherLegPositions, legPositionTargets, 0.05f)) || (m_data->CurrentTime - m_estimatedStateCompleteTime > 200.0))
        {
            #if DEBUG_NUMOTION_VERBOSITY > 4
            if((m_data->CurrentTime - m_estimatedStateCompleteTime > 200.0))
            {
                debug << "State timed out" << endl;
            }
            else
            {
                debug << "targetLegPositions[4] = " << targetLegPositions[4] << endl;
                debug << "otherLegPositions[4] = " << otherLegPositions[4] << endl;
                debug << "legPositionTargets[4] = " << legPositionTargets[4] << endl;
                debug << "Targets Reached" << endl;
            }
            #endif
            m_stateCommandGiven = false;
            return true;
        }
    }
    return false;
}


bool NAOKick::LiftKickingLeg(KickingLeg p_kickingLeg, float speed)
{
    bool validData = true;
    NUData::id_t kickingLeg;
    NUData::id_t supportLeg;
    if(p_kickingLeg == rightLeg)
    {
        kickingLeg = NUData::RLeg;
        supportLeg = NUData::LLeg;
    }
    else if(p_kickingLeg == leftLeg)
    {
        kickingLeg = NUData::LLeg;
        supportLeg = NUData::RLeg;
    }
    else return true;

    bool kickContact;
    validData = validData && m_data->getContact(kickingLeg, kickContact);
    if(!validData) return false;

    vector<float> kickLegPositions, supportLegPositions;
    validData = validData && m_data->getTarget(kickingLeg,kickLegPositions);
    validData = validData && m_data->getTarget(supportLeg,supportLegPositions);

    validData = validData && (kickLegPositions.size() >= 6);
    static vector<float>kickLegTargets;
    if(validData)
    {
        if(!m_stateCommandGiven)
        {
            kickLegTargets = supportLegPositions;
            kickLegTargets[3] = 1.4f;
            kickLegTargets[1] = -kickLegTargets[3] / 2.0f;
            kickLegTargets[5] = -kickLegTargets[3] / 2.0f;
            m_estimatedStateCompleteTime = MoveLimbToPositionWithSpeed(kickingLeg, kickLegPositions, kickLegTargets, speed, 75.0, 1.0);

            m_stateCommandGiven = true;
            #if DEBUG_NUMOTION_VERBOSITY > 4
            debug << "NAOKick::LiftKickingLeg - Motion Command Given.";
            debug << " Estimated Completion Time = " << m_estimatedStateCompleteTime << endl;
            #endif
        }

        if(allEqual(kickLegTargets, kickLegPositions, 0.05f) || (m_data->CurrentTime - m_estimatedStateCompleteTime > 200.0))
        {
            m_stateCommandGiven = false;
            return true;
        }
    }
    return false;
}

bool NAOKick::doPoise(KickingLeg poiseLeg, float angleChange, float speed)
{
    bool validData = true;
    NUData::id_t kickingLeg;
    if(poiseLeg == rightLeg)
        kickingLeg = NUSensorsData::RLeg;
    else if(poiseLeg == leftLeg)
        kickingLeg = NUSensorsData::LLeg;
    else return true;

    bool kickContact;
    validData = validData && m_data->getContact(kickingLeg, kickContact);
    if(!validData) return false;

    vector<float>kickLegPositions;
    validData = validData && m_data->getTarget(kickingLeg,kickLegPositions);
    validData = validData && (kickLegPositions.size() >= 6);

    static vector<float>kickLegTargets;

    if(validData)
    {
        if(!m_stateCommandGiven)
        {
            kickLegTargets.assign(kickLegPositions.begin(), kickLegPositions.end());
            kickLegTargets[1] += angleChange;
            kickLegTargets[3] += angleChange;
            FlattenFoot(kickLegTargets);
            LimitJoints(poiseLeg, kickLegTargets);

            m_estimatedStateCompleteTime = MoveLimbToPositionWithSpeed(kickingLeg, kickLegPositions, kickLegTargets, speed , m_defaultMotorGain, 1.0);
            m_stateCommandGiven = true;
            #if DEBUG_NUMOTION_VERBOSITY > 4
            debug << "NAOKick::doPoise - Motion Command Given.";
            debug << " Estimated completion time = " << m_estimatedStateCompleteTime << endl;
            #endif
        }
        if(allEqual(kickLegTargets, kickLegPositions, 0.05f) || (m_data->CurrentTime - m_estimatedStateCompleteTime > 200.0))
        {
            m_stateCommandGiven = false;
            return true;
        }
    }
    return false;
}

double NAOKick::TimeBetweenFrames()
{
    return (m_currentTimestamp - m_previousTimestamp);
}

float NAOKick::perSec2perFrame(float value)
{
    return value * (TimeBetweenFrames() / 1000.0);
}

float NAOKick::SpeedMultiplier()
{
    return TimeBetweenFrames() / 20.0f;
}

float NAOKick::GainMultiplier()
{
    return TimeBetweenFrames() / 20.0f;
}

bool NAOKick::LimitJoints(KickingLeg leg, vector<float> jointPositions)
{
    bool changed = false;
    float previousValue;
    if(leg == rightLeg)
    {
        for(unsigned int i = 0; i < jointPositions.size(); i++)
        {
            previousValue = jointPositions[i];
            jointPositions[i] = crop(previousValue, m_rightLegLimits[i].min, m_rightLegLimits[i].max);
            changed = changed || (jointPositions[i] != previousValue);
        }
    }
    else if (leg == leftLeg)
    {
        for(unsigned int i = 0; i < jointPositions.size(); i++)
        {
            previousValue = jointPositions[i];
            jointPositions[i] = crop(previousValue, m_leftLegLimits[i].min, m_leftLegLimits[i].max);
            changed = changed || (jointPositions[i] != previousValue);
        }
    }
    return changed;
}


bool NAOKick::IsPastTime(float time){
    return (m_data->CurrentTime > time);
}

bool NAOKick::BalanceCoP(KickingLeg p_supportLeg, float targetX, float targetY)
{
    bool validData = true;

    NUData::id_t supportLeg;
    if(p_supportLeg == rightLeg)
        supportLeg = NUData::RLeg;
    else if(p_supportLeg == leftLeg)
        supportLeg = NUSensorsData::LLeg;
    else return true;

    bool supportFootContact, isSupport;
    validData = validData && m_data->getContact(supportLeg, supportFootContact);
    validData = validData && m_data->getSupport(supportLeg, isSupport);
    if(validData && supportFootContact)
    {
        vector<float>supportLegJoints;
        validData = validData && m_data->getTarget(supportLeg, supportLegJoints);
        validData = validData && (supportLegJoints.size() >= 6);
		vector<float> cop;
        float force(0.0f);
        validData = validData && m_data->getCoP(supportLeg, cop);
        validData = validData && m_data->getForce(supportLeg,force);
        if(validData)
        {
            BalanceCoPLevelTorso(p_supportLeg, supportLegJoints, cop[0], cop[1], targetX, targetY);
            //BalanceCoPHipAndAnkle(supportLegJoints, copx, copy);
            LimitJoints(p_supportLeg,supportLegJoints);
            vector<float> vel (6, 0);
            vector<float> gain (6, m_defaultMotorGain);
            m_actions->add(supportLeg, m_data->CurrentTime, supportLegJoints, gain);
            return true;
        }
    }
    return false;
}

void NAOKick::BalanceCoPLevelTorso(KickingLeg theLeg, vector<float>& jointAngles, float CoPx, float CoPy, float targetX, float targetY)
{
    // Linear controller to centre CoP
    const float gainx = -0.01 * GainMultiplier();
    const float gainy = 0.01 * GainMultiplier();
    const float targetCoPx = targetX;
    const float targetCoPy = targetY;

    const float deltax = targetCoPx - CoPx;
    const float deltay = targetCoPy - CoPy;
//    const float gainx = m_variableGainValue * GainMultiplier();
//    const float gainy = m_variableGainValue * GainMultiplier();

    jointLimit hipRollLimits;
    if(theLeg == leftLeg)
    {
        hipRollLimits = m_leftLegLimits[0];
    }
    else
    {
        hipRollLimits = m_rightLegLimits[0];
    }

    // Roll correction
    // Ankle Roll
    float newAnkleRoll;
    newAnkleRoll = jointAngles[4] + gainy * asin(deltay / 35.0);
    if( (newAnkleRoll > hipRollLimits.max) || (newAnkleRoll < hipRollLimits.min))
    {
        jointAngles[0] += gainy * asin(deltay / 35.0);
    }
    else
    {
        // Hip Roll - Reverse of pitch to maintain vertical torso.
        jointAngles[4] = newAnkleRoll;
        jointAngles[0] = -jointAngles[4];
    }

    // Pitch correction
    jointAngles[5] += gainx * asin(deltax / 35.0f);
    // Hip Pitch - Reverse of pitch to maintain vertical torso.
    jointAngles[1] = -(jointAngles[3] + jointAngles[5]);
    
    return;
}

void NAOKick::BalanceCoPHipAndAnkle(vector<float>& jointAngles, float CoPx, float CoPy, float targetX, float targetY)
{
    // Linear controller to centre CoP
//    const float gainx = 0.008 * GainMultiplier();
//    const float gainy = 0.008 * GainMultiplier();
    const float gainx = m_variableGainValue * GainMultiplier();
    const float gainy = m_variableGainValue * GainMultiplier();
    // Roll correction
    // Ankle Roll
    jointAngles[4] -= gainy * asin(CoPy / 35.0);
    // Hip Roll - Reverse of pitch to maintain vertical torso.
    jointAngles[0] -= 10 * gainy * asin(CoPy / 35.0);

    // Pitch correction
    jointAngles[5] -= gainx * asin(CoPx / 35.0f);
    // Hip Pitch - Reverse of pitch to maintain vertical torso.
    jointAngles[1] -= 10 * gainx * asin(CoPx / 35.0f);
    return;
}

bool NAOKick::AlignXposition(KickingLeg p_kickingLeg, float speed, float xPos)
{
    const float gain = 0.01;
    bool validData = true;
    NUData::id_t kickingLeg;
    NUActionatorsData::id_t kickingHipPitch;
    NUActionatorsData::id_t kickingKneePitch;
    NUActionatorsData::id_t kickingAnklePitch;
    jointLimit hipPitchJointLimits;
    jointLimit kneePitchJointLimits;
    jointLimit anklePitchJointLimits;

    vector<float> sltransform, kltransform;
    Matrix supportLegTransform, kickingLegTransform;
    m_data->get(NUSensorsData::SupportLegTransform, sltransform);
    supportLegTransform = Matrix4x4fromVector(sltransform);
    if(p_kickingLeg == rightLeg)
    {
        kickingLeg = NUData::RLeg;
        kickingHipPitch = NUActionatorsData::RHipPitch;
        kickingKneePitch = NUActionatorsData::RKneePitch;
        kickingAnklePitch = NUActionatorsData::RAnklePitch;
        validData = validData && m_data->get(NUSensorsData::RLegTransform, kltransform);
        kickingLegTransform = Matrix4x4fromVector(kltransform);
        hipPitchJointLimits = m_rightLegLimits[1];
        kneePitchJointLimits = m_rightLegLimits[3];
        anklePitchJointLimits = m_rightLegLimits[5];
    }
    else if(p_kickingLeg == leftLeg)
    {
        kickingLeg = NUSensorsData::LLeg;
        kickingHipPitch = NUActionatorsData::LHipPitch;
        kickingKneePitch = NUActionatorsData::LKneePitch;
        kickingAnklePitch = NUActionatorsData::LAnklePitch;
        validData = validData && m_data->get(NUSensorsData::LLegTransform, kltransform);
        kickingLegTransform = Matrix4x4fromVector(kltransform);
        hipPitchJointLimits = m_leftLegLimits[1];
        kneePitchJointLimits = m_leftLegLimits[3];
        anklePitchJointLimits = m_leftLegLimits[5];
    }
    else return true;

    vector<float>kickLegJoints;
    vector<float>kickLegPositions;
    validData = validData && m_data->getTarget(kickingLeg,kickLegJoints);
    validData = validData && m_data->getPosition(kickingLeg,kickLegPositions);
    validData = validData && (kickLegJoints.size() >= 6);
    vector<float> footPosition = Kinematics::PositionFromTransform(kickingLegTransform);
    float deltaX = xPos - footPosition[0];
    bool jointLimitReached = false;
    if(validData)
    {
        const float targetHeight = 5.0f;

        float currentHeightOffGround = m_kinematicModel->CalculateRelativeFootHeight(supportLegTransform,kickingLegTransform);
        #if DEBUG_NUMOTION_VERBOSITY > 4
        debug << "X Position = " << xPos << "foot Position = (" << footPosition[0] << "," << footPosition[1] << "," << footPosition[2] << ")" << endl;
        debug << "Calculated height of foot from ground  = " << currentHeightOffGround << endl;
        #endif
        float deltaTheta = crop(gain*deltaX,-speed,speed);
        float calcKneePitchAngle = kickLegJoints[3] - 2*deltaTheta;

        float calcHipPitchAngle = kickLegJoints[1] - crop(gain*(targetHeight-currentHeightOffGround),-speed,speed);

        float calcAnklePitchAngle = FlatFootAnklePitch(calcHipPitchAngle,calcKneePitchAngle);
        float newHipPitchAngle = crop(calcHipPitchAngle,hipPitchJointLimits.min,hipPitchJointLimits.max);
        float newKneePitchAngle = crop(calcKneePitchAngle,kneePitchJointLimits.min,kneePitchJointLimits.max);
        float newAnklePitchAngle = crop(calcAnklePitchAngle,anklePitchJointLimits.min,anklePitchJointLimits.max);
        m_actions->add(kickingHipPitch,m_data->CurrentTime,newHipPitchAngle,m_defaultMotorGain);
        m_actions->add(kickingKneePitch,m_data->CurrentTime,newKneePitchAngle,m_defaultMotorGain);
        m_actions->add(kickingAnklePitch,m_data->CurrentTime,newAnklePitchAngle,m_defaultMotorGain);
        jointLimitReached = (newHipPitchAngle != calcHipPitchAngle) || (newKneePitchAngle != calcKneePitchAngle) || (newAnklePitchAngle != calcAnklePitchAngle);
    }
    return (fabs(deltaX) < 0.5) || jointLimitReached;
}

bool NAOKick::AlignYposition(KickingLeg p_kickingLeg, float speed, float yPos)
{
    const float gain = 0.01;
    bool validData = true;
    NUData::id_t kickingLeg;
    NUActionatorsData::id_t kickingHipRoll;
    vector<float> kltransform;
    Matrix kickingLegTransform;
    jointLimit hipJointLimits;
    if(p_kickingLeg == rightLeg)
    {
        kickingLeg = NUData::RLeg;
        kickingHipRoll = NUActionatorsData::RHipRoll;
        validData = validData && m_data->get(NUSensorsData::RLegTransform, kltransform);
        kickingLegTransform = Matrix4x4fromVector(kltransform);
        hipJointLimits = m_rightLegLimits[0];
    }
    else if(p_kickingLeg == leftLeg)
    {
        kickingLeg = NUSensorsData::LLeg;
        kickingHipRoll = NUActionatorsData::LHipRoll;
        validData = validData && m_data->get(NUSensorsData::LLegTransform, kltransform);
        kickingLegTransform = Matrix4x4fromVector(kltransform);
        hipJointLimits = m_leftLegLimits[0];
    }
    else return true;

    vector<float>kickLegJoints;
    vector<float>kickLegPositions;
    validData = validData && m_data->getTarget(kickingLeg,kickLegJoints);
    validData = validData && m_data->getPosition(kickingLeg,kickLegPositions);
    validData = validData && (kickLegJoints.size() >= 6);
    vector<float> footPosition = Kinematics::PositionFromTransform(kickingLegTransform);
    float deltaY = yPos - footPosition[1];
    #if DEBUG_NUMOTION_VERBOSITY > 4
    debug << "Delta Y = " << deltaY;
    #endif
    bool jointLimitReached = false;
    if(validData)
    {
        #if DEBUG_NUMOTION_VERBOSITY > 4
        debug << "Y Position = " << yPos << "foot Position = (" << footPosition[0] << "," << footPosition[1] << "," << footPosition[2] << ")" << endl;
        #endif
        float deltaTheta = crop(gain*deltaY,-speed,speed);
        float calcHipRollAngle = kickLegJoints[0] + deltaTheta;
        float newHipRollAngle = crop(calcHipRollAngle,hipJointLimits.min,hipJointLimits.max);
        m_actions->add(kickingHipRoll,m_data->CurrentTime,newHipRollAngle,m_defaultMotorGain);
        jointLimitReached = (newHipRollAngle != calcHipRollAngle);
    }
    return (fabs(deltaY) < 0.5) || jointLimitReached;
}

void NAOKick::FlattenFoot(vector<float>& jointAngles)
{
    jointAngles[5] = FlatFootAnklePitch(jointAngles[1], jointAngles[3]);
    jointAngles[4] = FlatFootAnkleRoll(jointAngles[0]);
    return;
}

float NAOKick::FlatFootAnklePitch(float hipPitch, float kneePitch)
{
    return -(hipPitch + kneePitch);
}

float NAOKick::FlatFootAnkleRoll(float hipRoll)
{
    return -(hipRoll);
}

void NAOKick::MaintainSwingHeight(KickingLeg supportLeg, vector<float>& supportLegJoints, KickingLeg swingLeg, vector<float>& swingLegJoints, float swingHeight)
{
    return;
}

void NAOKick::MoveArmsToKickPose(KickingLeg leadingArmleg, float speed)
{
    return;
    NUData::id_t leadingArm;
    NUData::id_t trailingArm;
    float mirroredJointMultiplier = 1.0;
    if(leadingArmleg == rightLeg)
    {
        leadingArm = NUActionatorsData::RArm;
        trailingArm = NUSensorsData::LArm;
    }
    else if(leadingArmleg == leftLeg)
    {
        leadingArm = NUActionatorsData::LArm;
        trailingArm = NUSensorsData::RArm;
        mirroredJointMultiplier *= -1.0f;
    }
    else return;

    bool validData = true;
    vector<float> leadingArmPositions,trailingArmPositions;
    validData = validData && m_data->getPosition(leadingArm,leadingArmPositions);
    validData = validData && m_data->getPosition(trailingArm,trailingArmPositions);

    if(validData)
    {
        vector<float> leadingArmTargets(leadingArmPositions),trailingArmTargets(trailingArmPositions);
        // Shoulder Yaw
        leadingArmTargets[0] = -deg2rad(35.0) * mirroredJointMultiplier;
        trailingArmTargets[0] = deg2rad(35.0) * mirroredJointMultiplier;

        // Shoulder Pitch
        leadingArmTargets[1] = PI/4.0f;
        trailingArmTargets[1] = 2.1f;

        // Elbow Yaw
        leadingArmTargets[2] = deg2rad(70.0) * mirroredJointMultiplier;
        trailingArmTargets[2] = -deg2rad(35.0) * mirroredJointMultiplier;

        // Elbow Roll
        leadingArmTargets[3] = 0.0f  * mirroredJointMultiplier;
        trailingArmTargets[3] = 0.0f * mirroredJointMultiplier;



        MoveLimbToPositionWithSpeed(leadingArm, leadingArmPositions, leadingArmTargets, speed , 75.0);
        MoveLimbToPositionWithSpeed(trailingArm, trailingArmPositions, trailingArmTargets, speed , 75.0);
    }
}

bool NAOKick::SwingLegForward(KickingLeg p_kickingLeg, float speed)
{
    float swingSpeed = speed;
    bool validData = true;

    NUData::id_t kickingLeg;
    NUActionatorsData::id_t kickingKneePitch;
    NUActionatorsData::id_t kickingHipPitch;
    NUActionatorsData::id_t kickingAnklePitch;
    if(p_kickingLeg == rightLeg)
    {
        kickingLeg = NUData::RLeg;
        kickingHipPitch = NUActionatorsData::RHipPitch;
        kickingKneePitch = NUActionatorsData::RKneePitch;
        kickingAnklePitch = NUActionatorsData::RAnklePitch;
    }
    else if(p_kickingLeg == leftLeg)
    {
        kickingLeg = NUData::LLeg;
        kickingHipPitch = NUActionatorsData::LHipPitch;
        kickingKneePitch = NUActionatorsData::LKneePitch;
        kickingAnklePitch = NUActionatorsData::LAnklePitch;
    }
    else return true;

    if(!validData) return false;

    vector<float>kickingLegJoints;
    validData = validData && m_data->getTarget(kickingLeg,kickingLegJoints);
    validData = validData && (kickingLegJoints.size() >= 6);

    const float targetHipPitch = -1.2;
    const float targetKneePitch = 0.7f;

    static float endHipAngle = 0;
    static float endKneeAngle = 0;
    static float endAnkleAngle = 0;

    if(validData)
    {

        if(m_stateCommandGiven)
        {
            if((endHipAngle-kickingLegJoints[1] >= -0.05) && (endKneeAngle-kickingLegJoints[3] >= -0.05))
                m_stateCommandGiven = false;
            else if(m_data->CurrentTime > m_estimatedStateCompleteTime)
                m_stateCommandGiven = false;
            if(!m_stateCommandGiven)
                return true;
        }

        if(!m_stateCommandGiven)
        {
            //float startHipAngle = kickingLegJoints[1];
            //float startKneeAngle = kickingLegJoints[3];
            //float startAnkleAngle = -(startHipAngle + startKneeAngle);

            endHipAngle = targetHipPitch;//startHipAngle - PI/4;
            endKneeAngle = targetKneePitch;//startKneeAngle - PI/4;
            endAnkleAngle = -(endHipAngle + endKneeAngle);

            //vector<double> hipTimes, kneeTimes, ankleTimes;
            //vector<float> hipPositions, hipVelocities, kneePositions, kneeVelocities, anklePositions, ankleVelocities;
            vector<float> kickingTargets(kickingLegJoints);
            kickingTargets[1] = endHipAngle;
            kickingTargets[3] = endKneeAngle;
            FlattenFoot(kickingTargets);
            
            m_estimatedStateCompleteTime = MoveLimbToPositionWithSpeed(kickingLeg,kickingLegJoints,kickingTargets,swingSpeed, 80.0f, 1.0f);
            m_stateCommandGiven = true;

            #if DEBUG_NUMOTION_VERBOSITY > 4
            debug << "NAOKick::SwingLegForward - Move Command Given. Estimated Completion Time = ";
            debug << m_estimatedStateCompleteTime << endl;
            #endif
        }
    }
    return false;
}

bool NAOKick::SwingLegSideward(KickingLeg p_kickingLeg, float speed)
{
    float swingSpeed = speed;
    bool validData = true;

    NUData::id_t kickingLeg;
    NUData::id_t supportLeg;
    if(p_kickingLeg == rightLeg)
    {
        kickingLeg = NUData::RLeg;
        supportLeg = NUData::LLeg;
    }
    else if(p_kickingLeg == leftLeg)
    {
        kickingLeg = NUData::LLeg;
        supportLeg = NUData::RLeg;
    }
    else return true;

    if(!validData) return false;

    vector<float>kickingLegJoints, supportLegJoints;
    validData = validData && m_data->getPosition(kickingLeg,kickingLegJoints);
    validData = validData && m_data->getPosition(supportLeg,supportLegJoints);
    validData = validData && (kickingLegJoints.size() >= 6);

    static vector<float> swingTargets;

    if(validData)
    {
        if(!m_stateCommandGiven)
        {
            swingTargets = supportLegJoints;

            swingTargets[1] = kickingLegJoints[1] - 0.125;
            swingTargets[3] = kickingLegJoints[3] + 0.25;
            FlattenFoot(swingTargets);

            m_estimatedStateCompleteTime = MoveLimbToPositionWithSpeed(kickingLeg, kickingLegJoints, swingTargets, swingSpeed , 100.0);
            m_stateCommandGiven = true;
            #if DEBUG_NUMOTION_VERBOSITY > 4
            debug << "NAOKick::SwingLegSideward - Move Command Given. Estimated Completion Time = ";
            debug << m_estimatedStateCompleteTime << endl;
            #endif
        }
        if(allEqual(swingTargets, kickingLegJoints, 0.05f) || (m_data->CurrentTime - m_estimatedStateCompleteTime > 200.0))
        {
            m_stateCommandGiven = false;
            return true;
        }
    }
    return false;
}

bool NAOKick::LowerLeg(KickingLeg p_kickingLeg, float speed)
{
    bool validData = true;
    NUData::id_t kickingLeg;
    NUData::id_t supportLeg;
    if(p_kickingLeg == rightLeg)
    {
        kickingLeg = NUData::RLeg;
        supportLeg = NUData::LLeg;
    }
    else if(p_kickingLeg == leftLeg)
    {
        kickingLeg = NUData::LLeg;
        supportLeg = NUData::RLeg;
    }
    else return true;

    bool kickContact;
    validData = validData && m_data->getContact(kickingLeg, kickContact);
    if(!validData) return false;

    vector<float>kickLegPositions, supportLegPositions;
    bool kickingLegContact;
    validData = validData && m_data->getTarget(kickingLeg,kickLegPositions);
    validData = validData && m_data->getTarget(supportLeg,supportLegPositions);
    validData = validData && m_data->getContact(kickingLeg, kickingLegContact);

    validData = validData && (kickLegPositions.size() >= 6);

    static vector<float>kickLegTargets;

    if(validData)
    {
        if(!m_stateCommandGiven)
        {
            kickLegTargets = supportLegPositions;
            m_estimatedStateCompleteTime = MoveLimbToPositionWithSpeed(kickingLeg, kickLegPositions, kickLegTargets, speed, 50.0, 1.0f);
            m_stateCommandGiven = true;
            #if DEBUG_NUMOTION_VERBOSITY > 4
            debug << "Motion Command Given." << endl;
            debug << "NAOKick::LowerLeg - Move Command Given. Estimated Completion Time = ";
            debug << m_estimatedStateCompleteTime << endl;
            #endif
        }

        if(allEqual(kickLegTargets, kickLegPositions, 0.05f) || (m_data->CurrentTime - m_estimatedStateCompleteTime > 200.0))
        {
            m_stateCommandGiven = false;
            return true;
        }
    }
    return false;
}

bool NAOKick::kickAbortCondition()
{
    return false;
}

float NAOKick::CalculateForwardSwingSpeed(float kickDistance)
{
    return 5.0f;
}

float NAOKick::CalculateSidewardSwingSpeed(float kickDistance)
{
    return 4.0f;
}

bool NAOKick::chooseLeg()
{
    //currently must be at zero position
	double theta = 0;//atan2(m_target_y-m_ball_y, m_target_x-m_ball_x);
	
	//approximate, assume robot torso moves negligible amount and only rotates
	double xtrans = m_ball_x*cos(theta) + m_ball_y*sin(theta);
	double ytrans = m_ball_y*cos(theta) - m_ball_x*sin(theta);

        #if DEBUG_NUMOTION_VERBOSITY > 1
        debug << "bool NAOKick::chooseLeg()" << endl;
        debug << "theta - " << theta << endl;
        debug << "xtrans - " << xtrans << endl;
        debug << "ytrans - " << ytrans << endl;
        #endif

        Vector2<float> ballLocation(m_ball_x, m_ball_y), leftFootRelativeBallLocation, rightFootRelativeBallLocation;

        Matrix leftFootTransform, rightFootTransform;
        vector<float> lftransform, rftransform;
        m_data->get(NUSensorsData::LLegTransform, lftransform);
    	leftFootTransform = Matrix4x4fromVector(lftransform);
    	m_data->get(NUSensorsData::RLegTransform, rftransform);
        rightFootTransform = Matrix4x4fromVector(rftransform);
        leftFootRelativeBallLocation = m_kinematicModel->TransformPositionToFoot(leftFootTransform, ballLocation);
        rightFootRelativeBallLocation = m_kinematicModel->TransformPositionToFoot(rightFootTransform, ballLocation);

        vector<float> leftPos = Kinematics::PositionFromTransform(leftFootTransform);
        vector<float> rightPos = Kinematics::PositionFromTransform(rightFootTransform);

        #if DEBUG_NUMOTION_VERBOSITY > 1
            debug << "Right Pos = (" << rightPos[0] << "," << rightPos[1] << "," << rightPos[2] << ")" << endl;
            debug << "Left Pos = (" << leftPos[0] << "," << leftPos[1] << "," << leftPos[2] << ")" << endl;

            debug << "Ball Position - " << endl;
            debug << "Origin Relative: (" << ballLocation.x << "," << ballLocation.y << ")" << endl;
            debug << "Right Foot Relative: (" << rightFootRelativeBallLocation.x << "," << rightFootRelativeBallLocation.y << ")" << endl;
            debug << "Left Foot Relative: (" << leftFootRelativeBallLocation.x << "," << leftFootRelativeBallLocation.y << ")" << endl;
        #endif

        const float fwdAngleRange = PI/4.0f;
        const float sideAngleRange = PI/8.0f;
        // Direction is forwardsish

        bool kickSelected = false;

        if(fabs(theta) < fwdAngleRange)
        {
            #if DEBUG_NUMOTION_VERBOSITY > 1
            debug << "Right foot: " << endl << RightFootForwardKickableArea.MinX() << " < " << leftFootRelativeBallLocation.x << " < " << RightFootForwardKickableArea.MaxX() << endl;
            debug << RightFootForwardKickableArea.MinY() << " < " << leftFootRelativeBallLocation.y << " < " << RightFootForwardKickableArea.MaxY() << endl;
            debug << "Left foot: " << endl << LeftFootForwardKickableArea.MinX() << " < " << rightFootRelativeBallLocation.x << " < " << LeftFootForwardKickableArea.MaxX() << endl;
            debug << LeftFootForwardKickableArea.MinY() << " < " << rightFootRelativeBallLocation.y << " < " << LeftFootForwardKickableArea.MaxY() << endl;
            #endif
            if(RightFootForwardKickableArea.PointInside(leftFootRelativeBallLocation.x,leftFootRelativeBallLocation.y))
            {
                m_kicking_leg = rightLeg;
                m_swingDirection = ForwardSwing;
                pose = PRE_KICK;
                kickSelected = true;
            }
            else if(LeftFootForwardKickableArea.PointInside(rightFootRelativeBallLocation.x,rightFootRelativeBallLocation.y))
            {
                m_kicking_leg = leftLeg;
                m_swingDirection = ForwardSwing;
                pose = PRE_KICK;
                kickSelected = true;
            }
        }
        // Direction is to the leftish
        else if(fabs(theta-PI/2.0f) < sideAngleRange)
        {
            if(RightFootLeftKickableArea.PointInside(m_ball_x,m_ball_y))
            {
                m_kicking_leg = rightLeg;
                m_swingDirection = LeftSwing;
                pose = PRE_KICK;
                kickSelected = true;
            }
            else if(LeftFootLeftKickableArea.PointInside(m_ball_x,m_ball_y))
            {
                m_kicking_leg = leftLeg;
                m_swingDirection = LeftSwing;
                pose = PRE_KICK;
                kickSelected = true;
            }
        }
        // Direction is rightish
        else if(fabs(theta+PI/2.0f) < sideAngleRange)
        {
            if(LeftFootRightKickableArea.PointInside(m_ball_x,m_ball_y))
            {
                m_kicking_leg = leftLeg;
                m_swingDirection = RightSwing;
                pose = PRE_KICK;
                kickSelected = true;
            }
            if(RightFootRightKickableArea.PointInside(m_ball_x,m_ball_y))
            {
                m_kicking_leg = rightLeg;
                m_swingDirection = RightSwing;
                pose = PRE_KICK;
                kickSelected = true;
            }
        }
    
        if(!kickSelected)
            stop();
        else
        {
            m_kick_enabled = true; // Set the kick to active now, we are doing stuff!
            m_kickWait = true;
            #if DEBUG_NUMOTION_VERBOSITY > 3
                debug << "Kick Selected - Kicking Foot: " << NUKick::toString(m_kicking_leg) << ", Swing Direction: " << toString(m_swingDirection) << endl;
            #endif
        }
        return kickSelected;
}

double NAOKick::MoveLimbToPositionWithSpeed(NUActionatorsData::id_t limbId, vector<float> currentPosition, vector<float> targetPosition, float maxSpeed , float gain, float smoothness)
{
    #if DEBUG_NUMOTION_VERBOSITY > 3
        debug << "NAOKick::MoveLimbToPositionWithSpeed(" << limbId.Name << "," << currentPosition << "," << targetPosition << "," << maxSpeed << "," << gain << "," << smoothness << endl;
    #endif
    
    const float movespeed = maxSpeed;
    vector< vector<float> > positions;
    vector< vector<float> > velocities;
    vector< vector<double> > times;

    // compute the time required to move into the initial pose for each limb
    double moveTime = 1000*(maxDifference(currentPosition, targetPosition)/movespeed);
    float startTime = m_data->CurrentTime;
    float endTime = startTime + moveTime;
    
    #if DEBUG_NUMOTION_VERBOSITY > 3
        debug << "NAOKick::MoveLimbToPositionWithSpeed: moveTime: " << moveTime << " startTime: " << startTime << " endTime: " << endTime << endl;
    #endif

    vector<double> endTimes(1,endTime);
    vector< vector<float> > endPositions(1,targetPosition);

    MotionCurves::calculate(startTime,endTimes,currentPosition,endPositions,smoothness,10,times,positions,velocities);

    m_actions->add(limbId,times,positions,gain);
    return endTime;
}

double NAOKick::MoveLegsToPositionWithSpeed(const vector<float>& targetPosition, float maxSpeed , float gain, float smoothness)
{
    vector<float> left_currentPosition, right_currentPosition;
    m_data->getPosition(NUSensorsData::LLeg, left_currentPosition);
    m_data->getPosition(NUSensorsData::RLeg, right_currentPosition);
    
    const float movespeed = maxSpeed;
    vector< vector<float> > positions;
    vector< vector<float> > velocities;
    vector< vector<double> > times;
    
    // compute the time required to move into the initial pose for each limb
    double left_moveTime = 1000*(maxDifference(left_currentPosition, targetPosition)/movespeed);
    double right_moveTime = 1000*(maxDifference(right_currentPosition, targetPosition)/movespeed);
    double moveTime = max(left_moveTime, right_moveTime);
    
    float startTime = m_data->CurrentTime;
    float endTime = startTime + moveTime;
    
    vector<double> endTimes(1,endTime);
    vector< vector<float> > endPositions(1, targetPosition);
    
    MotionCurves::calculate(startTime,endTimes,left_currentPosition,endPositions,smoothness,10,times,positions,velocities);
    m_actions->add(NUActionatorsData::LLeg, times, positions, gain);
    
    MotionCurves::calculate(startTime,endTimes,right_currentPosition,endPositions,smoothness,10,times,positions,velocities);
    m_actions->add(NUActionatorsData::RLeg, times, positions, gain);
    
    return endTime;
}

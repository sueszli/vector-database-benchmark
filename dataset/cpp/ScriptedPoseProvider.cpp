/*! @file ScriptedPoseProvider.cpp
    @brief Implementation of Pose behaviour class

    @author Josh Wilson
 
 Copyright (c) 2010 Josh Wilson
 
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

#include "ScriptedPoseProvider.h"

#include "Infrastructure/Jobs/JobList.h"
#include "Infrastructure/NUActionatorsData/NUActionatorsData.h"

#include "Infrastructure/Jobs/MotionJobs/HeadJob.h"
#include "Infrastructure/Jobs/MotionJobs/WalkJob.h"
#include "Infrastructure/Jobs/MotionJobs/MotionFreezeJob.h"
#include "Infrastructure/Jobs/VisionJobs/SaveImagesJob.h"


#include <math.h>
#include "debug.h"
#include "debugverbositybehaviour.h"

using namespace std;

ScriptedPoseProvider::ScriptedPoseProvider(Behaviour* manager) : BehaviourProvider(manager)
{
    m_pitch_index = 0;
    m_yaw_index = 25;
    m_num_pitch_motions = 15;
    m_num_yaw_motions = 49;
    isStart = 0;
    m_saving_images = false;

    m_script = MotionScript("Test");
    m_script_playing = false;

}


ScriptedPoseProvider::~ScriptedPoseProvider()
{

}

void ScriptedPoseProvider::doBehaviour()
{
    //Saving images
    if(singleChestClick())
    {
	m_saving_images = true;
        m_jobs->addVisionJob(new SaveImagesJob(m_saving_images, true));
	

    }
    else if(m_saving_images == true)
    {
	m_saving_images = false;
        m_jobs->addVisionJob(new SaveImagesJob(m_saving_images, true));
    }
    

    doSelectedMotion();
    
}

void ScriptedPoseProvider::doSelectedMotion()
{
    //Initialisation: First 50 Frames, will be used to stand up
    if (isStart < 10)
    {
        vector<float> zero(m_actions->getSize(NUActionatorsData::Head), 0);
        m_actions->add(NUActionatorsData::Head, m_current_time, zero, 50);
        m_jobs->addMotionJob(new WalkJob(0.001,0.001,0.001));
		isStart++;
    }
    else if (isStart < 50)
    {
        m_jobs->addMotionJob(new MotionFreezeJob());
    	isStart++;
    }
    //Start the behaviour:
    else
    {
    	if(!m_script_playing)
    	{
    		m_script_playing = true;
    		m_script.play(m_data, m_actions);
    	}

        m_jobs->addMotionJob(new WalkJob(0,0,0));
    }
}

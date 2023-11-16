/*! @file FinishedState.cpp
    @brief Implementation of the finished soccer state

    @author Jason Kulk
 
  Copyright (c) 2010 Jason Kulk
 
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

#include "FinishedState.h"
#include "../SoccerProvider.h"

#include "Infrastructure/Jobs/JobList.h"
#include "Infrastructure/NUActionatorsData/NUActionatorsData.h"
#include "NUPlatform/NUActionators/NUSounds.h"

#include "Infrastructure/Jobs/MotionJobs/MotionKillJob.h"
#include "Infrastructure/Jobs/MotionJobs/HeadJob.h"

FinishedState::FinishedState(SoccerProvider* provider) : SoccerState(provider)
{
}

FinishedState::~FinishedState()
{
}

BehaviourState* FinishedState::nextState()
{
    return this;
}

void FinishedState::doState()
{
    if (m_provider->stateChanged())
    {   // play a sound, and stop moving
        m_actions->add(NUActionatorsData::Sound, m_actions->CurrentTime, NUSounds::FINISHED);
        m_jobs->addMotionJob(new MotionKillJob());
    }
    // In finished the chest led should be off
    m_actions->add(NUActionatorsData::ChestLed, m_actions->CurrentTime, vector<float>(3,0));
}


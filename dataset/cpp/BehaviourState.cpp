/*! @file BehaviourState.cpp
    @brief Implementation of behaviour state class

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

#include "BehaviourState.h"
#include "BehaviourProvider.h"

#include "debug.h"
#include "debugverbositybehaviour.h"

using namespace std;

BehaviourState* BehaviourState::getNextState()
{
    if (m_processed)
        return nextState();
    else
        return this;
}

void BehaviourState::process(JobList* jobs, NUSensorsData* data, NUActionatorsData* actions, FieldObjects* fieldobjects, GameInformation* gameinfo, TeamInformation* teaminfo)
{
    m_data = data;
    m_actions = actions;
    m_jobs = jobs;
    m_field_objects = fieldobjects;
    m_game_info = gameinfo;
    m_team_info = teaminfo;
    m_processed = true;
    doState();
}

/*! @brief Destroys the behaviour state
 */
BehaviourState::~BehaviourState()
{
}



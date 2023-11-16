/*! @file WatchDogThread.h
    @brief Declaration of the sense->move thread class.

    @author Jason Kulk
 
 Copyright (c) 2010 Jason Kulk
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "WatchDogThread.h"
#include "NUbot.h"

#include "Infrastructure/NUBlackboard.h"
#include "Infrastructure/NUSensorsData/NUSensorsData.h"
#include "Infrastructure/NUActionatorsData/NUActionatorsData.h"
#include "NUPlatform/NUPlatform.h"
#include "Infrastructure/GameInformation/GameInformation.h"

#ifdef USE_VISION
    #include "Vision/Vision.h"
#endif

#include "debug.h"
#include "debugverbositynubot.h"
#include "debugverbositythreading.h"
#include "nubotconfig.h"

#include <errno.h>

#if DEBUG_NUBOT_VERBOSITY > DEBUG_THREADING_VERBOSITY
    #define DEBUG_VERBOSITY DEBUG_NUBOT_VERBOSITY
#else
    #define DEBUG_VERBOSITY DEBUG_THREADING_VERBOSITY
#endif

/*! @brief Constructs the sense->move thread
 */

WatchDogThread::WatchDogThread(NUbot* nubot) : PeriodicThread(string("WatchDogThread"), 1000, 0)
{
    #if DEBUG_VERBOSITY > 0
        debug << "WatchDogThread::WatchDogThread(" << nubot << ") with priority " << static_cast<int>(m_priority) << endl;
    #endif
    m_nubot = nubot;
}

WatchDogThread::~WatchDogThread()
{
    #if DEBUG_VERBOSITY > 0
        debug << "WatchDogThread::~WatchDogThread()" << endl;
    #endif
    stop();
}

void WatchDogThread::periodicFunction()
{
	Blackboard->GameInfo->sendAlivePacket();
    bool ok = Platform->displayBatteryState();
    if (Blackboard->Sensors->CurrentTime > 20000)
    {
        ok &= Platform->verifySensors();

        #ifdef USE_VISION
            ok &= Platform->verifyVision(1000.0*m_nubot->m_vision->getNumFramesDropped()/m_period, 1000.0*m_nubot->m_vision->getNumFramesProcessed()/m_period);
        #endif
    }
    
    if (not ok)
        Blackboard->GameInfo->requestForPickup();
}

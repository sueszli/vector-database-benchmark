/*! @file NUSoundThread.h
    @brief Implementation of the saveimages thread class.

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

#include "NUSoundThread.h"

#include "debug.h"
#include "debugverbositynuactionators.h"
#include "nubotdataconfig.h"
#include "targetconfig.h"
#include <stdlib.h>
#include <errno.h>



/*! @brief Constructs the sound thread
 */

NUSoundThread::NUSoundThread() : QueueThread<std::string>(std::string("NUSoundThread"), 0)
{
    #if DEBUG_NUACTIONATORS_VERBOSITY > 0
        debug << "NUSoundThread::NUSoundThread() with priority " << static_cast<int>(m_priority) << std::endl;
    #endif
    #if defined(TARGET_OS_IS_WINDOWS)
        m_player_command = std::string("start/min sndrec32 /play /close ");
    #elif defined(TARGET_OS_IS_DARWIN)
        m_player_command = std::string("afplay ");
    #else
        m_player_command = std::string("aplay ");
    #endif
    m_sound_dir = std::string(DATA_DIR) + std::string("Sounds/");
    start();
}

/*! @brief Destroys the sound thread
 */
NUSoundThread::~NUSoundThread()
{
    #if DEBUG_NUACTIONATORS_VERBOSITY > 0
        debug << "NUSoundThread::~NUSoundThread()" << std::endl;
    #endif
}

/*! @brief The sound threads main loop
 */
void NUSoundThread::run()
{
    #if DEBUG_NUACTIONATORS_VERBOSITY > 0
        debug << "NUSoundThread::run()" << std::endl;
    #endif
    
    int err = 0;
    while (err == 0 && errno != EINTR)
    {
        waitForCondition();
        // ------------------------------------------------------------------------------------------------------------------------------------------
        debug << "NUSoundThread Processing: " << m_player_command + m_sound_dir + m_queue.front() << std::endl;
        err = system((m_player_command + m_sound_dir + m_queue.front()).c_str());
        m_queue.pop_front();
        // ------------------------------------------------------------------------------------------------------------------------------------------
    } 
    errorlog << "NUSoundThread is exiting. err: " << err << " errno: " << errno << std::endl;
}

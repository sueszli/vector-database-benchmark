/*
 * Copyright (C) 2015  Maxim Noah Khailo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * In addition, as a special exception, the copyright holders give 
 * permission to link the code of portions of this program with the 
 * Botan library under certain conditions as described in each 
 * individual source file, and distribute linked combinations 
 * including the two.
 *
 * You must obey the GNU General Public License in all respects for 
 * all of the code used other than Botan. If you modify file(s) with 
 * this exception, you may extend this exception to your version of the 
 * file(s), but you are not obligated to do so. If you do not wish to do 
 * so, delete this exception statement from your version. If you delete 
 * this exception statement from all source files in the program, then 
 * also delete it here.
 */

#include "gui/app/app_reaper.hpp"
#include "util/log.hpp"
#include "util/dbc.hpp"

namespace fire
{
    namespace gui
    {
        namespace app
        {
            void reap_thread(app_reaper* r)
            try
            {
                REQUIRE(r);

                while(!r->_closed.is_done())
                try
                {
                    closed_app app;
                    if(!r->_closed.pop(app, true))
                        continue;

                    LOG << "reaping app " << app.name << "(" << app.id << ")" << std::endl;
                    app.back->stop();

                    r->emit_cleanup(app);

                    LOG << "reaped app " << app.name << "(" << app.id << ")" << std::endl;
                }
                catch(std::exception& e)
                {
                    LOG << "Error reaping app: " << e.what() << std::endl;
                }
                catch(...)
                {
                    LOG << "Unknown error reaping app" << std::endl;
                }
            }
            catch(...)
            {
                LOG << "exit: app_reaper::reap_threaad" << std::endl;
            }

            app_reaper::app_reaper(QWidget* parent)
            {
                REQUIRE(parent);
                qRegisterMetaType<closed_app>("closed_app");
                connect(this, SIGNAL(got_cleanup(closed_app)), this, SLOT(do_cleanup(closed_app)));

                _hidden = new QWidget{parent};
                _hidden->setHidden(true);
                _thread.reset(new std::thread{reap_thread, this});

                ENSURE(_hidden);
            }

            app_reaper::~app_reaper()
            {
                REQUIRE(_hidden);
                if(!_closed.is_done()) stop();
            }

            void app_reaper::reap(closed_app& app)
            {
                if(_closed.is_done()) return;

                //stop frontend
                app.front->stop();

                //migrate widgets to hidden widgt
                app.hidden = new QWidget{_hidden};
                app.front->set_parent(app.hidden);

                //push to closed queue to be reaped
                _closed.push(app);

                ENSURE(app.hidden);
            }

            void app_reaper::emit_cleanup(closed_app c)
            {
                emit got_cleanup(c);
            }

            void app_reaper::do_cleanup(closed_app app)
            {
                //cleanup widgets
                REQUIRE(app.hidden);
                delete app.hidden;
            }

            void app_reaper::stop()
            {
                LOG << "stopping reaper" << std::endl;
                _closed.done();
                _thread->join();
                LOG << "stopped reaper" << std::endl;
            }
        }
    }
}

/*
 * Copyright (C) 2014  Maxim Noah Khailo
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

#include <QtWidgets>

#include "gui/app/script_app.hpp"
#include "gui/util.hpp"
#include "util/uuid.hpp"
#include "util/dbc.hpp"
#include "util/log.hpp"

#include <QTimer>

#include <functional>

namespace m = fire::message;
namespace ms = fire::messages;
namespace us = fire::user;
namespace s = fire::conversation;
namespace u = fire::util;
namespace l = fire::gui::lua;

namespace fire
{
    namespace gui
    {
        namespace app
        {
            const std::string SCRIPT_APP = "SCRIPT_APP";

            script_app::script_app(
                    app_ptr app, 
                    app_service_ptr as,
                    app_reaper_ptr ar,
                    s::conversation_service_ptr conversation_s,
                    s::conversation_ptr conversation) :
                generic_app{},
                _from_id(conversation->user_service()->user().info().id()),
                _id(u::uuid()),
                _conversation_service{conversation_s},
                _conversation{conversation},
                _app{app},
                _app_service{as},
                _app_reaper{ar}
            {
                REQUIRE(conversation_s);
                REQUIRE(conversation);
                REQUIRE(app);
                REQUIRE(as);
                REQUIRE(ar);

                init();

                INVARIANT(_conversation_service);
                INVARIANT(_conversation);
                INVARIANT(_app);
                INVARIANT(_app_service);
                INVARIANT(_app_reaper);
                INVARIANT_FALSE(_id.empty());
            }

            script_app::script_app(
                    const std::string& from_id, 
                    const std::string& id, 
                    app_ptr app, 
                    app_service_ptr as, 
                    app_reaper_ptr ar, 
                    s::conversation_service_ptr conversation_s,
                    s::conversation_ptr conversation) :
                generic_app{},
                _from_id(from_id),
                _id(id),
                _conversation_service{conversation_s},
                _conversation{conversation},
                _app{app},
                _app_service{as},
                _app_reaper{ar}
            {
                REQUIRE(conversation_s);
                REQUIRE(conversation);
                REQUIRE(app);
                REQUIRE(as);
                REQUIRE(ar);
                REQUIRE_FALSE(id.empty());

                init();

                INVARIANT(_conversation_service);
                INVARIANT(_conversation);
                INVARIANT(_app);
                INVARIANT(_app_service);
                INVARIANT(_app_reaper);
                INVARIANT_FALSE(_id.empty());
            }

            script_app::~script_app()
            {
                INVARIANT(_app);
                INVARIANT(_back);
                INVARIANT(_app_reaper);
                LOG << "closing app " << _app->name() << "(" << _app->id() << ")" << std::endl;
                closed_app c;
                c.name = _app->name();
                c.id = _app->id();
                c.front = _front;
                c.back = _back;
                _app_reaper->reap(c);
            }

            void script_app::setup_decorations()
            {
                INVARIANT(_app);
                INVARIANT(_app_service);
                REQUIRE_FALSE(_clone);

                set_title(_app->name().c_str());

                //setup error widget
                _err = new QLabel;
                _err->setVisible(false);
                _err->setToolTip(tr("The App Encountered an Error"));
                make_error(*_err);
                layout()->addWidget(_err, 0,0);

                //setup mic
                _mic = new QPushButton;
                make_mic(*_mic);
                _mic->setVisible(false);
                connect(_mic, SIGNAL(clicked()), this, SLOT(toggle_mic()));
                layout()->addWidget(_mic, 0,1);

                CHECK_FALSE(_resource.mic);
                update_mic_widget();


                //setup clone
                _clone = new QPushButton;
                make_install(*_clone);

                //color the install icon depending on whether the app is already installed
                bool exists = _app_service->available_apps().count(_app->id());
                if(exists) 
                {
                    _clone->setToolTip(tr("update app"));
                    _clone->setStyleSheet("border: 0px; color: 'grey';");
                }
                else
                {
                    _clone->setToolTip(tr("install app"));
                }

                connect(_clone, SIGNAL(clicked()), this, SLOT(clone_app()));

                layout()->addWidget(_clone, 0,2);

                ENSURE(_clone);
                ENSURE(_mic);
            }

            void script_app::update_mic_widget()
            {
                INVARIANT(_mic);
                if(_resource.mic)
                {
                    make_green(*_mic);
                    _mic->setToolTip(tr("disable microphone"));
                }
                else
                {
                    make_red(*_mic);
                    _mic->setToolTip(tr("enable microphone"));
                }
            }

            void script_app::toggle_mic()
            {
                INVARIANT(_front);

                _resource.mic = !_resource.mic;
                update_mic_widget();

                if(_resource.mic) _front->mic_enable();
                else _front->mic_disable();
            }

            void script_app::init()
            {
                INVARIANT(root());
                INVARIANT(layout());
                INVARIANT(_conversation);
                INVARIANT(_app);

                //setup frontend
                setup_decorations();

                _canvas = new QWidget;
                _canvas_layout = new QGridLayout{_canvas};
                layout()->addWidget(_canvas, 1,0,2,3);
                set_main(_canvas);

                //setup mail
                _mail = std::make_shared<m::mailbox>(_id);
                _sender = std::make_shared<ms::sender>(_conversation->user_service(), _mail);

                ENSURE(_mail);
                ENSURE(_sender);
            }

            void script_app::start()
            {
                INVARIANT(root());
                INVARIANT(layout());
                INVARIANT(_conversation);
                INVARIANT(_app);

                //setup frontend
                auto front = std::make_shared<qtw::qt_frontend>(_canvas, _canvas_layout, nullptr);
                connect(front.get(), SIGNAL(alerted()), this, SLOT(got_alert()));
                connect(front.get(), SIGNAL(do_adjust_size()), this, SLOT(got_adjust_size()));
                connect(front.get(), SIGNAL(mic_added()), this, SLOT(got_mic_added()));

                _front = std::make_shared<qtw::qt_frontend_client>(front);
                connect(_front.get(), SIGNAL(got_report_error(const std::string&)), this, SLOT(got_error(const std::string&)));

                //setup api and backend
                _api = std::make_shared<l::lua_api>(
                        _app, 
                        _sender, 
                        _conversation, 
                        _conversation_service, 
                        _front.get());
                _api->who_started_id = _from_id;

                _back = std::make_shared<l::backend_client>(_api, _mail); 

                //assign backend to frontend
                _front->set_backend(_back.get());

                //run script and start backend on seperate thread
                _back->run(_app->code());
                _back->start();

                //setup mail service
                INVARIANT(_api);
                INVARIANT(_front);
                INVARIANT(_back);
                INVARIANT(_conversation);
            }

            void script_app::contact_quit(const std::string& id)
            {
                REQUIRE_FALSE(id.empty());
                INVARIANT(_back);

                _back->contact_quit(id);
            }

            const std::string& script_app::id() const
            {
                ENSURE_FALSE(_id.empty());
                return _id;
            }

            const std::string& script_app::type() const
            {
                ENSURE_FALSE(SCRIPT_APP.empty());
                return SCRIPT_APP;
            }

            m::mailbox_ptr script_app::mail()
            {
                ENSURE(_mail);
                return _mail;
            }

            void script_app::clone_app()
            {
                INVARIANT(_app_service);
                INVARIANT(_app);

                install_app_gui(*_app, *_app_service, this);
            }

            void script_app::got_alert()
            {
                alerted();
            }

            void script_app::got_adjust_size()
            {
                adjust_size();
            }

            void script_app::got_mic_added()
            {
                INVARIANT(_mic);
                _mic->setVisible(true);
            }

            void script_app::got_error(const std::string&)
            {
                INVARIANT(_err);
                _err->setVisible(true);
            }
        }
    }
}


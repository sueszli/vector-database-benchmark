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

#include "message/post_office.hpp"
#include "util/dbc.hpp"
#include "util/log.hpp"
#include "util/string.hpp"

namespace u = fire::util;

namespace fire
{
    namespace message
    {
        namespace
        {
            const double MIN_THREAD_SLEEP = 1; //in milliseconds 
            const double SLEEP_STEP = 5; //in milliseconds 
            const double MAX_THREAD_SLEEP = 51; //in milliseconds 
        }

        void send_thread(post_office* o)
        try
        {
            REQUIRE(o);

            double thread_sleep = MIN_THREAD_SLEEP;
            while(!o->_done)
            try
            {
                bool sent = false;

                auto boxes = o->boxes();
                for(auto p : boxes)
                {
                    auto wp = p.second;
                    auto sp = wp.lock();
                    if(!sp) continue;

                    message m;
                    if(!sp->pop_outbox(m)) continue;

                    m.meta.from.push_front(sp->address());

                    CHECK_EQUAL(m.meta.from.size(), 1);

                    o->send(m);
                    sent = true;
                }

                if(!sent) 
                {
                    util::sleep_thread(thread_sleep);
                    thread_sleep = std::min(MAX_THREAD_SLEEP, thread_sleep + SLEEP_STEP);
                }
                else thread_sleep = MIN_THREAD_SLEEP;
            }
            catch(std::exception& e)
            {
                INVARIANT(o);
                LOG << "Error sending message in post_office `" << o->address() << "'. " << e.what() << std::endl; 
            }
            catch(...)
            {
                LOG << "Unexpected error sending message in post_office `" << o->address() << "'." << std::endl; 
            }
        }
        catch(...)
        {
            LOG << "exit: postoffice::send_thread" << std::endl;
        }
         
        post_office::post_office() :
                _address{},
                _boxes{},
                _offices{},
                _parent{},
                _done{false}
        {
            _send_thread.reset(new std::thread{send_thread, this});

            INVARIANT(_send_thread);
        }

        post_office::post_office(const std::string& a) : 
                _address(a),
                _boxes{},
                _offices{},
                _parent{},
                _done{false}
        {
            _send_thread.reset(new std::thread{send_thread, this});

            INVARIANT(_send_thread);
        }

        post_office::~post_office()
        {
            INVARIANT(_send_thread);
            _done = true;
            _send_thread->join();
        }

        const std::string& post_office::address() const
        {
            return _address;
        }

        void post_office::address(const std::string& a)
        {
            _address = a;
        }

        bool post_office::send(message m)
        {
            metadata& meta = m.meta;
            if(meta.to.empty()) return false;

            if(meta.to.size() > 1 && meta.to.front() == _address)
                meta.to.pop_front();

            //route to child post
            if(meta.to.size() > 1)
            {
                {std::lock_guard<std::mutex> lock(_post_m);
                    const auto& to = meta.to.front();
                    auto p = _offices.find(to);

                    if(p != _offices.end())
                    {
                        auto wp = p->second;
                        if(auto sp = wp.lock())
                        {
                            message cp = m;
                            cp.meta.to.pop_front();
                            cp.meta.from.push_front(_address);

                            if(sp->send(cp)) return true;
                        }
                    }
                }

                //the post office address is added here
                //to the from so that the receive can send message
                //back to sender
                m.meta.from.push_front(_address);

                //send to parent.
                //otherwise, try to send message to outside world
                return _parent ? _parent->send(m) : send_outside(m);
            }

            //route to mailbox
            CHECK_EQUAL(meta.to.size(), 1);

            {std::lock_guard<std::mutex> lock(_box_m);
                const auto& to = meta.to.front();
                auto p = _boxes.find(to);

                if(p != _boxes.end())
                {
                    auto wb = p->second;
                    if(auto sb = wb.lock())
                    {
                        sb->push_inbox(m);
                        return true;
                    }
                }
            }

            //could not send message
            return false;
        }

        void post_office::clean_mailboxes()
        {
            u::string_set to_be_deleted;
            for(auto p : _boxes)
            {
                if(p.second.lock()) continue;
                to_be_deleted.insert(to_be_deleted.end(), p.first);
            }

            for(const auto& address : to_be_deleted)
            {
                LOG << "removing mailbox: " << address << std::endl;
                _boxes.erase(address);
            }
        }

        bool post_office::add(mailbox_wptr p)
        {
            std::lock_guard<std::mutex> lock(_box_m);

            auto sp = p.lock();
            if(!sp) return false;

            REQUIRE_FALSE(sp->address().empty());

            clean_mailboxes();
            _boxes[sp->address()] = p;

            return true;
        }

        bool post_office::has(mailbox_wptr p) const
        {
            std::lock_guard<std::mutex> lock(_box_m);

            auto sp = p.lock();
            if(!sp) return false;

            return _boxes.count(sp->address()) > 0;
        }

        void post_office::remove_mailbox(const std::string& n)
        {
            std::lock_guard<std::mutex> lock(_box_m);
            _boxes.erase(n);
        }

        mailboxes post_office::boxes() const
        {
            std::lock_guard<std::mutex> lock(_box_m);
            return _boxes;
        }

        bool post_office::add(post_office_wptr p)
        {
            std::lock_guard<std::mutex> lock(_post_m);

            auto sp = p.lock();
            if(!sp) return false;

            REQUIRE_NOT_EQUAL(sp.get(), this);
            REQUIRE_FALSE(sp->address().empty());
            REQUIRE_FALSE(_offices.count(sp->address()));

            sp->parent(this);
            _offices[sp->address()] = p;

            return true;
        }

        bool post_office::has(post_office_wptr p) const
        {
            std::lock_guard<std::mutex> lock(_post_m);

            auto sp = p.lock();
            if(!sp) return false;

            return _offices.count(sp->address()) > 0;
        }

        void post_office::remove_post_office(const std::string& n)
        {
            std::lock_guard<std::mutex> lock(_post_m);
            _offices.erase(n);
        }

        const post_offices& post_office::offices() const
        {
            std::lock_guard<std::mutex> lock(_box_m);
            return _offices;
        }

        post_office* post_office::parent() 
        { 
            return _parent; 
        }

        const post_office* post_office::parent() const 
        { 
            return _parent; 
        }

        void post_office::parent(post_office* p) 
        { 
            REQUIRE(p);
            REQUIRE_NOT_EQUAL(p, this);
            _parent = p;
        }
        
        bool post_office::send_outside(const message&)
        {
            //subclasses need to implement this
            return false;
        }

        const mailbox_stats& post_office::outside_stats() const
        {
            return _outside_stats;
        }

        mailbox_stats& post_office::outside_stats() 
        {
            return _outside_stats;
        }

        void post_office::outside_stats(bool on)
        {
            _outside_stats.on = on;
        }
    }
}

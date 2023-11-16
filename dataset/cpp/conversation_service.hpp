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

#ifndef FIRESTR_CONVERSATION_CONVERSATION_SERVICE_H
#define FIRESTR_CONVERSATION_CONVERSATION_SERVICE_H

#include "conversation/conversation.hpp"
#include "service/service.hpp"
#include "message/post_office.hpp"
#include "message/mailbox.hpp"
#include "messages/sender.hpp"

#include <map>
#include <mutex>

namespace fire
{
    namespace conversation
    {
        using conversation_map = std::map<std::string, conversation_ptr>;
        using app_addresses = std::set<std::string>;

        class conversation_service : public service::service 
        {
            public:
                conversation_service(
                        message::post_office_ptr,
                        user::user_service_ptr,
                        message::mailbox_ptr event = nullptr);
                ~conversation_service();

            public:
                conversation_ptr sync_conversation(
                        const std::string& from_id, //may be empty 
                        const std::string& id, 
                        const user::contact_list&, 
                        const app_addresses&);

                conversation_ptr create_conversation(const std::string& id);
                conversation_ptr create_conversation(user::contact_list&);
                conversation_ptr create_conversation();
                void quit_conversation(const std::string& id);

            public:
                conversation_ptr conversation_by_id(const std::string&);

                void add_contact_to_conversation( 
                        const user::user_info_ptr contact, 
                        conversation_ptr conversation);

                void sync_existing_conversation(conversation_ptr conversation);
                void sync_existing_conversation(const std::string& conversation_id);
                void broadcast_message(const message::message&);

            public:
                user::user_service_ptr user_service();

            public:
                void fire_conversation_alert(const std::string& id, bool visible);

            protected:
                void received_sync(const message::message&);
                void received_quit(const message::message&);
                void received_ask_contact_req(const message::message&);
                void received_ask_contact_res(const message::message&);

                void ask_about(
                        const std::string& id,
                        conversation_ptr conversation);

                void add_contact_to_conversation_p( 
                        const user::user_info_ptr contact, 
                        conversation_ptr conversation);

            private:
                void init_handlers();
                void fire_new_conversation_event(const std::string& id);
                void fire_quit_conversation_event(const std::string& id);
                void fire_conversation_synced_event(const std::string& id);

                void fire_contact_removed(
                        const std::string& conversation_id,
                        const std::string& contact_id);

                void fire_contact_added(
                        const std::string& conversation_id,
                        const std::string& contact_id);

                void fire_not_part_of_clique(
                        const std::string& conversation_id,
                        const std::string& contact_id,
                        const contact_id_set&);

                void request_apps(
                        const std::string& from_id,
                        conversation_ptr s, 
                        const app_addresses&);

            private:
                message::post_office_ptr _post;
                user::user_service_ptr _user_service;
                messages::sender_ptr _sender;
                conversation_map _conversations;
                std::mutex _mutex;
        };

        using conversation_service_ptr = std::shared_ptr<conversation_service>;
        using conversation_servie_wptr = std::weak_ptr<conversation_service>;

        //events
        namespace event
        {
            extern const std::string NEW_CONVERSATION;
            f_message(new_conversation)
            {
                std::string conversation_id;

                f_message_init(new_conversation, NEW_CONVERSATION);
                f_serialize
                {
                    f_s(conversation_id);
                }
            };

            extern const std::string QUIT_CONVERSATION;
            f_message(quit_conversation)
            {
                std::string conversation_id;

                f_message_init(quit_conversation, QUIT_CONVERSATION);
                f_serialize
                {
                    f_s(conversation_id);
                }
            };

            extern const std::string CONVERSATION_SYNCED;
            f_message(conversation_synced)
            {
                std::string conversation_id;

                f_message_init(conversation_synced, CONVERSATION_SYNCED);
                f_serialize
                {
                    f_s(conversation_id);
                }
            };

            extern const std::string CONTACT_REMOVED;
            f_message(contact_removed)
            {
                std::string conversation_id;
                std::string contact_id;

                f_message_init(contact_removed, CONTACT_REMOVED);
                f_serialize
                {
                    f_s(conversation_id);
                    f_s(contact_id);
                }
            };

            extern const std::string CONTACT_ADDED;
            f_message(contact_added)
            {
                std::string conversation_id;
                std::string contact_id;

                f_message_init(contact_added, CONTACT_ADDED);
                f_serialize
                {
                    f_s(conversation_id);
                    f_s(contact_id);
                }
            };

            extern const std::string NOT_PART_OF_CLIQUE;
            f_message(not_part_of_clique)
            {
                std::string conversation_id;
                std::string contact_id;
                contact_id_set dont_know;

                f_message_init(not_part_of_clique, NOT_PART_OF_CLIQUE);
                f_serialize
                {
                    f_s(conversation_id);
                    f_s(contact_id);
                    f_s(dont_know);
                }
            };

            extern const std::string CONVERSATION_ALERT;
            f_message(conversation_alert)
            {
                std::string conversation_id;
                bool visible;

                f_message_init(conversation_alert, CONVERSATION_ALERT);
                f_serialize
                {
                    f_s(conversation_id);
                    f_s(visible);
                }
            };
        }
    }
}

#endif


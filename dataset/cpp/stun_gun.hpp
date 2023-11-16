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
#ifndef FIRESTR_NETWORK_STUNGUN_H
#define FIRESTR_NETWORK_STUNGUN_H

#include "util/bytes.hpp"

#include <QObject>
#include <QtWidgets>
#include <QtNetwork>
#include <string>
#include <memory>
#include <boost/cstdint.hpp>

namespace fire
{
    namespace network
    {
        enum stun_state { stun_in_progress, stun_failed, stun_success};
        class stun_gun : public QObject
        {
            Q_OBJECT
            public:
                stun_gun(QObject* parent, const std::string& stun_server, const std::string stun_port, const std::string port);

            public:
                void send_stun_request();

            public:
                stun_state state() const;

                const std::string& stun_server() const;
                const std::string& stun_port() const;
                const std::string& internal_port() const;
                const std::string& external_ip() const;
                const std::string& external_port() const;

            public slots:
                void got_response();
                void error(QAbstractSocket::SocketError);

            private:

                QUdpSocket* _socket;
                stun_state _state;
                std::string _stun_server;
                std::string _stun_port;
                std::string _int_port;
                std::string _ext_ip;
                std::string _ext_port;
        };

        using stun_gun_ptr = std::shared_ptr<stun_gun>;
    }
}
#endif


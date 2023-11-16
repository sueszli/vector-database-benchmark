/*
    LANShare - LAN file transfer.
    Copyright (C) 2016 Abdul Aris R. <abdularisrahmanudin10@gmail.com>

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

#include "transferserver.h"

#include "settings.h"

TransferServer::TransferServer(DeviceListModel* devList, QObject *parent) : QObject(parent)
{
    mDevList = devList;
    mServer = new QTcpServer(this);
    connect(mServer, &QTcpServer::newConnection, this, &TransferServer::onNewConnection);
}

bool TransferServer::listen(const QHostAddress &addr)
{
    return mServer->listen(addr, Settings::instance()->getTransferPort());
}

void TransferServer::onNewConnection()
{
    QTcpSocket* socket = mServer->nextPendingConnection();
    if (socket) {
        Device dev = mDevList->device(socket->peerAddress());
        Receiver* rec = new Receiver(dev, socket);
        mReceivers.push_back(rec);
        emit newReceiverAdded(rec);
    }
}

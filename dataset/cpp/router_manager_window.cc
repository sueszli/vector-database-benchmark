//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "client/ui/router_manager/router_manager_window.h"

#include "base/logging.h"
#include "base/peer/user.h"
#include "qt_base/application.h"
#include "client/router.h"
#include "client/router_proxy.h"
#include "client/router_window_proxy.h"
#include "client/ui/client_settings.h"
#include "client/ui/router_manager/router_user_dialog.h"
#include "client/ui/router_manager/router_status_dialog.h"
#include "ui_router_manager_window.h"

#include <QClipboard>
#include <QCollator>
#include <QDateTime>
#include <QFileDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMenu>
#include <QMessageBox>
#include <QTimer>

namespace client {

namespace {

class ColumnAction : public QAction
{
public:
    ColumnAction(const QString& text, int index, QObject* parent)
        : QAction(text, parent),
          index_(index)
    {
        setCheckable(true);
    }

    int columnIndex() const { return index_; }

private:
    const int index_;
    DISALLOW_COPY_AND_ASSIGN(ColumnAction);
};

class HostTreeItem : public QTreeWidgetItem
{
public:
    explicit HostTreeItem(const proto::Session& session)
    {
        updateItem(session);

        QString time = QLocale::system().toString(QDateTime::fromSecsSinceEpoch(
            static_cast<uint>(session.timepoint())), QLocale::ShortFormat);

        setIcon(0, QIcon(":/img/host.png"));
        setText(0, QString::fromStdString(session.computer_name()));
        setText(1, QString::fromStdString(session.ip_address()));
        setText(2, time);

        proto::Version version = session.version();
        setText(4, QString("%1.%2.%3.%4")
            .arg(version.major()).arg(version.minor()).arg(version.patch()).arg(version.revision()));
        setText(5, QString::fromStdString(session.os_name()));
    }

    ~HostTreeItem() override = default;

    void updateItem(const proto::Session& updated_session)
    {
        session = updated_session;
        QString id;

        proto::HostSessionData session_data;
        if (session_data.ParseFromString(session.session_data()))
        {
            for (int i = 0; i < session_data.host_id_size(); ++i)
            {
                id += QString::number(session_data.host_id(i));

                if (i != session_data.host_id_size() - 1)
                    id += QLatin1String(", ");
            }
        }
        else
        {
            LOG(LS_ERROR) << "Unable to parse session data";
        }

        setText(3, id);
    }

    // QTreeWidgetItem implementation.
    bool operator<(const QTreeWidgetItem& other) const override
    {
        int column = treeWidget()->sortColumn();
        if (column == 0)
        {
            QCollator collator;
            collator.setCaseSensitivity(Qt::CaseInsensitive);
            collator.setNumericMode(true);

            return collator.compare(text(0), other.text(0)) <= 0;
        }
        else if (column == 2)
        {
            const HostTreeItem* other_item = static_cast<const HostTreeItem*>(&other);
            return session.timepoint() < other_item->session.timepoint();
        }
        else
        {
            return QTreeWidgetItem::operator<(other);
        }
    }

    proto::Session session;

private:
    DISALLOW_COPY_AND_ASSIGN(HostTreeItem);
};

class RelayTreeItem : public QTreeWidgetItem
{
public:
    explicit RelayTreeItem(const proto::Session& session)
    {
        updateItem(session);

        QString time = QLocale::system().toString(QDateTime::fromSecsSinceEpoch(
            static_cast<uint>(session.timepoint())), QLocale::ShortFormat);

        setIcon(0, QIcon(":/img/computer.png"));
        setText(0, QString::fromStdString(session.ip_address()));
        setText(1, time);

        const proto::Version& version = session.version();
        
        setText(3, QString("%1.%2.%3.%4")
            .arg(version.major()).arg(version.minor()).arg(version.patch()).arg(version.revision()));
        setText(4, QString::fromStdString(session.computer_name()));
        setText(5, QString::fromStdString(session.os_name()));
    }

    void updateItem(const proto::Session& updated_session)
    {
        session = updated_session;

        proto::RelaySessionData session_data;
        if (session_data.ParseFromString(session.session_data()))
        {
            setText(2, QString::number(session_data.pool_size()));
        }
    }

    // QTreeWidgetItem implementation.
    bool operator<(const QTreeWidgetItem &other) const override
    {
        if (treeWidget()->sortColumn() == 1)
        {
            const RelayTreeItem* other_item = static_cast<const RelayTreeItem*>(&other);
            return session.timepoint() < other_item->session.timepoint();
        }

        return QTreeWidgetItem::operator<(other);
    }

    proto::Session session;

private:
    DISALLOW_COPY_AND_ASSIGN(RelayTreeItem);
};

class PeerConnectionTreeItem : public QTreeWidgetItem
{
public:
    explicit PeerConnectionTreeItem(const proto::PeerConnection& connection)
    {
        updateItem(connection);

        setIcon(0, QIcon(":/img/user.png"));
        setText(0, QString::fromStdString(conn.client_user_name()));
        setText(1, QString::number(conn.host_id()));
        setText(2, QString::fromStdString(conn.host_address()));
        setText(3, QString::fromStdString(conn.client_address()));
    }

    void updateItem(const proto::PeerConnection& connection)
    {
        conn = connection;
        setText(4, RouterManagerWindow::sizeToString(conn.bytes_transferred()));
        setText(5, RouterManagerWindow::delayToString(conn.duration()));
        setText(6, RouterManagerWindow::delayToString(conn.idle_time()));
    }

    // QTreeWidgetItem implementation.
    bool operator<(const QTreeWidgetItem &other) const override
    {
        int column = treeWidget()->sortColumn();
        if (column == 1)
        {
            const PeerConnectionTreeItem* other_item =
                static_cast<const PeerConnectionTreeItem*>(&other);
            return conn.host_id() < other_item->conn.host_id();
        }
        else if (column == 4)
        {
            const PeerConnectionTreeItem* other_item =
                static_cast<const PeerConnectionTreeItem*>(&other);
            return conn.bytes_transferred() < other_item->conn.bytes_transferred();
        }
        else if (column == 5)
        {
            const PeerConnectionTreeItem* other_item =
                static_cast<const PeerConnectionTreeItem*>(&other);
            return conn.duration() < other_item->conn.duration();
        }
        else if (column == 6)
        {
            const PeerConnectionTreeItem* other_item =
                static_cast<const PeerConnectionTreeItem*>(&other);
            return conn.idle_time() < other_item->conn.idle_time();
        }

        return QTreeWidgetItem::operator<(other);
    }

    proto::PeerConnection conn;

private:
    DISALLOW_COPY_AND_ASSIGN(PeerConnectionTreeItem);
};

class UserTreeItem : public QTreeWidgetItem
{
public:
    explicit UserTreeItem(const proto::User& user)
        : user(base::User::parseFrom(user))
    {
        setText(0, QString::fromStdString(user.name()));

        if (user.flags() & base::User::ENABLED)
            setIcon(0, QIcon(":/img/user.png"));
        else
            setIcon(0, QIcon(":/img/user-disabled.png"));
    }

    // QTreeWidgetItem implementation.
    bool operator<(const QTreeWidgetItem& other) const override
    {
        int column = treeWidget()->sortColumn();
        if (column == 0)
        {
            QCollator collator;
            collator.setCaseSensitivity(Qt::CaseInsensitive);
            collator.setNumericMode(true);

            return collator.compare(text(0), other.text(0)) <= 0;
        }
        else
        {
            return QTreeWidgetItem::operator<(other);
        }
    }

    base::User user;

private:
    DISALLOW_COPY_AND_ASSIGN(UserTreeItem);
};

//--------------------------------------------------------------------------------------------------
void copyTextToClipboard(const QString& text)
{
    if (text.isEmpty())
        return;

    QClipboard* clipboard = QApplication::clipboard();
    if (!clipboard)
        return;

    clipboard->setText(text);
}

} // namespace

//--------------------------------------------------------------------------------------------------
RouterManagerWindow::RouterManagerWindow(QWidget* parent)
    : QMainWindow(parent),
      ui(std::make_unique<Ui::RouterManagerWindow>()),
      status_dialog_(new RouterStatusDialog(this)),
      window_proxy_(std::make_shared<RouterWindowProxy>(
          qt_base::Application::uiTaskRunner(), this))
{
    LOG(LS_INFO) << "Ctor";
    ui->setupUi(this);

    ui->label_active_conn->setText(tr("Active peers: %1").arg(0));

    connect(ui->button_refresh_hosts, &QPushButton::clicked,
            this, &RouterManagerWindow::refreshSessionList);

    connect(ui->button_disconnect_host, &QPushButton::clicked,
            this, &RouterManagerWindow::disconnectHost);

    connect(ui->button_disconnect_all_hosts, &QPushButton::clicked,
            this, &RouterManagerWindow::disconnectAllHosts);

    connect(ui->button_refresh_relay, &QPushButton::clicked,
            this, &RouterManagerWindow::refreshSessionList);

    connect(ui->button_refresh_users, &QPushButton::clicked,
            this, &RouterManagerWindow::refreshUserList);

    connect(ui->button_add_user, &QPushButton::clicked,
            this, &RouterManagerWindow::addUser);

    connect(ui->button_modify_user, &QPushButton::clicked,
            this, &RouterManagerWindow::modifyUser);

    connect(ui->button_delete_user, &QPushButton::clicked,
            this, &RouterManagerWindow::deleteUser);

    connect(ui->tree_users, &QTreeWidget::currentItemChanged,
            this, &RouterManagerWindow::onCurrentUserChanged);

    connect(ui->tree_hosts, &QTreeWidget::currentItemChanged,
            this, &RouterManagerWindow::onCurrentHostChanged);

    connect(ui->tree_relay, &QTreeWidget::currentItemChanged,
            this, &RouterManagerWindow::onCurrentRelayChanged);

    ui->tree_hosts->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->tree_hosts, &QTreeWidget::customContextMenuRequested,
            this, &RouterManagerWindow::onHostsContextMenu);

    ui->tree_relay->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->tree_relay, &QTreeWidget::customContextMenuRequested,
            this, &RouterManagerWindow::onRelaysContextMenu);

    ui->tree_active_conn->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->tree_active_conn, &QTreeWidget::customContextMenuRequested,
            this, &RouterManagerWindow::onActiveConnContextMenu);

    ui->tree_users->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->tree_users, &QTreeWidget::customContextMenuRequested,
            this, &RouterManagerWindow::onUsersContextMenu);

    connect(ui->button_save_hosts, &QPushButton::clicked,
            this, &RouterManagerWindow::saveHostsToFile);

    connect(ui->button_save_relays, &QPushButton::clicked,
            this, &RouterManagerWindow::saveRelaysToFile);

    ui->tree_hosts->header()->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->tree_hosts->header(), &QHeaderView::customContextMenuRequested,
            this, [this](const QPoint& pos)
    {
        onContextMenuForTreeHeader(ui->tree_hosts, pos);
    });

    ui->tree_relay->header()->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->tree_relay->header(), &QHeaderView::customContextMenuRequested,
            this, [this](const QPoint& pos)
    {
        onContextMenuForTreeHeader(ui->tree_relay, pos);
    });

    ui->tree_active_conn->header()->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->tree_active_conn->header(), &QHeaderView::customContextMenuRequested,
            this, [this](const QPoint& pos)
    {
        onContextMenuForTreeHeader(ui->tree_active_conn, pos);
    });

    ui->tree_hosts->setMouseTracking(true);
    connect(ui->tree_hosts, &QTreeWidget::itemEntered,
            this, [this](QTreeWidgetItem* /* item */, int column)
    {
        current_hosts_column_ = column;
    });

    ui->tree_relay->setMouseTracking(true);
    connect(ui->tree_relay, &QTreeWidget::itemEntered,
            this, [this](QTreeWidgetItem* /* item */, int column)
    {
        current_relays_column_ = column;
    });

    ui->tree_active_conn->setMouseTracking(true);
    connect(ui->tree_active_conn, &QTreeWidget::itemEntered,
            this, [this](QTreeWidgetItem* /* item */, int column)
    {
        current_active_conn_column_ = column;
    });

    ClientSettings settings;
    restoreState(settings.routerManagerState());

    QTimer* update_timer = new QTimer(this);
    connect(update_timer, &QTimer::timeout, this, [this]()
    {
        refreshSessionList();
    });
    update_timer->start(std::chrono::seconds(3));
}

//--------------------------------------------------------------------------------------------------
RouterManagerWindow::~RouterManagerWindow()
{
    LOG(LS_INFO) << "Dtor";

    ClientSettings settings;
    settings.setRouterManagerState(saveState());

    window_proxy_->dettach();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::connectToRouter(const RouterConfig& router_config)
{
    LOG(LS_INFO) << "Connecting to router";

    peer_address_ = QString::fromStdU16String(router_config.address);
    peer_port_ = router_config.port;

    status_dialog_->setStatus(tr("Connecting to %1:%2...").arg(peer_address_).arg(peer_port_));
    status_dialog_->show();
    status_dialog_->activateWindow();

    connect(status_dialog_, &RouterStatusDialog::sig_canceled, [this]()
    {
        LOG(LS_INFO) << "Connection canceled";
        router_proxy_.reset();
        close();
    });

    std::unique_ptr<Router> router = std::make_unique<Router>(
        window_proxy_, qt_base::Application::ioTaskRunner());

    router->setUserName(router_config.username);
    router->setPassword(router_config.password);

    router_proxy_ = std::make_unique<RouterProxy>(
        qt_base::Application::ioTaskRunner(), std::move(router));

    router_proxy_->connectToRouter(router_config.address, router_config.port);
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onConnected(const base::Version& peer_version)
{
    LOG(LS_INFO) << "Connected to router";
    status_dialog_->hide();

    ui->statusbar->showMessage(tr("Connected to: %1:%2 (version %3)")
                               .arg(peer_address_)
                               .arg(peer_port_)
                               .arg(QString::fromStdString(peer_version.toString())));
    show();
    activateWindow();

    if (router_proxy_)
    {
        router_proxy_->refreshSessionList();
        router_proxy_->refreshUserList();
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onDisconnected(base::TcpChannel::ErrorCode error_code)
{
    const char* message;

    switch (error_code)
    {
        case base::TcpChannel::ErrorCode::INVALID_PROTOCOL:
            message = QT_TR_NOOP("Violation of the communication protocol.");
            break;

        case base::TcpChannel::ErrorCode::ACCESS_DENIED:
            message = QT_TR_NOOP("Cryptography error (message encryption or decryption failed).");
            break;

        case base::TcpChannel::ErrorCode::NETWORK_ERROR:
            message = QT_TR_NOOP("An error occurred with the network (e.g., the network cable was accidentally plugged out).");
            break;

        case base::TcpChannel::ErrorCode::CONNECTION_REFUSED:
            message = QT_TR_NOOP("Connection was refused by the peer (or timed out).");
            break;

        case base::TcpChannel::ErrorCode::REMOTE_HOST_CLOSED:
            message = QT_TR_NOOP("Remote host closed the connection.");
            break;

        case base::TcpChannel::ErrorCode::SPECIFIED_HOST_NOT_FOUND:
            message = QT_TR_NOOP("Host address was not found.");
            break;

        case base::TcpChannel::ErrorCode::SOCKET_TIMEOUT:
            message = QT_TR_NOOP("Socket operation timed out.");
            break;

        case base::TcpChannel::ErrorCode::ADDRESS_IN_USE:
            message = QT_TR_NOOP("Address specified is already in use and was set to be exclusive.");
            break;

        case base::TcpChannel::ErrorCode::ADDRESS_NOT_AVAILABLE:
            message = QT_TR_NOOP("Address specified does not belong to the host.");
            break;

        default:
        {
            if (error_code != base::TcpChannel::ErrorCode::UNKNOWN)
            {
                LOG(LS_ERROR) << "Unknown error code: " << static_cast<int>(error_code);
            }

            message = QT_TR_NOOP("An unknown error occurred.");
        }
        break;
    }

    status_dialog_->setStatus(tr("Error: %1").arg(tr(message)));
    status_dialog_->show();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onVersionMismatch(const base::Version& router, const base::Version& client)
{
    QString router_version = QString::fromStdString(router.toString());
    QString client_version = QString::fromStdString(client.toString());
    QString message = tr("The Router version is newer than the Client version (%1 > %2). "
                         "Please update the application.").arg(router_version).arg(client_version);

    status_dialog_->setStatus(tr("Error: %1").arg(message));
    status_dialog_->show();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onAccessDenied(base::ClientAuthenticator::ErrorCode error_code)
{
    const char* message;

    switch (error_code)
    {
        case base::ClientAuthenticator::ErrorCode::SUCCESS:
            message = QT_TR_NOOP("Authentication successfully completed.");
            break;

        case base::ClientAuthenticator::ErrorCode::NETWORK_ERROR:
            message = QT_TR_NOOP("Network authentication error.");
            break;

        case base::ClientAuthenticator::ErrorCode::PROTOCOL_ERROR:
            message = QT_TR_NOOP("Violation of the data exchange protocol.");
            break;

        case base::ClientAuthenticator::ErrorCode::ACCESS_DENIED:
            message = QT_TR_NOOP("An error occured while authenticating: wrong user name or password.");
            break;

        case base::ClientAuthenticator::ErrorCode::SESSION_DENIED:
            message = QT_TR_NOOP("Specified session type is not allowed for the user.");
            break;

        default:
            message = QT_TR_NOOP("An unknown error occurred.");
            break;
    }

    status_dialog_->setStatus(tr("Error: %1").arg(tr(message)));
    status_dialog_->show();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onSessionList(std::shared_ptr<proto::SessionList> session_list)
{
    QTreeWidget* tree_hosts = ui->tree_hosts;
    QTreeWidget* tree_relay = ui->tree_relay;

    int host_count = 0;
    int relay_count = 0;

    auto has_session_with_id = [](const proto::SessionList& session_list, int64_t session_id)
    {
        for (int i = 0; i < session_list.session_size(); ++i)
        {
            if (session_list.session(i).session_id() == session_id)
                return true;
        }

        return false;
    };

    // Remove from the UI all hosts that are not in the list.
    for (int i = tree_hosts->topLevelItemCount() - 1; i >= 0; --i)
    {
        HostTreeItem* item = static_cast<HostTreeItem*>(tree_hosts->topLevelItem(i));

        if (!has_session_with_id(*session_list, item->session.session_id()))
             delete item;
    }

    // Remove from the UI all relays that are not in the list.
    for (int i = tree_relay->topLevelItemCount() - 1; i >= 0; --i)
    {
        RelayTreeItem* item = static_cast<RelayTreeItem*>(tree_relay->topLevelItem(i));

        if (!has_session_with_id(*session_list, item->session.session_id()))
             delete item;
    }

    // Adding and updating elements in the UI.
    for (int i = 0; i < session_list->session_size(); ++i)
    {
        const proto::Session& session = session_list->session(i);

        switch (session.session_type())
        {
            case proto::ROUTER_SESSION_HOST:
            {
                bool found = false;

                for (int j = 0; j < tree_hosts->topLevelItemCount(); ++j)
                {
                    HostTreeItem* item = static_cast<HostTreeItem*>(tree_hosts->topLevelItem(j));
                    if (item->session.session_id() == session.session_id())
                    {
                        item->updateItem(session);
                        found = true;
                        break;
                    }
                }

                if (!found)
                    tree_hosts->addTopLevelItem(new HostTreeItem(session));

                ++host_count;
            }
            break;

            case proto::ROUTER_SESSION_RELAY:
            {
                bool found = false;

                for (int j = 0; j < tree_relay->topLevelItemCount(); ++j)
                {
                    RelayTreeItem* item = static_cast<RelayTreeItem*>(tree_relay->topLevelItem(j));
                    if (item->session.session_id() == session.session_id())
                    {
                        item->updateItem(session);
                        found = true;
                        break;
                    }
                }

                if (!found)
                    tree_relay->addTopLevelItem(new RelayTreeItem(session));

                ++relay_count;
            }
            break;

            default:
                break;
        }
    }

    updateRelayStatistics();

    ui->label_hosts_conn_count->setText(QString::number(host_count));
    ui->label_relay_conn_count->setText(QString::number(relay_count));

    if (relay_count == 0)
    {
        ui->label_active_conn->setEnabled(false);
        ui->tree_active_conn->setEnabled(false);
        ui->tree_active_conn->clear();
    }

    afterRequest();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onSessionResult(std::shared_ptr<proto::SessionResult> session_result)
{
    if (session_result->error_code() != proto::SessionResult::SUCCESS)
    {
        const char* message;

        switch (session_result->error_code())
        {
            case proto::SessionResult::INVALID_REQUEST:
                message = QT_TR_NOOP("Invalid request.");
                break;

            case proto::SessionResult::INTERNAL_ERROR:
                message = QT_TR_NOOP("Unknown internal error.");
                break;

            case proto::SessionResult::INVALID_SESSION_ID:
                message = QT_TR_NOOP("Invalid session ID was passed.");
                break;

            default:
                message = QT_TR_NOOP("Unknown error type.");
                break;
        }

        QMessageBox::warning(this, tr("Warning"), tr(message), QMessageBox::Ok);
    }

    refreshSessionList();
    afterRequest();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onUserList(std::shared_ptr<proto::UserList> user_list)
{
    QTreeWidget* tree_users = ui->tree_users;
    tree_users->clear();

    for (int i = 0; i < user_list->user_size(); ++i)
        tree_users->addTopLevelItem(new UserTreeItem(user_list->user(i)));

    afterRequest();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onUserResult(std::shared_ptr<proto::UserResult> user_result)
{
    if (user_result->error_code() != proto::UserResult::SUCCESS)
    {
        const char* message;

        switch (user_result->error_code())
        {
            case proto::UserResult::INTERNAL_ERROR:
                message = QT_TR_NOOP("Unknown internal error.");
                break;

            case proto::UserResult::INVALID_DATA:
                message = QT_TR_NOOP("Invalid data was passed.");
                break;

            case proto::UserResult::ALREADY_EXISTS:
                message = QT_TR_NOOP("A user with the specified name already exists.");
                break;

            default:
                message = QT_TR_NOOP("Unknown error type.");
                break;
        }

        QMessageBox::warning(this, tr("Warning"), tr(message), QMessageBox::Ok);
    }

    refreshUserList();
    afterRequest();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::closeEvent(QCloseEvent* /* event */)
{
    if (router_proxy_)
        router_proxy_->disconnectFromRouter();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onHostsContextMenu(const QPoint& pos)
{
    QMenu menu;

    QAction* disconnect_action = menu.addAction(tr("Disconnect"));

    menu.addSeparator();

    QAction* disconnect_all_action = menu.addAction(tr("Disconnect All"));
    QAction* refresh_action = menu.addAction(tr("Refresh"));

    menu.addSeparator();

    QAction* copy_row = menu.addAction(tr("Copy Row"));
    QAction* copy_col = menu.addAction(tr("Copy Value"));

    menu.addSeparator();

    QAction* save_action = menu.addAction(tr("Save to file..."));

    QTreeWidgetItem* current_item = ui->tree_hosts->currentItem();
    if (!current_item)
    {
        copy_row->setEnabled(false);
        copy_col->setEnabled(false);
    }

    QAction* action = menu.exec(ui->tree_hosts->viewport()->mapToGlobal(pos));
    if (action == save_action)
    {
        saveHostsToFile();
    }
    else if (action == disconnect_action)
    {
        disconnectHost();
    }
    else if (action == disconnect_all_action)
    {
        disconnectAllHosts();
    }
    else if (action == refresh_action)
    {
        refreshSessionList();
    }
    else if (action == copy_row)
    {
        copyRowFromTree(ui->tree_hosts->currentItem());
    }
    else if (action == copy_col)
    {
        copyColumnFromTree(ui->tree_hosts->currentItem(), current_hosts_column_);
    }
    else
    {
        // Nothing
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onRelaysContextMenu(const QPoint& pos)
{
    QMenu menu;

    QAction* disconnect_action = menu.addAction(tr("Disconnect"));
    menu.addSeparator();
    QAction* disconnect_all_action = menu.addAction(tr("Disconnect All"));
    QAction* refresh_action = menu.addAction(tr("Refresh"));
    menu.addSeparator();
    QAction* copy_row = menu.addAction(tr("Copy Row"));
    QAction* copy_col = menu.addAction(tr("Copy Value"));
    menu.addSeparator();
    QAction* save_action = menu.addAction(tr("Save to file..."));

    QTreeWidgetItem* current_item = ui->tree_relay->currentItem();
    if (!current_item)
    {
        copy_row->setEnabled(false);
        copy_col->setEnabled(false);
    }

    QAction* action = menu.exec(ui->tree_relay->viewport()->mapToGlobal(pos));
    if (action == save_action)
    {
        saveRelaysToFile();
    }
    else if (action == copy_row)
    {
        copyRowFromTree(current_item);
    }
    else if (action == copy_col)
    {
        copyColumnFromTree(current_item, current_relays_column_);
    }
    else if (action == refresh_action)
    {
        refreshSessionList();
    }
    else if (action == disconnect_action)
    {
        disconnectRelay();
    }
    else if (action == disconnect_all_action)
    {
        disconnectAllRelays();
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onActiveConnContextMenu(const QPoint& pos)
{
    RelayTreeItem* relay_item = static_cast<RelayTreeItem*>(ui->tree_relay->currentItem());
    if (!relay_item)
        return;

    QMenu menu;

    QAction* disconnect = menu.addAction(tr("Disconnect"));
    menu.addSeparator();
    QAction* disconnect_all = menu.addAction(tr("Disconnect All"));
    menu.addSeparator();
    QAction* copy_row = menu.addAction(tr("Copy Row"));
    QAction* copy_col = menu.addAction(tr("Copy Value"));

    PeerConnectionTreeItem* peer_item =
        static_cast<PeerConnectionTreeItem*>(ui->tree_active_conn->currentItem());
    if (!peer_item)
    {
        disconnect->setEnabled(false);
        copy_row->setEnabled(false);
        copy_col->setEnabled(false);

        if (ui->tree_active_conn->topLevelItemCount() == 0)
            return;
    }

    QAction* action = menu.exec(ui->tree_active_conn->viewport()->mapToGlobal(pos));
    if (action == copy_row)
    {
        copyRowFromTree(peer_item);
    }
    else if (action == copy_col)
    {
        copyColumnFromTree(peer_item, current_active_conn_column_);
    }
    else if (action == disconnect)
    {
        if (router_proxy_)
        {
            router_proxy_->disconnectPeerSession(
                relay_item->session.session_id(), peer_item->conn.session_id());
        }
    }
    else if (action == disconnect_all)
    {
        if (router_proxy_)
        {
            for (int i = 0; i < ui->tree_active_conn->topLevelItemCount(); ++i)
            {
                PeerConnectionTreeItem* item =
                    static_cast<PeerConnectionTreeItem*>(ui->tree_active_conn->topLevelItem(i));
                if (item)
                {
                    router_proxy_->disconnectPeerSession(
                        relay_item->session.session_id(), item->conn.session_id());
                }
            }
        }
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onUsersContextMenu(const QPoint& pos)
{
    QMenu menu;

    QAction* modify_action = menu.addAction(tr("Modify"));
    modify_action->setIcon(QIcon(":/img/pencil.png"));

    QAction* delete_action = menu.addAction(tr("Delete"));
    delete_action->setIcon(QIcon(":/img/minus.png"));

    QTreeWidgetItem* current_item = ui->tree_users->itemAt(pos);
    if (!current_item)
    {
        modify_action->setVisible(false);
        delete_action->setVisible(false);
    }
    else
    {
        menu.addSeparator();
    }

    QAction* add_action = menu.addAction(tr("Add"));
    add_action->setIcon(QIcon(":/img/plus.png"));

    QAction* refresh_action = menu.addAction(tr("Refresh"));
    refresh_action->setIcon(QIcon(":/img/arrow-circle-double.png"));

    QAction* action = menu.exec(ui->tree_users->viewport()->mapToGlobal(pos));
    if (action == modify_action)
    {
        modifyUser();
    }
    else if (action == delete_action)
    {
        deleteUser();
    }
    else if (action == add_action)
    {
        addUser();
    }
    else if (action == refresh_action)
    {
        refreshUserList();
    }
    else
    {
        // Nothing
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onContextMenuForTreeHeader(QTreeWidget* tree, const QPoint& pos)
{
    QHeaderView* header = tree->header();
    QMenu menu;

    for (int i = 1; i < header->count(); ++i)
    {
        ColumnAction* action = new ColumnAction(tree->headerItem()->text(i), i, &menu);
        action->setChecked(!header->isSectionHidden(i));
        menu.addAction(action);
    }

    ColumnAction* action = dynamic_cast<ColumnAction*>(
        menu.exec(header->viewport()->mapToGlobal(pos)));
    if (!action)
        return;

    header->setSectionHidden(action->columnIndex(), !action->isChecked());
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::copyRowFromTree(QTreeWidgetItem* item)
{
    if (!item)
        return;

    QString result;

    int column_count = item->columnCount();
    if (column_count > 2)
    {
        for (int i = 0; i < column_count; ++i)
        {
            QString text = item->text(i);

            if (!text.isEmpty())
                result += text + QLatin1Char(' ');
        }

        result.chop(1);
    }
    else
    {
        result = item->text(0) + QLatin1String(": ") + item->text(1);
    }

    copyTextToClipboard(result);
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::copyColumnFromTree(QTreeWidgetItem* item, int column)
{
    if (!item)
        return;

    copyTextToClipboard(item->text(column));
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::updateRelayStatistics()
{
    RelayTreeItem* item = static_cast<RelayTreeItem*>(ui->tree_relay->currentItem());
    if (item)
    {
        ui->label_active_conn->setEnabled(true);
        ui->tree_active_conn->setEnabled(true);

        proto::RelaySessionData session_data;
        if (session_data.ParseFromString(item->session.session_data()))
        {
            if (session_data.has_relay_stat())
            {
                const proto::RelaySessionData::RelayStat& relay_stat = session_data.relay_stat();

                auto has_session_with_id = [](const proto::RelaySessionData::RelayStat& relay_stat,
                    uint64_t session_id)
                {
                    for (int i = 0; i < relay_stat.peer_connection_size(); ++i)
                    {
                        if (relay_stat.peer_connection(i).session_id() == session_id)
                            return true;
                    }

                    return false;
                };

                // Remove from the UI all connections that are not in the list.
                for (int i = ui->tree_active_conn->topLevelItemCount() - 1; i >= 0; --i)
                {
                    PeerConnectionTreeItem* item =
                        static_cast<PeerConnectionTreeItem*>(ui->tree_active_conn->topLevelItem(i));
                    if (!has_session_with_id(relay_stat, item->conn.session_id()))
                        delete item;
                }

                ui->label_active_conn->setText(
                    tr("Active peers: %1").arg(relay_stat.peer_connection_size()));

                for (int i = 0; i < relay_stat.peer_connection_size(); ++i)
                {
                    const proto::PeerConnection& connection = relay_stat.peer_connection(i);

                    bool found = false;

                    for (int j = 0; j < ui->tree_active_conn->topLevelItemCount(); ++j)
                    {
                        PeerConnectionTreeItem* item =
                            static_cast<PeerConnectionTreeItem*>(ui->tree_active_conn->topLevelItem(j));

                        if (item->conn.session_id() == connection.session_id())
                        {
                            item->updateItem(connection);
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                        ui->tree_active_conn->addTopLevelItem(new PeerConnectionTreeItem(connection));
                }

                return;
            }
            else
            {
                LOG(LS_INFO) << "No relay statistics";
            }
        }
        else
        {
            LOG(LS_ERROR) << "Unable to parse session data";
        }
    }

    ui->label_active_conn->setEnabled(false);
    ui->tree_active_conn->setEnabled(false);
    ui->label_active_conn->setText(tr("Active peers: %1").arg(0));
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::refreshSessionList()
{
    if (router_proxy_)
    {
        beforeRequest();
        router_proxy_->refreshSessionList();
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::disconnectRelay()
{
    RelayTreeItem* tree_item = static_cast<RelayTreeItem*>(ui->tree_hosts->currentItem());
    if (!tree_item)
        return;

    QMessageBox message_box(QMessageBox::Question,
                            tr("Confirmation"),
                            tr("Are you sure you want to disconnect session \"%1\"?")
                                .arg(QString::fromStdString(tree_item->session.computer_name())),
                            QMessageBox::Yes | QMessageBox::No,
                            this);
    message_box.button(QMessageBox::Yes)->setText(tr("Yes"));
    message_box.button(QMessageBox::No)->setText(tr("No"));

    if (message_box.exec() == QMessageBox::No)
    {
        return;
    }

    if (router_proxy_)
    {
        beforeRequest();
        router_proxy_->stopSession(tree_item->session.session_id());
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::disconnectAllRelays()
{
    if (!router_proxy_)
        return;

    QMessageBox message_box(QMessageBox::Question,
                            tr("Confirmation"),
                            tr("Are you sure you want to disconnect all relays?"),
                            QMessageBox::Yes | QMessageBox::No,
                            this);
    message_box.button(QMessageBox::Yes)->setText(tr("Yes"));
    message_box.button(QMessageBox::No)->setText(tr("No"));

    if (message_box.exec() == QMessageBox::No)
        return;

    beforeRequest();

    QTreeWidget* tree_relay = ui->tree_relay;
    for (int i = 0; i < tree_relay->topLevelItemCount(); ++i)
    {
        RelayTreeItem* tree_item = static_cast<RelayTreeItem*>(tree_relay->topLevelItem(i));
        if (tree_item)
            router_proxy_->stopSession(tree_item->session.session_id());
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::disconnectHost()
{
    HostTreeItem* tree_item = static_cast<HostTreeItem*>(ui->tree_hosts->currentItem());
    if (!tree_item)
        return;

    QMessageBox message_box(QMessageBox::Question,
                            tr("Confirmation"),
                            tr("Are you sure you want to disconnect session \"%1\"?")
                                .arg(QString::fromStdString(tree_item->session.computer_name())),
                            QMessageBox::Yes | QMessageBox::No,
                            this);
    message_box.button(QMessageBox::Yes)->setText(tr("Yes"));
    message_box.button(QMessageBox::No)->setText(tr("No"));

    if (message_box.exec() == QMessageBox::No)
    {
        return;
    }

    if (router_proxy_)
    {
        beforeRequest();
        router_proxy_->stopSession(tree_item->session.session_id());
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::disconnectAllHosts()
{
    if (!router_proxy_)
        return;

    QMessageBox message_box(QMessageBox::Question,
                            tr("Confirmation"),
                            tr("Are you sure you want to disconnect all hosts?"),
                            QMessageBox::Yes | QMessageBox::No,
                            this);
    message_box.button(QMessageBox::Yes)->setText(tr("Yes"));
    message_box.button(QMessageBox::No)->setText(tr("No"));

    if (message_box.exec() == QMessageBox::No)
        return;

    beforeRequest();

    QTreeWidget* tree_hosts = ui->tree_hosts;
    for (int i = 0; i < tree_hosts->topLevelItemCount(); ++i)
    {
        HostTreeItem* tree_item = static_cast<HostTreeItem*>(tree_hosts->topLevelItem(i));
        if (tree_item)
            router_proxy_->stopSession(tree_item->session.session_id());
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::refreshUserList()
{
    if (router_proxy_)
    {
        beforeRequest();
        router_proxy_->refreshUserList();
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::addUser()
{
    QTreeWidget* tree_users = ui->tree_users;
    std::vector<std::u16string> users;

    for (int i = 0; i < tree_users->topLevelItemCount(); ++i)
        users.emplace_back(static_cast<UserTreeItem*>(tree_users->topLevelItem(i))->user.name);

    RouterUserDialog dialog(base::User(), users, this);
    if (dialog.exec() == QDialog::Accepted)
    {
        if (router_proxy_)
        {
            beforeRequest();
            router_proxy_->addUser(dialog.user().serialize());
        }
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::modifyUser()
{
    LOG(LS_INFO) << "[ACTION] Modify user";

    QTreeWidget* tree_users = ui->tree_users;
    UserTreeItem* tree_item = static_cast<UserTreeItem*>(tree_users->currentItem());
    if (!tree_item)
    {
        LOG(LS_INFO) << "No selected item";
        return;
    }

    std::vector<std::u16string> users;

    for (int i = 0; i < tree_users->topLevelItemCount(); ++i)
    {
        UserTreeItem* current_item = static_cast<UserTreeItem*>(tree_users->topLevelItem(i));
        if (current_item->text(0).compare(tree_item->text(0), Qt::CaseInsensitive) != 0)
            users.emplace_back(current_item->user.name);
    }

    RouterUserDialog dialog(tree_item->user, users, this);
    if (dialog.exec() == QDialog::Accepted)
    {
        if (router_proxy_)
        {
            beforeRequest();
            router_proxy_->modifyUser(dialog.user().serialize());
        }
    }

}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::deleteUser()
{
    LOG(LS_INFO) << "[ACTION] Delete user";

    UserTreeItem* tree_item = static_cast<UserTreeItem*>(ui->tree_users->currentItem());
    if (!tree_item)
    {
        LOG(LS_INFO) << "No selected item";
        return;
    }

    int64_t entry_id = tree_item->user.entry_id;
    if (entry_id == 1)
    {
        LOG(LS_INFO) << "Unable to delete built-in user";
        QMessageBox::warning(this, tr("Warning"), tr("You cannot delete a built-in user."));
        return;
    }

    QMessageBox message_box(QMessageBox::Question,
                            tr("Confirmation"),
                            tr("Are you sure you want to delete user \"%1\"?")
                                .arg(tree_item->text(0)),
                            QMessageBox::Yes | QMessageBox::No,
                            this);
    message_box.button(QMessageBox::Yes)->setText(tr("Yes"));
    message_box.button(QMessageBox::No)->setText(tr("No"));

    if (message_box.exec() == QMessageBox::Yes)
    {
        LOG(LS_INFO) << "[ACTION] Accepted by user";
        if (router_proxy_)
        {
            beforeRequest();
            router_proxy_->deleteUser(entry_id);
        }
    }
    else
    {
        LOG(LS_INFO) << "[ACTION] Rejected by user";
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onCurrentUserChanged(
    QTreeWidgetItem* /* current */, QTreeWidgetItem* /* previous */)
{
    ui->button_modify_user->setEnabled(true);
    ui->button_delete_user->setEnabled(true);
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onCurrentHostChanged(QTreeWidgetItem* /* current */,
                                               QTreeWidgetItem* /* previous */)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::onCurrentRelayChanged(
    QTreeWidgetItem* /* current */, QTreeWidgetItem* /* previous */)
{
    ui->tree_active_conn->clear();
    updateRelayStatistics();
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::beforeRequest()
{
    //
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::afterRequest()
{
    //
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::saveHostsToFile()
{
    LOG(LS_INFO) << "[ACTION] Save hosts to file";

    QString selected_filter;
    QString file_path = QFileDialog::getSaveFileName(this,
                                                     tr("Save File"),
                                                     QString(),
                                                     tr("JSON files (*.json)"),
                                                     &selected_filter);
    if (file_path.isEmpty() || selected_filter.isEmpty())
    {
        LOG(LS_INFO) << "No selected path";
        return;
    }

    QFile file(file_path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        LOG(LS_INFO) << "Unable to open file: " << file.errorString().toStdString();
        QMessageBox::warning(this,
                             tr("Warning"),
                             tr("Could not open file for writing."),
                             QMessageBox::Ok);
        return;
    }

    QTreeWidget* tree = ui->tree_hosts;
    QJsonArray root_array;

    for (int i = 0; i < tree->topLevelItemCount(); ++i)
    {
        const proto::Session& session =
            static_cast<HostTreeItem*>(tree->topLevelItem(i))->session;

        QJsonObject host_object;

        host_object.insert("session_id", QString::number(session.session_id()));
        host_object.insert("computer_name", QString::fromStdString(session.computer_name()));
        host_object.insert("operating_system", QString::fromStdString(session.os_name()));
        host_object.insert("ip_address", QString::fromStdString(session.ip_address()));

        QString version = QString("%1.%2.%3.%4")
            .arg(session.version().major()).arg(session.version().minor())
            .arg(session.version().patch()).arg(session.version().revision());
        host_object.insert("version", version);

        QString time = QLocale::system().toString(QDateTime::fromSecsSinceEpoch(
            static_cast<uint>(session.timepoint())), QLocale::ShortFormat);
        host_object.insert("connect_time", time);

        QJsonArray id_array;

        proto::HostSessionData session_data;
        if (session_data.ParseFromString(session.session_data()))
        {
            for (int i = 0; i < session_data.host_id_size(); ++i)
                id_array.append(QString::number(session_data.host_id(i)));
        }

        host_object.insert("host_ids", id_array);
        root_array.append(host_object);
    }

    QJsonObject root_object;
    root_object.insert("hosts", root_array);

    int64_t written = file.write(QJsonDocument(root_object).toJson());
    if (written <= 0)
    {
        LOG(LS_INFO) << "Unable to write file: " << file.errorString().toStdString();
        QMessageBox::warning(this,
                             tr("Warning"),
                             tr("Unable to write file."),
                             QMessageBox::Ok);
        return;
    }
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::saveRelaysToFile()
{
    LOG(LS_INFO) << "[ACTION] Save relays to file";

    QString selected_filter;
    QString file_path = QFileDialog::getSaveFileName(this,
                                                     tr("Save File"),
                                                     QString(),
                                                     tr("JSON files (*.json)"),
                                                     &selected_filter);
    if (file_path.isEmpty() || selected_filter.isEmpty())
    {
        LOG(LS_INFO) << "No selected path";
        return;
    }

    QFile file(file_path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        LOG(LS_INFO) << "Unable to open file: " << file.errorString().toStdString();
        QMessageBox::warning(this,
                             tr("Warning"),
                             tr("Could not open file for writing."),
                             QMessageBox::Ok);
        return;
    }

    QTreeWidget* tree = ui->tree_relay;
    QJsonArray root_array;

    for (int i = 0; i < tree->topLevelItemCount(); ++i)
    {
        const proto::Session& session =
            static_cast<RelayTreeItem*>(tree->topLevelItem(i))->session;

        QJsonObject relay_object;

        relay_object.insert("session_id", QString::number(session.session_id()));
        relay_object.insert("computer_name", QString::fromStdString(session.computer_name()));
        relay_object.insert("operating_system", QString::fromStdString(session.os_name()));
        relay_object.insert("ip_address", QString::fromStdString(session.ip_address()));

        QString version = QString("%1.%2.%3.%4")
            .arg(session.version().major()).arg(session.version().minor())
            .arg(session.version().patch()).arg(session.version().revision());
        relay_object.insert("version", version);

        QString time = QLocale::system().toString(QDateTime::fromSecsSinceEpoch(
            static_cast<uint>(session.timepoint())), QLocale::ShortFormat);
        relay_object.insert("connect_time", time);

        proto::RelaySessionData session_data;
        if (session_data.ParseFromString(session.session_data()))
            relay_object.insert("pool_size", QString::number(session_data.pool_size()));

        if (session_data.has_relay_stat())
        {
            const proto::RelaySessionData::RelayStat& relay_stat = session_data.relay_stat();

            QJsonArray active_array;
            for (int i = 0; i < relay_stat.peer_connection_size(); ++i)
            {
                const proto::PeerConnection& conn = relay_stat.peer_connection(i);

                QJsonObject conn_object;
                conn_object.insert("host_address", QString::fromStdString(conn.host_address()));
                conn_object.insert("host_id", QString::number(conn.host_id()));
                conn_object.insert("client_address", QString::fromStdString(conn.client_address()));
                conn_object.insert("user_name", QString::fromStdString(conn.client_user_name()));
                conn_object.insert("bytes_transferred", static_cast<long long>(conn.bytes_transferred()));
                conn_object.insert("duration", static_cast<long long>(conn.duration()));
                conn_object.insert("idle", static_cast<long long>(conn.idle_time()));

                active_array.append(conn_object);
            }

            relay_object.insert("connections", active_array);
            relay_object.insert("uptime", static_cast<long long>(relay_stat.uptime()));
        }

        root_array.append(relay_object);
    }

    QJsonObject root_object;
    root_object.insert("relays", root_array);

    int64_t written = file.write(QJsonDocument(root_object).toJson());
    if (written <= 0)
    {
        LOG(LS_INFO) << "Unable to write file: " << file.errorString().toStdString();
        QMessageBox::warning(this,
                             tr("Warning"),
                             tr("Unable to write file."),
                             QMessageBox::Ok);
        return;
    }
}

//--------------------------------------------------------------------------------------------------
QByteArray RouterManagerWindow::saveState()
{
    QByteArray buffer;

    {
        QDataStream stream(&buffer, QIODevice::WriteOnly);
        stream.setVersion(QDataStream::Qt_5_12);

        stream << saveGeometry();
        stream << ui->tree_hosts->header()->saveState();
        stream << ui->tree_relay->header()->saveState();
        stream << ui->splitter_relay_hori->saveState();
        stream << ui->tree_active_conn->header()->saveState();
    }

    return buffer;
}

//--------------------------------------------------------------------------------------------------
void RouterManagerWindow::restoreState(const QByteArray& state)
{
    QDataStream stream(state);
    stream.setVersion(QDataStream::Qt_5_12);

    QByteArray window_geometry;
    QByteArray hosts_columns_state;
    QByteArray relays_columns_state;
    QByteArray relay_splitter_hori;
    QByteArray active_conn_columns_state;

    stream >> window_geometry;
    stream >> hosts_columns_state;
    stream >> relays_columns_state;
    stream >> relay_splitter_hori;
    stream >> active_conn_columns_state;

    if (!window_geometry.isEmpty())
        restoreGeometry(window_geometry);

    if (!hosts_columns_state.isEmpty())
        ui->tree_hosts->header()->restoreState(hosts_columns_state);

    if (!relays_columns_state.isEmpty())
        ui->tree_relay->header()->restoreState(relays_columns_state);

    if (!relay_splitter_hori.isEmpty())
    {
        ui->splitter_relay_hori->restoreState(relay_splitter_hori);
    }
    else
    {
        int side_size = height() / 2;

        QList<int> sizes;
        sizes.push_back(side_size);
        sizes.push_back(side_size);
        ui->splitter_relay_hori->setSizes(sizes);
    }

    if (!active_conn_columns_state.isEmpty())
        ui->tree_active_conn->header()->restoreState(active_conn_columns_state);
}

//--------------------------------------------------------------------------------------------------
// static
QString RouterManagerWindow::delayToString(uint64_t delay)
{
    uint64_t days = (delay / 86400);
    uint64_t hours = (delay % 86400) / 3600;
    uint64_t minutes = ((delay % 86400) % 3600) / 60;
    uint64_t seconds = ((delay % 86400) % 3600) % 60;

    QString seconds_string = tr("%n seconds", "", static_cast<int>(seconds));
    QString minutes_string = tr("%n minutes", "", static_cast<int>(minutes));
    QString hours_string = tr("%n hours", "", static_cast<int>(hours));

    if (!days)
    {
        if (!hours)
        {
            if (!minutes)
            {
                return seconds_string;
            }
            else
            {
                return minutes_string + QLatin1Char(' ') + seconds_string;
            }
        }
        else
        {
            return hours_string + QLatin1Char(' ') +
                   minutes_string + QLatin1Char(' ') +
                   seconds_string;
        }
    }
    else
    {
        QString days_string = tr("%n days", "", static_cast<int>(days));

        return days_string + QLatin1Char(' ') +
               hours_string + QLatin1Char(' ') +
               minutes_string + QLatin1Char(' ') +
               seconds_string;
    }
}

//--------------------------------------------------------------------------------------------------
// static
QString RouterManagerWindow::sizeToString(int64_t size)
{
    static const int64_t kKB = 1024LL;
    static const int64_t kMB = kKB * 1024LL;
    static const int64_t kGB = kMB * 1024LL;
    static const int64_t kTB = kGB * 1024LL;

    QString units;
    int64_t divider;

    if (size >= kTB)
    {
        units = tr("TB");
        divider = kTB;
    }
    else if (size >= kGB)
    {
        units = tr("GB");
        divider = kGB;
    }
    else if (size >= kMB)
    {
        units = tr("MB");
        divider = kMB;
    }
    else if (size >= kKB)
    {
        units = tr("kB");
        divider = kKB;
    }
    else
    {
        units = tr("B");
        divider = 1;
    }

    return QString("%1 %2")
        .arg(static_cast<double>(size) / static_cast<double>(divider), 0, 'g', 4)
        .arg(units);
}

} // namespace client

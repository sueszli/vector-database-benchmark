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

#include "console/fast_connect_dialog.h"

#include "base/logging.h"
#include "base/net/address.h"
#include "build/build_config.h"
#include "client/config_factory.h"
#include "client/router_config_storage.h"
#include "client/ui/client_settings.h"
#include "client/ui/desktop/desktop_config_dialog.h"
#include "client/ui/desktop/qt_desktop_window.h"
#include "client/ui/file_transfer/qt_file_manager_window.h"
#include "client/ui/sys_info/qt_system_info_window.h"
#include "client/ui/text_chat/qt_text_chat_window.h"
#include "common/desktop_session_constants.h"
#include "common/ui/session_type.h"
#include "console/application.h"
#include "console/computer_factory.h"

#include <QMessageBox>
#include <QTimer>

namespace console {

//--------------------------------------------------------------------------------------------------
FastConnectDialog::FastConnectDialog(QWidget* parent,
                                     const QString& address_book_guid,
                                     const proto::address_book::ComputerGroupConfig& default_config,
                                     const std::optional<client::RouterConfig>& router_config)
    : QDialog(parent),
      address_book_guid_(address_book_guid),
      default_config_(default_config),
      router_config_(router_config)
{
    LOG(LS_INFO) << "Ctor";

    ui.setupUi(this);

    QPushButton* cancel_button = ui.button_box->button(QDialogButtonBox::StandardButton::Cancel);
    if (cancel_button)
        cancel_button->setText(tr("Cancel"));

    readState();

    if (!default_config_.session_config().has_desktop_manage())
    {
        ComputerFactory::setDefaultDesktopManageConfig(
            default_config_.mutable_session_config()->mutable_desktop_manage());
    }

    if (!default_config_.session_config().has_desktop_view())
    {
        ComputerFactory::setDefaultDesktopViewConfig(
            default_config_.mutable_session_config()->mutable_desktop_view());
    }

    QComboBox* combo_address = ui.combo_address;

    combo_address->addItems(state_.history);
    combo_address->setCurrentIndex(0);

    auto add_session = [this](const QString& icon, proto::SessionType session_type)
    {
        ui.combo_session_type->addItem(QIcon(icon),
                                       common::sessionTypeToLocalizedString(session_type),
                                       QVariant(session_type));
    };

    add_session(":/img/monitor-keyboard.png", proto::SESSION_TYPE_DESKTOP_MANAGE);
    add_session(":/img/monitor.png", proto::SESSION_TYPE_DESKTOP_VIEW);
    add_session(":/img/folder-stand.png", proto::SESSION_TYPE_FILE_TRANSFER);
    add_session(":/img/computer_info.png", proto::SESSION_TYPE_SYSTEM_INFO);
    add_session(":/img/text-chat.png", proto::SESSION_TYPE_TEXT_CHAT);

    int current_session_type = ui.combo_session_type->findData(QVariant(state_.session_type));
    if (current_session_type != -1)
    {
        ui.combo_session_type->setCurrentIndex(current_session_type);
        sessionTypeChanged(current_session_type);
    }

    connect(ui.button_clear, &QPushButton::clicked, this, [this]()
    {
        QMessageBox message_box(QMessageBox::Question,
                                tr("Confirmation"),
                                tr("The list of entered addresses will be cleared. Continue?"),
                                QMessageBox::Yes | QMessageBox::No,
                                this);
        message_box.button(QMessageBox::Yes)->setText(tr("Yes"));
        message_box.button(QMessageBox::No)->setText(tr("No"));

        if (message_box.exec() == QMessageBox::Yes)
        {
            ui.combo_address->clear();
            state_.history.clear();
            writeState();
        }
    });

    connect(ui.combo_session_type, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FastConnectDialog::sessionTypeChanged);

    connect(ui.button_session_config, &QPushButton::clicked,
            this, &FastConnectDialog::sessionConfigButtonPressed);

    connect(ui.checkbox_use_session_params, &QCheckBox::toggled, this, [this]()
    {
        sessionTypeChanged(ui.combo_session_type->currentIndex());
    });

    connect(ui.button_box, &QDialogButtonBox::clicked, this, &FastConnectDialog::onButtonBoxClicked);

    combo_address->setFocus();

    QTimer::singleShot(0, this, [this]()
    {
        setFixedHeight(sizeHint().height());
        adjustSize();
    });
}

//--------------------------------------------------------------------------------------------------
FastConnectDialog::~FastConnectDialog()
{
    LOG(LS_INFO) << "Dtor";
    writeState();
}

//--------------------------------------------------------------------------------------------------
void FastConnectDialog::sessionTypeChanged(int item_index)
{
    state_.session_type = static_cast<proto::SessionType>(
        ui.combo_session_type->itemData(item_index).toInt());

    LOG(LS_INFO) << "[ACTION] Session type changed: "
                 << common::sessionTypeToString(state_.session_type);

    if (ui.checkbox_use_session_params->isChecked())
    {
        ui.button_session_config->setEnabled(false);
        return;
    }

    switch (state_.session_type)
    {
        case proto::SESSION_TYPE_DESKTOP_MANAGE:
        case proto::SESSION_TYPE_DESKTOP_VIEW:
            ui.button_session_config->setEnabled(true);
            break;

        default:
            ui.button_session_config->setEnabled(false);
            break;
    }
}

//--------------------------------------------------------------------------------------------------
void FastConnectDialog::sessionConfigButtonPressed()
{
    LOG(LS_INFO) << "[ACTION] Session config button pressed";

    proto::SessionType session_type = static_cast<proto::SessionType>(
        ui.combo_session_type->currentData().toInt());

    switch (session_type)
    {
        case proto::SESSION_TYPE_DESKTOP_MANAGE:
        {
            client::DesktopConfigDialog dialog(session_type,
                                               state_.desktop_manage_config,
                                               common::kSupportedVideoEncodings,
                                               this);

            if (dialog.exec() == client::DesktopConfigDialog::Accepted)
                state_.desktop_manage_config = dialog.config();
        }
        break;

        case proto::SESSION_TYPE_DESKTOP_VIEW:
        {
            client::DesktopConfigDialog dialog(session_type,
                                               state_.desktop_view_config,
                                               common::kSupportedVideoEncodings,
                                               this);

            if (dialog.exec() == client::DesktopConfigDialog::Accepted)
                state_.desktop_view_config = dialog.config();
        }
        break;

        default:
            break;
    }
}

//--------------------------------------------------------------------------------------------------
void FastConnectDialog::onButtonBoxClicked(QAbstractButton* button)
{
    if (ui.button_box->standardButton(button) == QDialogButtonBox::Cancel)
    {
        LOG(LS_INFO) << "[ACTION] Rejected by user";
        reject();
        close();
        return;
    }

    LOG(LS_INFO) << "[ACTION] Accepted by user";

    QComboBox* combo_address = ui.combo_address;
    QString current_address = combo_address->currentText();
    bool host_id_entered = true;

    for (int i = 0; i < current_address.length(); ++i)
    {
        if (!current_address[i].isDigit())
        {
            host_id_entered = false;
            break;
        }
    }

    if (host_id_entered && !router_config_.has_value())
    {
        LOG(LS_ERROR) << "Router not configured";
        QMessageBox::warning(this,
                             tr("Warning"),
                             tr("Connection by ID is specified but the router is not configured. "
                                "Check the parameters of the router in the properties of the "
                                "address book."),
                             QMessageBox::Ok);
        return;
    }

    client::Config client_config;

    if (!host_id_entered)
    {
        LOG(LS_INFO) << "Direct connection selected";

        base::Address address = base::Address::fromString(
            current_address.toStdU16String(), DEFAULT_HOST_TCP_PORT);

        if (!address.isValid())
        {
            LOG(LS_ERROR) << "Invalid address entered";
            QMessageBox::warning(this,
                                 tr("Warning"),
                                 tr("An invalid computer address was entered."),
                                 QMessageBox::Ok);
            combo_address->setFocus();
            return;
        }

        client_config.address_or_id = address.host();
        client_config.port = address.port();
    }
    else
    {
        LOG(LS_INFO) << "Relay connection selected";
        client_config.address_or_id = current_address.toStdU16String();
    }

    client_config.session_type = static_cast<proto::SessionType>(
        ui.combo_session_type->currentData().toInt());

    if (ui.checkbox_use_creds->isChecked())
    {
        client_config.username = base::utf16FromUtf8(default_config_.username());
        client_config.password = base::utf16FromUtf8(default_config_.password());
    }

    client_config.router_config = router_config_;

    int current_index = combo_address->findText(current_address);
    if (current_index != -1)
        combo_address->removeItem(current_index);

    combo_address->insertItem(0, current_address);
    combo_address->setCurrentIndex(0);

    state_.history.clear();
    for (int i = 0; i < std::min(combo_address->count(), 15); ++i)
        state_.history.append(combo_address->itemText(i));

    client::SessionWindow* session_window = nullptr;

    switch (client_config.session_type)
    {
        case proto::SESSION_TYPE_DESKTOP_MANAGE:
        {
            proto::DesktopConfig desktop_config;

            if (ui.checkbox_use_session_params->isChecked())
                desktop_config = default_config_.session_config().desktop_manage();
            else
                desktop_config = state_.desktop_manage_config;

            session_window = new client::QtDesktopWindow(
                state_.session_type, desktop_config);
        }
        break;

        case proto::SESSION_TYPE_DESKTOP_VIEW:
        {
            proto::DesktopConfig desktop_config;

            if (ui.checkbox_use_session_params->isChecked())
                desktop_config = default_config_.session_config().desktop_view();
            else
                desktop_config = state_.desktop_view_config;

            session_window = new client::QtDesktopWindow(
                state_.session_type, desktop_config);
        }
        break;

        case proto::SESSION_TYPE_FILE_TRANSFER:
            session_window = new client::QtFileManagerWindow();
            break;

        case proto::SESSION_TYPE_SYSTEM_INFO:
            session_window = new client::QtSystemInfoWindow();
            break;

        case proto::SESSION_TYPE_TEXT_CHAT:
            session_window = new client::QtTextChatWindow();
            break;

        default:
            NOTREACHED();
            break;
    }

    if (!session_window)
        return;

    session_window->setAttribute(Qt::WA_DeleteOnClose);
    if (!session_window->connectToHost(client_config))
    {
        LOG(LS_ERROR) << "Unable to connect";
        session_window->close();
    }
    else
    {
        LOG(LS_INFO) << "Close fast connect dialog";
        accept();
        close();
    }
}

//--------------------------------------------------------------------------------------------------
void FastConnectDialog::readState()
{
    QDataStream stream(Application::instance()->settings().fastConnectConfig(address_book_guid_));
    stream.setVersion(QDataStream::Qt_5_12);

    int session_type = 0;
    QByteArray desktop_manage_config;
    QByteArray desktop_view_config;
    bool creds_from_address_book = false;
    bool session_params_from_address_book = false;

    stream >> state_.history >> session_type >> desktop_manage_config >> desktop_view_config
           >> creds_from_address_book >> session_params_from_address_book;

    if (session_type != 0)
        state_.session_type = static_cast<proto::SessionType>(session_type);
    else
        state_.session_type = proto::SESSION_TYPE_DESKTOP_MANAGE;

    if (!desktop_manage_config.isEmpty())
    {
        state_.desktop_manage_config.ParseFromArray(
            desktop_manage_config.data(), desktop_manage_config.size());
    }
    else
    {
        state_.desktop_manage_config = client::ConfigFactory::defaultDesktopManageConfig();
    }

    if (!desktop_view_config.isEmpty())
    {
        state_.desktop_view_config.ParseFromArray(
            desktop_view_config.data(), desktop_view_config.size());
    }
    else
    {
        state_.desktop_view_config = client::ConfigFactory::defaultDesktopViewConfig();
    }

    ui.checkbox_use_creds->setChecked(creds_from_address_book);
    ui.checkbox_use_session_params->setChecked(session_params_from_address_book);
}

//--------------------------------------------------------------------------------------------------
void FastConnectDialog::writeState()
{
    QByteArray buffer;

    {
        int session_type = static_cast<int>(state_.session_type);
        QByteArray desktop_manage_config =
            QByteArray::fromStdString(state_.desktop_manage_config.SerializeAsString());
        QByteArray desktop_view_config =
            QByteArray::fromStdString(state_.desktop_view_config.SerializeAsString());
        bool creds_from_address_book = ui.checkbox_use_creds->isChecked();
        bool session_params_from_address_book = ui.checkbox_use_session_params->isChecked();

        QDataStream stream(&buffer, QIODevice::WriteOnly);
        stream.setVersion(QDataStream::Qt_5_12);

        stream << state_.history << session_type << desktop_manage_config << desktop_view_config
               << creds_from_address_book << session_params_from_address_book;
    }

    Application::instance()->settings().setFastConnectConfig(address_book_guid_, buffer);
}

} // namespace console

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

#include "client/ui/update_settings_dialog.h"

#include "base/logging.h"
#include "build/build_config.h"
#include "client/ui/client_settings.h"

#include <QPushButton>

namespace client {

//--------------------------------------------------------------------------------------------------
UpdateSettingsDialog::UpdateSettingsDialog(QWidget* parent)
    : QDialog(parent)
{
    LOG(LS_INFO) << "Ctor";
    ui.setupUi(this);

    QPushButton* cancel_button = ui.button_box->button(QDialogButtonBox::StandardButton::Cancel);
    if (cancel_button)
        cancel_button->setText(tr("Cancel"));

    ClientSettings settings;

    ui.checkbox_check_updates->setChecked(settings.checkUpdates());
    ui.edit_server->setText(settings.updateServer());

    if (settings.updateServer() == DEFAULT_UPDATE_SERVER)
    {
        ui.checkbox_custom_server->setChecked(false);
        ui.edit_server->setEnabled(false);
    }
    else
    {
        ui.checkbox_custom_server->setChecked(true);
        ui.edit_server->setEnabled(true);
    }

    connect(ui.checkbox_custom_server, &QCheckBox::toggled, this, [this](bool checked)
    {
        ui.edit_server->setEnabled(checked);

        if (!checked)
            ui.edit_server->setText(DEFAULT_UPDATE_SERVER);
    });

    connect(ui.button_box, &QDialogButtonBox::clicked, this, [this](QAbstractButton* button)
    {
        if (ui.button_box->standardButton(button) == QDialogButtonBox::Ok)
        {
            ClientSettings settings;
            settings.setCheckUpdates(ui.checkbox_check_updates->isChecked());
            settings.setUpdateServer(ui.edit_server->text());
        }

        close();
    });
}

//--------------------------------------------------------------------------------------------------
UpdateSettingsDialog::~UpdateSettingsDialog()
{
    LOG(LS_INFO) << "Dtor";
}

} // namespace client

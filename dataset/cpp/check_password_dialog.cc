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

#include "host/ui/check_password_dialog.h"

#include "host/system_settings.h"
#include "qt_base/qt_logging.h"

#include <QAbstractButton>
#include <QMessageBox>
#include <QPushButton>
#include <QTimer>

namespace host {

//--------------------------------------------------------------------------------------------------
CheckPasswordDialog::CheckPasswordDialog(QWidget* parent)
    : QDialog(parent)
{
    LOG(LS_INFO) << "Ctor";
    ui.setupUi(this);

    QPushButton* cancel_button = ui.button_box->button(QDialogButtonBox::StandardButton::Cancel);
    if (cancel_button)
        cancel_button->setText(tr("Cancel"));

    connect(ui.button_box, &QDialogButtonBox::clicked,
            this, &CheckPasswordDialog::onButtonBoxClicked);

    QTimer::singleShot(0, this, [=]()
    {
        setFixedHeight(sizeHint().height());
    });
}

//--------------------------------------------------------------------------------------------------
CheckPasswordDialog::~CheckPasswordDialog()
{
    LOG(LS_INFO) << "Dtor";
}

//--------------------------------------------------------------------------------------------------
void CheckPasswordDialog::onButtonBoxClicked(QAbstractButton* button)
{
    QDialogButtonBox::StandardButton standard_button = ui.button_box->standardButton(button);
    if (standard_button == QDialogButtonBox::Ok)
    {
        LOG(LS_INFO) << "[ACTION] Accepted by user";

        QString password = ui.edit_pass->text();

        if (!SystemSettings::isValidPassword(password.toStdString()))
        {
            LOG(LS_INFO) << "Invalid password entered";

            QMessageBox::warning(this,
                                 tr("Warning"),
                                 tr("You entered an incorrect password."),
                                 QMessageBox::Ok);
            ui.edit_pass->selectAll();
            ui.edit_pass->setFocus();
            return;
        }

        accept();
    }
    else
    {
        LOG(LS_INFO) << "[ACTION] Rejected by user";
        reject();
    }

    close();
}

} // namespace host

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

#include "console/computer_group_dialog.h"

#include "base/logging.h"
#include "base/crypto/secure_memory.h"
#include "console/computer_group_dialog_desktop.h"
#include "console/computer_group_dialog_general.h"
#include "console/computer_group_dialog_parent.h"

#include <QAbstractButton>
#include <QDateTime>
#include <QMessageBox>

namespace console {

namespace {

constexpr int kMaxNameLength = 64;
constexpr int kMinNameLength = 1;
constexpr int kMaxCommentLength = 2048;

enum ItemType
{
    ITEM_TYPE_PARENT,
    ITEM_TYPE_GENERAL,
    ITEM_TYPE_DESKTOP_MANAGE,
    ITEM_TYPE_DESKTOP_VIEW
};

} // namespace

//--------------------------------------------------------------------------------------------------
ComputerGroupDialog::ComputerGroupDialog(QWidget* parent,
                                         Mode mode,
                                         const QString& parent_name,
                                         proto::address_book::ComputerGroup* computer_group)
    : QDialog(parent),
      mode_(mode),
      computer_group_(computer_group)
{
    LOG(LS_INFO) << "Ctor";
    ui.setupUi(this);

    QPushButton* cancel_button = ui.button_box->button(QDialogButtonBox::StandardButton::Cancel);
    if (cancel_button)
        cancel_button->setText(tr("Cancel"));

    restoreGeometry(settings_.computerGroupDialogGeometry());

    connect(ui.button_box, &QDialogButtonBox::clicked,
            this, &ComputerGroupDialog::buttonBoxClicked);

    connect(ui.tree_category, &QTreeWidget::currentItemChanged,
            this, &ComputerGroupDialog::onTabChanged);

    ui.edit_parent_name->setText(parent_name);
    ui.edit_name->setText(QString::fromStdString(computer_group_->name()));
    ui.edit_comment->setPlainText(QString::fromStdString(computer_group->comment()));

    QTreeWidgetItem* general_item = new QTreeWidgetItem(ITEM_TYPE_GENERAL);
    general_item->setIcon(0, QIcon(":/img/computer.png"));
    general_item->setText(0, tr("General"));

    QTreeWidgetItem* sessions_item = new QTreeWidgetItem(ITEM_TYPE_PARENT);
    sessions_item->setIcon(0, QIcon(":/img/settings.png"));
    sessions_item->setText(0, tr("Sessions"));

    ui.tree_category->addTopLevelItem(general_item);
    ui.tree_category->addTopLevelItem(sessions_item);

    QTreeWidgetItem* desktop_manage_item = new QTreeWidgetItem(ITEM_TYPE_DESKTOP_MANAGE);
    desktop_manage_item->setIcon(0, QIcon(":/img/monitor-keyboard.png"));
    desktop_manage_item->setText(0, tr("Manage"));

    QTreeWidgetItem* desktop_view_item = new QTreeWidgetItem(ITEM_TYPE_DESKTOP_VIEW);
    desktop_view_item->setIcon(0, QIcon(":/img/monitor.png"));
    desktop_view_item->setText(0, tr("View"));

    sessions_item->addChild(desktop_manage_item);
    sessions_item->addChild(desktop_view_item);

    ComputerGroupDialogParent* parent_tab =
        new ComputerGroupDialogParent(ITEM_TYPE_PARENT, false, ui.widget);
    ComputerGroupDialogGeneral* general_tab =
        new ComputerGroupDialogGeneral(ITEM_TYPE_GENERAL, false, ui.widget);
    ComputerGroupDialogDesktop* desktop_manage_tab =
        new ComputerGroupDialogDesktop(ITEM_TYPE_DESKTOP_MANAGE, false, ui.widget);
    ComputerGroupDialogDesktop* desktop_view_tab =
        new ComputerGroupDialogDesktop(ITEM_TYPE_DESKTOP_VIEW, false, ui.widget);

    general_tab->restoreSettings(computer_group_->config());
    desktop_manage_tab->restoreSettings(
        proto::SESSION_TYPE_DESKTOP_MANAGE, computer_group_->config());
    desktop_view_tab->restoreSettings(
        proto::SESSION_TYPE_DESKTOP_VIEW, computer_group_->config());

    tabs_.append(general_tab);
    tabs_.append(desktop_manage_tab);
    tabs_.append(desktop_view_tab);
    tabs_.append(parent_tab);

    QSize min_size;

    for (auto it = tabs_.begin(); it != tabs_.end(); ++it)
    {
        QWidget* tab = *it;
        min_size.setWidth(std::max(tab->sizeHint().width(), min_size.width()));
        min_size.setHeight(std::max(tab->minimumSizeHint().height(), min_size.height()));
    }

    ui.widget->setMinimumSize(min_size);
    ui.widget->installEventFilter(this);

    ui.tree_category->setCurrentItem(general_item);
    ui.tree_category->expandAll();
}

//--------------------------------------------------------------------------------------------------
ComputerGroupDialog::~ComputerGroupDialog()
{
    LOG(LS_INFO) << "Dtor";
}

//--------------------------------------------------------------------------------------------------
void ComputerGroupDialog::closeEvent(QCloseEvent* event)
{
    LOG(LS_INFO) << "Close event";
    settings_.setComputerGroupDialogGeometry(saveGeometry());
    QDialog::closeEvent(event);
}

//--------------------------------------------------------------------------------------------------
bool ComputerGroupDialog::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == ui.widget && event->type() == QEvent::Resize)
    {
        for (auto it = tabs_.begin(); it != tabs_.end(); ++it)
        {
            QWidget* tab = *it;
            tab->resize(ui.widget->size());
        }
    }

    return QDialog::eventFilter(watched, event);
}

//--------------------------------------------------------------------------------------------------
void ComputerGroupDialog::keyPressEvent(QKeyEvent* event)
{
    if ((event->key() == Qt::Key_Return) && (event->modifiers() & Qt::ControlModifier))
    {
        if (saveChanges())
        {
            accept();
            close();
        }
    }

    QDialog::keyPressEvent(event);
}

//--------------------------------------------------------------------------------------------------
void ComputerGroupDialog::buttonBoxClicked(QAbstractButton* button)
{
    if (ui.button_box->standardButton(button) == QDialogButtonBox::Ok)
    {
        LOG(LS_INFO) << "[ACTION] Accepted by user";

        if (!saveChanges())
            return;

        accept();
    }
    else
    {
        LOG(LS_INFO) << "[ACTION] Rejected by user";
        reject();
    }

    close();
}

//--------------------------------------------------------------------------------------------------
void ComputerGroupDialog::onTabChanged(QTreeWidgetItem* current)
{
    if (current)
        showTab(current->type());
}

//--------------------------------------------------------------------------------------------------
void ComputerGroupDialog::showError(const QString& message)
{
    QMessageBox(QMessageBox::Warning, tr("Warning"), message, QMessageBox::Ok, this).exec();
}

//--------------------------------------------------------------------------------------------------
void ComputerGroupDialog::showTab(int type)
{
    for (auto it = tabs_.begin(); it != tabs_.end(); ++it)
    {
        QWidget* tab = *it;
        if (static_cast<ComputerGroupDialogTab*>(tab)->type() == type)
            tab->show();
        else
            tab->hide();
    }
}

//--------------------------------------------------------------------------------------------------
bool ComputerGroupDialog::saveChanges()
{
    QString name = ui.edit_name->text();
    if (name.length() > kMaxNameLength)
    {
        LOG(LS_ERROR) << "Too long name: " << name.length();
        showError(tr("Too long name. The maximum length of the name is %n characters.",
                     "", kMaxNameLength));
        ui.edit_name->setFocus();
        ui.edit_name->selectAll();
        return false;
    }
    else if (name.length() < kMinNameLength)
    {
        LOG(LS_ERROR) << "Name can not be empty";
        showError(tr("Name can not be empty."));
        ui.edit_name->setFocus();
        return false;
    }

    QString comment = ui.edit_comment->toPlainText();
    if (comment.length() > kMaxCommentLength)
    {
        LOG(LS_ERROR) << "Too long comment: " << comment.length();
        showError(tr("Too long comment. The maximum length of the comment is %n characters.",
                     "", kMaxCommentLength));
        ui.edit_comment->setFocus();
        ui.edit_comment->selectAll();
        return false;
    }

    for (auto it = tabs_.begin(); it != tabs_.end(); ++it)
    {
        QWidget* tab = *it;
        int type = static_cast<ComputerGroupDialogTab*>(tab)->type();

        if (type == ITEM_TYPE_GENERAL)
        {
            ComputerGroupDialogGeneral* general_tab =
                static_cast<ComputerGroupDialogGeneral*>(tab);

            if (!general_tab->saveSettings(computer_group_->mutable_config()))
                return false;
        }
        else if (type == ITEM_TYPE_DESKTOP_MANAGE)
        {
            ComputerGroupDialogDesktop* desktop_tab =
                static_cast<ComputerGroupDialogDesktop*>(tab);

            desktop_tab->saveSettings(proto::SESSION_TYPE_DESKTOP_MANAGE,
                computer_group_->mutable_config());
        }
        else if (type == ITEM_TYPE_DESKTOP_VIEW)
        {
            ComputerGroupDialogDesktop* desktop_tab =
                static_cast<ComputerGroupDialogDesktop*>(tab);

            desktop_tab->saveSettings(proto::SESSION_TYPE_DESKTOP_VIEW,
                computer_group_->mutable_config());
        }
    }

    int64_t current_time = QDateTime::currentSecsSinceEpoch();

    if (mode_ == CreateComputerGroup)
        computer_group_->set_create_time(current_time);

    computer_group_->set_modify_time(current_time);
    computer_group_->set_name(name.toStdString());
    computer_group_->set_comment(comment.toStdString());

    return true;
}

} // namespace console

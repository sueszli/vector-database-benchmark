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

#include "client/ui/sys_info/sys_info_widget_licenses.h"

#include "base/macros_magic.h"
#include "common/system_info_constants.h"

#include <QMenu>

namespace client {

namespace {

class Item : public QTreeWidgetItem
{
public:
    Item(const char* icon_path, const QString& text, const QList<QTreeWidgetItem*>& childs)
    {
        QIcon icon(icon_path);

        setIcon(0, icon);
        setText(0, text);

        for (const auto& child : childs)
        {
            child->setIcon(0, icon);

            for (int i = 0; i < child->childCount(); ++i)
            {
                QTreeWidgetItem* item = child->child(i);
                if (item)
                    item->setIcon(0, icon);
            }
        }

        addChildren(childs);
    }

private:
    DISALLOW_COPY_AND_ASSIGN(Item);
};

//--------------------------------------------------------------------------------------------------
QTreeWidgetItem* mk(const QString& param, const QString& value)
{
    QTreeWidgetItem* item = new QTreeWidgetItem();

    item->setText(0, param);
    item->setText(1, value);

    return item;
}

} // namespace

//--------------------------------------------------------------------------------------------------
SysInfoWidgetLicenses::SysInfoWidgetLicenses(QWidget* parent)
    : SysInfoWidget(parent)
{
    ui.setupUi(this);

    connect(ui.action_copy_row, &QAction::triggered, this, [this]()
    {
        copyRow(ui.tree->currentItem());
    });

    connect(ui.action_copy_name, &QAction::triggered, this, [this]()
    {
        copyColumn(ui.tree->currentItem(), 0);
    });

    connect(ui.action_copy_value, &QAction::triggered, this, [this]()
    {
        copyColumn(ui.tree->currentItem(), 1);
    });

    connect(ui.tree, &QTreeWidget::customContextMenuRequested,
            this, &SysInfoWidgetLicenses::onContextMenu);

    connect(ui.tree, &QTreeWidget::itemDoubleClicked,
            this, [this](QTreeWidgetItem* item, int /* column */)
    {
        copyRow(item);
    });
}

//--------------------------------------------------------------------------------------------------
SysInfoWidgetLicenses::~SysInfoWidgetLicenses() = default;

//--------------------------------------------------------------------------------------------------
std::string SysInfoWidgetLicenses::category() const
{
    return common::kSystemInfo_Licenses;
}

//--------------------------------------------------------------------------------------------------
void SysInfoWidgetLicenses::setSystemInfo(const proto::system_info::SystemInfo& system_info)
{
    ui.tree->clear();

    if (!system_info.has_licenses())
    {
        ui.tree->setEnabled(false);
        return;
    }

    const proto::system_info::Licenses& licenses = system_info.licenses();

    for (int i = 0; i < licenses.license_size(); ++i)
    {
        const proto::system_info::Licenses::License& license = licenses.license(i);
        QList<QTreeWidgetItem*> group;

        for (int j = 0; j < license.field_size(); ++j)
        {
            const proto::system_info::Licenses::License::Field& field = license.field(j);

            QString title;
            switch (field.type())
            {
                case proto::system_info::Licenses::License::Field::TYPE_LICENSE_TYPE:
                    title = tr("License Type");
                    break;

                case proto::system_info::Licenses::License::Field::TYPE_LICENSE_VERSION:
                    title = tr("License Version");
                    break;

                case proto::system_info::Licenses::License::Field::TYPE_ORGANIZATION:
                    title = tr("Organization");
                    break;

                case proto::system_info::Licenses::License::Field::TYPE_OWNER:
                    title = tr("Owner");
                    break;

                case proto::system_info::Licenses::License::Field::TYPE_PRODUCT_ID:
                    title = tr("Product ID");
                    break;

                case proto::system_info::Licenses::License::Field::TYPE_PRODUCT_KEY:
                    title = tr("Product Key");
                    break;

                default:
                    title = tr("Unknown Field");
                    break;
            }

            if (!field.value().empty())
                group << mk(title, QString::fromStdString(field.value()));
        }

        if (!group.isEmpty())
        {
            ui.tree->addTopLevelItem(
                new Item(":/img/license-key.png", QString::fromStdString(license.product_name()), group));
        }
    }

    for (int i = 0; i < ui.tree->topLevelItemCount(); ++i)
        ui.tree->topLevelItem(i)->setExpanded(true);

    ui.tree->resizeColumnToContents(0);
}

//--------------------------------------------------------------------------------------------------
QTreeWidget* SysInfoWidgetLicenses::treeWidget()
{
    return ui.tree;
}

//--------------------------------------------------------------------------------------------------
void SysInfoWidgetLicenses::onContextMenu(const QPoint& point)
{
    QTreeWidgetItem* current_item = ui.tree->itemAt(point);
    if (!current_item)
        return;

    ui.tree->setCurrentItem(current_item);

    QMenu menu;
    menu.addAction(ui.action_copy_row);
    menu.addAction(ui.action_copy_name);
    menu.addAction(ui.action_copy_value);

    menu.exec(ui.tree->viewport()->mapToGlobal(point));
}

} // namespace client

/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "QDisplayPropertyWidget.h"
#include "ModifyObject.h"
#include "DataWidget.h"
#include "QDisplayDataWidget.h"
#include "QDisplayLinkWidget.h"
#include "QDataDescriptionWidget.h"
#include "QTabulationModifyObject.h"
#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::Message ;

namespace sofa::gui::qt
{

QDisplayTreeItemWidget::QDisplayTreeItemWidget(QWidget* parent, QTreeWidgetItem* item) : QWidget(parent)
    , treeWidgetItem(item)
{

}

QDisplayTreeItemWidget::~QDisplayTreeItemWidget()
{

}

void QDisplayTreeItemWidget::updateDirtyWidget()
{
    const QObjectList& parentList = children();
    for(QObjectList::const_iterator itParent = parentList.begin(); itParent != parentList.end(); ++itParent)
    {
        QWidget* parentWidget = dynamic_cast<QWidget*>(*itParent);
        if(parentWidget)
        {
            const QObjectList& childList = parentWidget->children();
            for(QObjectList::const_iterator itChild = childList.begin(); itChild != childList.end(); ++itChild)
            {
                QWidget* childWidget = dynamic_cast<QWidget*>(*itChild);
                if(childWidget)
                {
                    childWidget->adjustSize();
                    childWidget->setHidden(false);
                }
            }
            parentWidget->adjustSize();
            parentWidget->setHidden(false);
        }
    }

    adjustSize();
    setHidden(false);
    treeWidgetItem->setHidden(false);
}

QDisplayPropertyWidget::QDisplayPropertyWidget(const ModifyObjectFlags& modifyFlags, QWidget* parent) : QTreeWidget(parent)
    , objects()
    , pinIcon()
    , modifyObjectFlags(modifyFlags)
{
    modifyObjectFlags.PROPERTY_WIDGET_FLAG = true;

    std::string filename = "textures/pin.png";
    sofa::helper::system::DataRepository.findFile(filename);
    pinIcon = QIcon(QString::fromStdString(filename));

    setColumnCount(2);
    //setIndentation(10);

    headerItem()->setText(0, QString("Property"));
    headerItem()->setText(1, QString("Value"));
    setSelectionMode(QAbstractItemView::NoSelection);
    //setSelectionBehavior(QAbstractItemView::SelectItems);
    setDragEnabled(false);
    setAcceptDrops(false);
    setDropIndicatorShown(false);
    //setDragDropMode(QAbstractItemView::InternalMove);
    setIndentation(0);

    setFocusPolicy(Qt::NoFocus);
    setAutoFillBackground(false);
}

QDisplayPropertyWidget::~QDisplayPropertyWidget()
{

}

void QDisplayPropertyWidget::addComponent(const QString& component, core::objectmodel::Base* base, QTreeWidgetItem* listItem, bool clear)
{
    if(clear)
        this->clear();

    // return now if the component to add is empty or if it is already in the tree
    if(component.isEmpty() || !base || findComponent(component))
        return;

    const sofa::core::objectmodel::Base::VecData& fields = base->getDataFields();

    // return now if the component has no field
    if(fields.empty())
        return;

    // finally, add the group
    QTreeWidgetItem *componentItem = new QTreeWidgetItem(this);
    QFont *font = new QFont();
    font->setBold(true);
    componentItem->setFont(0, *font);
    componentItem->setText(0, component);
    QPushButton* pin = new QPushButton(this);
    pin->setFixedSize(QSize(18, 18));
    pin->setCheckable(true);
    pin->setIcon(pinIcon);
    setItemWidget(componentItem, 1, pin);
    componentItem->setExpanded(true);
    const QBrush *backgroundBrush = new QBrush(QColor(20, 20, 20));
    const QBrush *foregroundBrush = new QBrush(QColor(255, 255, 255));
    componentItem->setBackground(0, *backgroundBrush);
    componentItem->setForeground(0, *foregroundBrush);
    componentItem->setTextAlignment(0, Qt::AlignLeft);
    componentItem->setBackground(1, *backgroundBrush);
    componentItem->setForeground(1, *foregroundBrush);
    componentItem->setTextAlignment(1, Qt::AlignRight);

    objects[componentItem] = std::pair<core::objectmodel::Base*, QTreeWidgetItem*>(base, listItem);

    // add data
    for(sofa::core::objectmodel::Base::VecData::const_iterator it = fields.begin(); it != fields.end(); ++it)
    {
        core::objectmodel::BaseData *data = *it;

        // ignore unnamed data
        if(data->getName().empty())
            continue;

        // for each data of the current object we determine where it belongs
        QString group = QString::fromStdString(data->getGroup());

        // use the default group if data does not belong to any group
        if(group.isEmpty())
            group = DefaultDataGroup();

        // finally, add the data
        addData(component, group, data);
    }


    // add links
    const sofa::core::objectmodel::Base::VecLink& links = base->getLinks();
    for(sofa::core::objectmodel::Base::VecLink::const_iterator it = links.begin(); it != links.end(); ++it)
    {
        core::objectmodel::BaseLink *link = *it;

        // ignore unnamed link
        if(link->getName().empty())
            continue;

        if(!link->storePath() && 0 == link->getSize())
            continue;

        // use the default link group
        QString group = DefaultLinkGroup();

        // finally, add the data
        addLink(component, group, link);
    }

    // add info
    {
        // use the default info group
        const QString group = DefaultInfoGroup();

        setDescription(component, group, base);
    }

    const bool notImplementedYet = false;
    if(!notImplementedYet)
    // add console
    {
        // use the default info group
        const QString group = DefaultLogGroup();

        setConsoleOutput(component, group, base);
    }
}

void QDisplayPropertyWidget::addGroup(const QString& component, const QString& group)
{
    // return now if the component does not exist
    QTreeWidgetItem *componentItem = nullptr;
    componentItem = findComponent(component);
    if(!componentItem)
        return;

    // return now if the group component already exist
    QTreeWidgetItem *groupItem = nullptr;
    groupItem = findGroup(component, group);
    if(groupItem)
        return;

    // assign the default label if group is an empty string
    QString groupLabel = group;
    if(group.isEmpty())
        groupLabel = DefaultDataGroup();

    // finally, add the group
    groupItem = new QTreeWidgetItem(componentItem);
    QFont *font = new QFont();
    font->setBold(true);
    groupItem->setFont(0, *font);
    groupItem->setText(0, groupLabel);
    groupItem->setExpanded(true);
    const QBrush *backgroundBrush = new QBrush(QColor(160, 160, 160));
    const QBrush *foregroundBrush = new QBrush(QColor(255, 255, 255));
    groupItem->setBackground(0, *backgroundBrush);
    groupItem->setForeground(0, *foregroundBrush);
    groupItem->setBackground(1, *backgroundBrush);
    groupItem->setForeground(1, *foregroundBrush);
}

void QDisplayPropertyWidget::addData(const QString& component, const QString& group, sofa::core::objectmodel::BaseData *data)
{
    if(!data || !data->isDisplayed())
        return;

    addGroup(component, group);
    QTreeWidgetItem *groupItem = nullptr;
    groupItem = findGroup(component, group);

    if(!groupItem)
        return;

    QTreeWidgetItem *dataItem = new QTreeWidgetItem(groupItem);
    const QBrush *brush = nullptr;
    if(groupItem->childCount() % 2 == 0)
        brush = new QBrush(QColor(255, 255, 191));
    else
        brush = new QBrush(QColor(255, 255, 222));
    dataItem->setBackground(0, *brush);
    dataItem->setBackground(1, *brush);

    data->setDisplayed(true);

    QDisplayTreeItemWidget *widget = new QDisplayTreeItemWidget(this, dataItem);
    QHBoxLayout *layout = new QHBoxLayout(widget);

    dataItem->setText(0, QString::fromStdString(data->getName()));
    dataItem->setToolTip(0, QString::fromStdString(data->getHelp()));
    QDisplayDataWidget *displayDataWidget = new QDisplayDataWidget(widget, data, modifyObjectFlags);
    layout->addWidget(displayDataWidget);

    connect(displayDataWidget, SIGNAL(WidgetDirty(bool)), widget, SLOT(updateDirtyWidget()));

    widget->setContentsMargins(0, 0, 0, 0);
    if(widget->layout())
    {
        widget->layout()->setContentsMargins(0, 0, 0, 0);
        widget->layout()->setSpacing(0);
    }
    setItemWidget(dataItem, 1, widget);
    dataItem->setToolTip(1, QString::fromStdString(data->getHelp()));
}

void QDisplayPropertyWidget::addLink(const QString& component, const QString& group, sofa::core::objectmodel::BaseLink *link)
{
    if(!link)
        return;

    addGroup(component, group);
    QTreeWidgetItem *groupItem = nullptr;
    groupItem = findGroup(component, group);

    if(!groupItem)
        return;

    QTreeWidgetItem *linkItem = new QTreeWidgetItem(groupItem);
    const QBrush *brush = nullptr;
    if(groupItem->childCount() % 2 == 0)
        brush = new QBrush(QColor(255, 255, 191));
    else
        brush = new QBrush(QColor(255, 255, 222));
    linkItem->setBackground(0, *brush);
    linkItem->setBackground(1, *brush);

    QDisplayTreeItemWidget *widget = new QDisplayTreeItemWidget(this, linkItem);
    QHBoxLayout *layout = new QHBoxLayout(widget);

    ModifyObjectFlags linkFlags = modifyObjectFlags;
    linkFlags.READONLY_FLAG = true;

    linkItem->setText(0, QString::fromStdString(link->getName()));
    linkItem->setToolTip(0, QString::fromStdString(link->getHelp()));
    QDisplayLinkWidget *displayLinkWidget = new QDisplayLinkWidget(widget, link, linkFlags);
    layout->addWidget(displayLinkWidget);

    connect(displayLinkWidget, SIGNAL(WidgetDirty(bool)), widget, SLOT(updateDirtyWidget()));

    widget->setContentsMargins(0, 0, 0, 0);
    if(widget->layout())
    {
        widget->layout()->setContentsMargins(0, 0, 0, 0);
        widget->layout()->setSpacing(0);
    }
    setItemWidget(linkItem, 1, widget);
    linkItem->setToolTip(1, QString::fromStdString(link->getHelp()));
}

void QDisplayPropertyWidget::setDescription(const QString& component, const QString& group, sofa::core::objectmodel::Base *base)
{
    if(!base)
        return;

    addGroup(component, group);
    QTreeWidgetItem *groupItem = nullptr;
    groupItem = findGroup(component, group);

    if(!groupItem)
        return;

    const QBrush *brush = nullptr;
    QFont categoryFont;
    categoryFont.setBold(true);

    // Instance
    QTreeWidgetItem *instanceItem = new QTreeWidgetItem(groupItem);
    if(groupItem->childCount() % 2 == 0)
        brush = new QBrush(QColor(255, 255, 191));
    else
        brush = new QBrush(QColor(255, 255, 222));

    instanceItem->setBackground(0, *brush);
    instanceItem->setBackground(1, *brush);

    instanceItem->setText(0, "Instance");
    instanceItem->setFont(0, categoryFont);

    {
        addDescriptionItem(groupItem, "Name", QString::fromStdString(base->getName()));

        addDescriptionItem(groupItem, "Class", QString::fromStdString(base->getClassName()));

        const std::string namespacename = sofa::helper::NameDecoder::decodeNamespaceName(typeid(*base));
        if (!namespacename.empty())
            addDescriptionItem(groupItem, "Namespace", QString::fromStdString(namespacename));

        if (!base->getTemplateName().empty())
            addDescriptionItem(groupItem, "Template", QString::fromStdString(base->getTemplateName()));
    }

    // Class
    core::ObjectFactory::ClassEntry entry = core::ObjectFactory::getInstance()->getEntry(base->getClassName());
    if(!entry.creatorMap.empty())
    {
        QTreeWidgetItem *classItem = new QTreeWidgetItem(groupItem);
        if(groupItem->childCount() % 2 == 0)
            brush = new QBrush(QColor(255, 255, 191));
        else
            brush = new QBrush(QColor(255, 255, 222));

        classItem->setBackground(0, *brush);
        classItem->setBackground(1, *brush);

        classItem->setText(0, "Class");
        classItem->setFont(0, categoryFont);

        if(!entry.description.empty() && std::string("TODO") != entry.description)
            addDescriptionItem(groupItem, "Description", QString::fromStdString(entry.description));

        const core::ObjectFactory::CreatorMap::iterator it = entry.creatorMap.find(base->getTemplateName());
        if(entry.creatorMap.end() != it && *it->second->getTarget())
            addDescriptionItem(groupItem, "Provided by", QString(it->second->getTarget()));

        if(!entry.authors.empty() && std::string("TODO") != entry.authors)
            addDescriptionItem(groupItem, "Authors", QString::fromStdString(entry.authors));

        if(!entry.license.empty() && std::string("TODO") != entry.license)
            addDescriptionItem(groupItem, "License", QString::fromStdString(entry.license));
    }
}

void QDisplayPropertyWidget::addDescriptionItem(QTreeWidgetItem *groupItem, const QString& name, const QString& description)
{
    QTreeWidgetItem *descriptionItem = new QTreeWidgetItem(groupItem);

    const QBrush *brush = nullptr;
    if(groupItem->childCount() % 2 == 0)
        brush = new QBrush(QColor(255, 255, 191));
    else
        brush = new QBrush(QColor(255, 255, 222));

    descriptionItem->setBackground(0, *brush);
    descriptionItem->setBackground(1, *brush);

    descriptionItem->setText(0, name);

    QDisplayTreeItemWidget *widget = new QDisplayTreeItemWidget(this, descriptionItem);
    new QLabel(description, widget);
    setItemWidget(descriptionItem, 1, widget);
}

void QDisplayPropertyWidget::setConsoleOutput(const QString& component, const QString& group, sofa::core::objectmodel::Base *base)
{
    if(base==nullptr)
        return;

    const std::string warnings = base->getLoggedMessagesAsString( { Message::Warning,
                                                                    Message::Error,
                                                                    Message::Fatal } );

    const std::string infos = base->getLoggedMessagesAsString( { Message::Info,
                                                                 Message::Advice,
                                                                 Message::Deprecated } );

   if((infos.empty() && warnings.empty()))
        return;

    addGroup(component, group);
    QTreeWidgetItem *groupItem = nullptr;
    groupItem = findGroup(component, group);

    if(!groupItem)
        return;

    // log outputs
    if(!infos.empty())
    {
        QTreeWidgetItem *consoleItem = new QTreeWidgetItem(groupItem);
        const QBrush *brush = nullptr;
        if(groupItem->childCount() % 2 == 0)
            brush = new QBrush(QColor(255, 255, 191));
        else
            brush = new QBrush(QColor(255, 255, 222));
        consoleItem->setBackground(0, *brush);
        consoleItem->setBackground(1, *brush);

        QDisplayTreeItemWidget *clearWidget = new QDisplayTreeItemWidget(this, consoleItem);
        QVBoxLayout *clearLayout = new QVBoxLayout(clearWidget);

        QPushButton* clearButton = new QPushButton("Clear output", clearWidget);
        clearButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        clearButton->setFixedHeight(200);
#if (QT_VERSION < QT_VERSION_CHECK(5, 14, 0))
        clearButton->setProperty("base", qVariantFromValue((void*) base));
#else
        clearButton->setProperty("base", QVariant::fromValue((void*) base));
#endif
        clearLayout->addWidget(clearButton);

        clearWidget->setContentsMargins(0, 0, 0, 0);
        clearLayout->setContentsMargins(0, 0, 0, 0);
        clearLayout->setSpacing(0);

        QDisplayTreeItemWidget *logWidget = new QDisplayTreeItemWidget(this, consoleItem);
        QVBoxLayout *logLayout = new QVBoxLayout(logWidget);

        QTextEdit* textEdit = new QTextEdit(QString::fromStdString(infos), logWidget);
        textEdit->setText(QString::fromStdString(infos));
        textEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        textEdit->setFixedHeight(200);
        textEdit->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
        textEdit->ensureCursorVisible();
        logLayout->addWidget(textEdit);

        logWidget->setContentsMargins(0, 0, 0, 0);
        logLayout->setContentsMargins(0, 0, 0, 0);
        logLayout->setSpacing(0);

        connect(clearButton, SIGNAL(clicked()), textEdit, SLOT(clear()));
        connect(clearButton, SIGNAL(clicked()), this, SLOT(clearComponentOutput()));

        setItemWidget(consoleItem, 0, clearWidget);
        setItemWidget(consoleItem, 1, logWidget);
    }

    // warnings output
    if(!warnings.empty())
    {
        QTreeWidgetItem *consoleItem = new QTreeWidgetItem(groupItem);
        const QBrush *brush = nullptr;
        if(groupItem->childCount() % 2 == 0)
            brush = new QBrush(QColor(255, 255, 191));
        else
            brush = new QBrush(QColor(255, 255, 222));
        consoleItem->setBackground(0, *brush);
        consoleItem->setBackground(1, *brush);

        QDisplayTreeItemWidget *clearWidget = new QDisplayTreeItemWidget(this, consoleItem);
        QVBoxLayout *clearLayout = new QVBoxLayout(clearWidget);

        QPushButton* clearButton = new QPushButton("Clear warning", clearWidget);
        clearButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        clearButton->setFixedHeight(200);
#if (QT_VERSION < QT_VERSION_CHECK(5, 14, 0))
        clearButton->setProperty("base", qVariantFromValue((void*) base));
#else
        clearButton->setProperty("base", QVariant::fromValue((void*) base));
#endif
        clearLayout->addWidget(clearButton);

        clearWidget->setContentsMargins(0, 0, 0, 0);
        clearLayout->setContentsMargins(0, 0, 0, 0);
        clearLayout->setSpacing(0);

        QDisplayTreeItemWidget *logWidget = new QDisplayTreeItemWidget(this, consoleItem);
        QVBoxLayout *logLayout = new QVBoxLayout(logWidget);

        QTextEdit* textEdit = new QTextEdit(logWidget);
        textEdit->setText(QString::fromStdString(warnings));
        textEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        textEdit->setFixedHeight(200);
        textEdit->moveCursor(QTextCursor::End, QTextCursor::MoveAnchor);
        textEdit->ensureCursorVisible();
        logLayout->addWidget(textEdit);

        logWidget->setContentsMargins(0, 0, 0, 0);
        logLayout->setContentsMargins(0, 0, 0, 0);
        logLayout->setSpacing(0);

        connect(clearButton, SIGNAL(clicked()), textEdit, SLOT(clear()));
        connect(clearButton, SIGNAL(clicked()), this, SLOT(clearComponentWarning()));

        setItemWidget(consoleItem, 0, clearWidget);
        setItemWidget(consoleItem, 1, logWidget);
    }
}

void QDisplayPropertyWidget::clear()
{
    QTreeWidgetItem *item = nullptr;
    const QPushButton* pin = nullptr;
    for(unsigned int i = 0; (item = topLevelItem(i));)
    {
        pin = static_cast<QPushButton*>(itemWidget(item, 1));
        if(pin && !pin->isChecked())
        {
            objects.erase(objects.find(item));
            takeTopLevelItem(i);
        }
        else
            ++i;
    }
}

void QDisplayPropertyWidget::clearAll()
{
    QTreeWidget::clear();
}

void QDisplayPropertyWidget::clearComponentOutput()
{
    const QObject* signalEmitter = sender();
    if(nullptr == signalEmitter)
        return;

    const QVariant variant = signalEmitter->property("base");
    const sofa::core::objectmodel::Base* base = static_cast<sofa::core::objectmodel::Base*>(variant.value<void*>());
    if(base)
        base->clearLoggedMessages();
}

void QDisplayPropertyWidget::clearComponentWarning()
{
    const QObject* signalEmitter = sender();
    if(nullptr == signalEmitter)
        return;

    const QVariant variant = signalEmitter->property("base");
    const sofa::core::objectmodel::Base* base = static_cast<sofa::core::objectmodel::Base*>(variant.value<void*>());
    if(base)
        base->clearLoggedMessages();
}

QTreeWidgetItem* QDisplayPropertyWidget::findComponent(const QString& component) const
{
    QTreeWidgetItem *componentItem = nullptr;
    for(unsigned int i = 0; (componentItem = topLevelItem(i)); ++i)
        if(componentItem->text(0) == component)
            break;

    return componentItem;
}

QTreeWidgetItem* QDisplayPropertyWidget::findGroup(const QString& component, const QString& group) const
{
    const QTreeWidgetItem *componentItem = nullptr;
    componentItem = findComponent(component);
    if(!componentItem)
        return nullptr;

    QTreeWidgetItem *groupItem = nullptr;
    for(unsigned int i = 0; (groupItem = componentItem->child(i)); ++i)
        if(groupItem->text(0) == group)
            break;

    return groupItem;
}

/*void QDisplayPropertyWidget::dragEnterEvent(QDragEnterEvent *event)
{
    QModelIndex index = indexAt(event->pos());
    if(!index.isValid() && !index.parent().isValid())
        return;

    std::cout << index.row() << " - " << index.column() << std::endl;
    QTreeWidgetItem* source = itemFromIndex(index);
    if(source->checkState(0) == Qt::Unchecked)
        return;

    QTreeWidget::dragEnterEvent(event);
}

void QDisplayPropertyWidget::dropEvent(QDropEvent *event)
{
    QModelIndex index = indexAt(event->pos());
    if(!index.isValid() && !index.parent().isValid())
        return;

    std::cout << index.row() << " - " << index.column() << std::endl;
    QTreeWidgetItem* target = itemFromIndex(index);
    if(target->checkState(0) == Qt::Unchecked)
        return;

    QTreeWidget::dropEvent(event);
}*/

/*void QDisplayPropertyWidget::dragMoveEvent(QDragMoveEvent *event)
{
    QModelIndex index = indexAt(event->pos());
    QTreeWidgetItem* source = itemFromIndex(index);
    if(source->checkState(0) == Qt::Checked)
        event->accept();
}

void QDisplayPropertyWidget::dragLeaveEvent(QDragLeaveEvent *event)
{
//	QModelIndex index = indexAt(event->pos());
// 	QTreeWidgetItem* source = itemFromIndex(index);
// 	if(source->checkState(0) == Qt::Checked)
// 		event->accept();
}

Qt::DropActions QDisplayPropertyWidget::supportedDropActions() const
{
    return Qt::CopyAction | Qt::MoveAction;
}*/

} //namespace sofa::gui::qt

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
#include "QDisplayLinkWidget.h"
#include "ModifyObject.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QValidator>

#define TEXTSIZE_THRESHOLD 45

namespace sofa::gui::qt
{

QDisplayLinkWidget::QDisplayLinkWidget(QWidget* parent,
        BaseLink* link,
        const ModifyObjectFlags& flags)
    : QGroupBox(parent),
      link_(link),
      linkinfowidget_(nullptr),
      linkwidget_(nullptr),
      numWidgets_(0)
{
    if(link_ == nullptr)
    {
        return;
    }

    gridLayout_ = new QHBoxLayout();
    this->setLayout(gridLayout_);

    parent->layout()->addWidget(this);
    parent->setContentsMargins(0,0,0,0);

    setToolTip(QString::fromStdString(link->getHelp()));

    LinkWidget::CreatorArgument dwarg;
    dwarg.name =  link_->getName();
    dwarg.link = link_;
    dwarg.parent = this;
    dwarg.readOnly = (!link_->storePath() && flags.READONLY_FLAG);

    linkwidget_= LinkWidget::CreateLinkWidget(dwarg);

    if (linkwidget_ == nullptr)
    {
        linkwidget_ = new QLinkSimpleEdit(this,dwarg.link->getName().c_str(), dwarg.link);
        linkwidget_->createWidgets();
        linkwidget_->setEnabled(!(dwarg.readOnly));
        assert(linkwidget_ != nullptr);
    }

    if(linkwidget_->layout())
    {
        linkwidget_->layout()->setAlignment(Qt::AlignCenter);
        linkwidget_->layout()->setContentsMargins(2, 2, 2, 2);
    }

    linkwidget_->setContentsMargins(0, 10, 0, 0);
    linkwidget_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);

    const std::string valuetype = link_->getValueTypeString();
    if (!valuetype.empty())
        linkwidget_->setToolTip(QString::fromStdString(valuetype));

    numWidgets_ += linkwidget_->sizeWidget();
    connect(linkwidget_,SIGNAL(WidgetDirty(bool)), this, SIGNAL ( WidgetDirty(bool) ) );
    connect(this, SIGNAL( WidgetUpdate() ), linkwidget_, SLOT( updateWidgetValue() ) );
    connect(this, SIGNAL( LinkUpdate() ), linkwidget_, SLOT(setWidgetDirty() ) );
    connect(this, SIGNAL( LinkUpdate() ), linkwidget_, SLOT(updateLinkValue() ) );
    connect(linkwidget_,SIGNAL(LinkOwnerDirty(bool)),this,SIGNAL(LinkOwnerDirty(bool)) );

    if(flags.PROPERTY_WIDGET_FLAG)
    {
        if(!dwarg.readOnly)
        {
            QWidget* refreshWidget = new QWidget(this);
            refreshWidget->setFixedSize(QSize(16, 16));
            QPushButton *refresh = new QPushButton(RefreshIcon(), "", refreshWidget);
            refresh->setHidden(true);
            refresh->setFixedSize(QSize(16, 16));
            refresh->setContentsMargins(0, 0, 0, 0);

            ++numWidgets_;

            connect(linkwidget_,SIGNAL(LinkBeingChanged()), refresh, SLOT ( show() ) );
            connect(refresh, SIGNAL(clicked()), this, SLOT(UpdateLink()));
            connect(refresh, SIGNAL(clicked(bool)), refresh, SLOT(setVisible(bool)));
        }

        setStyleSheet("QGroupBox{border:0;}");
        setContentsMargins(0, 0, 0, 0);
    }
    else
    {
        setTitle(QString::fromStdString(link_->getName()));
        setContentsMargins(2,2,4,4);
    }
    gridLayout_->addWidget(linkwidget_);
}

void QDisplayLinkWidget::UpdateLink()
{
    emit LinkUpdate();
}

void QDisplayLinkWidget::UpdateWidgets()
{
    emit WidgetUpdate();
}

QLinkSimpleEdit::QLinkSimpleEdit(QWidget* parent, const char* name, BaseLink* link)
    : LinkWidget(parent,name,link)
{
}

bool QLinkSimpleEdit::createWidgets()
{
    const QString str  = QString::fromStdString(getBaseLink()->getValueString());
    QLayout* layout = new QHBoxLayout(this);
    if( str.length() > TEXTSIZE_THRESHOLD )
    {
        innerWidget_.type = TEXTEDIT;
        innerWidget_.widget.textEdit = new QTextEdit(this);
        innerWidget_.widget.textEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
        innerWidget_.widget.textEdit->setContentsMargins(0, 0, 0, 0);
        innerWidget_.widget.textEdit->setText(str);
        innerWidget_.widget.textEdit->setFixedHeight(60);
        connect(innerWidget_.widget.textEdit , SIGNAL( textChanged() ), this, SIGNAL(LinkBeingChanged()));
        connect(innerWidget_.widget.textEdit , SIGNAL( textChanged() ), this, SLOT(update()));
        layout->addWidget(innerWidget_.widget.textEdit);
    }
    else
    {
        innerWidget_.type = LINEEDIT;
        innerWidget_.widget.lineEdit  = new QLineEdit(this);
        innerWidget_.widget.lineEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
        innerWidget_.widget.lineEdit->setContentsMargins(0, 0, 0, 0);
        innerWidget_.widget.lineEdit->setText(str);
        connect( innerWidget_.widget.lineEdit, SIGNAL(textChanged(const QString&)), this, SIGNAL(LinkBeingChanged()));
        connect( innerWidget_.widget.lineEdit, SIGNAL(textChanged(const QString&)), this, SLOT(update()));
        layout->addWidget(innerWidget_.widget.lineEdit);
    }
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    return true;
}

void QLinkSimpleEdit::readFromLink()
{
    const QString str = QString::fromStdString(getBaseLink()->getValueString());
    if(innerWidget_.type == TEXTEDIT)
    {
        innerWidget_.widget.textEdit->setText(str);
    }
    else if(innerWidget_.type == LINEEDIT)
    {
        innerWidget_.widget.lineEdit->setText(str);
    }
}

void QLinkSimpleEdit::writeToLink()
{
    if(!getBaseLink())
        return ;

    if(getBaseLink()->isReadOnly())
        return;

    std::string value;
    if( innerWidget_.type == TEXTEDIT)
    {
        value = innerWidget_.widget.textEdit->toPlainText().toStdString();
    }
    else if( innerWidget_.type == LINEEDIT)
    {
        value = innerWidget_.widget.lineEdit->text().toStdString();
    }
    getBaseLink()->read(value);
}

} //namespace sofa::gui::qt

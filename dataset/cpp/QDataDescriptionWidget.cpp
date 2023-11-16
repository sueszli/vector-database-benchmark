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
#include "QDataDescriptionWidget.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseNode.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>



namespace sofa::gui::qt
{
void QDataDescriptionWidget::addRow(QGridLayout* grid, const std::string& title,
                                    const std::string& value, unsigned int row,
                                    unsigned int /*minimumWidth*/)
{
    QLabel* titlew = new QLabel(QString(title.c_str()));
    grid->addWidget(titlew, row, 0, Qt::AlignTop);

    QLabel* tmplabel = (new QLabel(QString(value.c_str())));
    tmplabel->setMinimumWidth(20);
    tmplabel->setWordWrap(true);
    tmplabel->setAlignment(Qt::AlignTop);
    tmplabel->setSizePolicy(QSizePolicy::MinimumExpanding,
                            QSizePolicy::MinimumExpanding);
    grid->addWidget(tmplabel, row, 1, Qt::AlignTop);
}

QDataDescriptionWidget::QDataDescriptionWidget(QWidget* parent, core::objectmodel::Base* object)
    :QWidget(parent)
{

    QVBoxLayout* tabLayout = new QVBoxLayout(this);
    tabLayout->setContentsMargins(0,0,0,0);
    tabLayout->setSpacing(1);
    tabLayout->setObjectName("tabInfoLayout");

    //Instance
    {
        QGroupBox *box = new QGroupBox(this);
        tabLayout->addWidget(box);
        QGridLayout* boxLayout = new QGridLayout();
        box->setLayout(boxLayout);

        box->setTitle(QString("Instance"));

        addRow(boxLayout, "Name", object->getName(), 0);
        addRow(boxLayout, "Class", object->getClassName(), 1);

        const std::string namespacename = sofa::helper::NameDecoder::decodeNamespaceName(typeid(*object));

        int nextRow = 2;
        if (!namespacename.empty())
        {
            addRow(boxLayout, "Namespace", namespacename, nextRow, 20);
            nextRow++;
        }
        if (!object->getTemplateName().empty())
        {
            addRow(boxLayout, "Template", object->getTemplateName(), nextRow, 20);
            nextRow++;
        }

        const core::objectmodel::BaseNode* node = object->toBaseNode(); // Node
        if (node && node->getNbParents()>1) // MultiNode
        {
            addRow(boxLayout, "Path", node->getPathName(), nextRow, 20);
            nextRow++;
        }

        tabLayout->addWidget( box );
    }


    //Class description
    core::ObjectFactory::ClassEntry entry = core::ObjectFactory::getInstance()->getEntry(object->getClassName());
    if (! entry.creatorMap.empty())
    {
        QGroupBox *box = new QGroupBox(this);
        tabLayout->addWidget(box);
        QGridLayout* boxLayout = new QGridLayout();
        box->setLayout(boxLayout);
        box->setTitle(QString("Class"));

        int nextRow = 0;
        if (!entry.description.empty() && entry.description != std::string("TODO"))
        {
            addRow(boxLayout, "Description", entry.description, nextRow, 20);
            nextRow++;
        }
        const core::ObjectFactory::CreatorMap::iterator it = entry.creatorMap.find(object->getTemplateName());
        if (it != entry.creatorMap.end() && *it->second->getTarget())
        {
            addRow(boxLayout, "Provided by",it->second->getTarget(), nextRow, 20);
            nextRow++;
        }

        if (!entry.authors.empty() && entry.authors != std::string("TODO"))
        {
            addRow(boxLayout, "Authors", entry.authors, nextRow, 20);
            nextRow++;
        }
        if (!entry.license.empty() && entry.license != std::string("TODO"))
        {
            addRow(boxLayout, "License", entry.license, nextRow, 20);
            nextRow++;
        }
        tabLayout->addWidget( box );
    }



    //Extra description
    std::vector<sofa::core::objectmodel::BaseData*> selecteddatum ;
    for(sofa::core::objectmodel::BaseData* datafield : object->getDataFields())
    {
        if( datafield->getGroup() == "Infos" )
            selecteddatum.push_back(datafield) ;
    }

    if(!selecteddatum.empty())
    {
        QGroupBox *box = new QGroupBox(this);
        tabLayout->addWidget(box);
        QGridLayout* boxLayout = new QGridLayout();

        box->setLayout(boxLayout);

        box->setTitle(QString("Extra informations"));

        unsigned int row = 0;
        for(const auto& data : selecteddatum)
        {
            addRow(boxLayout, data->getName(), data->getValueString(), row++);
        }
    }


    tabLayout->addStretch();
}




} //namespace sofa::gui::qt

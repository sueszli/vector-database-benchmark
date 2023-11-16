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
#include "PickHandlerCallBacks.h"
#include "RealGUI.h"
#include "viewer/SofaViewer.h"
#include "QSofaListView.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/gui/component/performer/MouseInteractor.h>

#include <QStatusBar>

using namespace sofa::gui::common;

namespace sofa::gui::qt
{

InformationOnPickCallBack::InformationOnPickCallBack()
    :gui(nullptr)
{
}

InformationOnPickCallBack::InformationOnPickCallBack(RealGUI *g)
    :gui(g)
{
}

void InformationOnPickCallBack::execute(const BodyPicked &body)
{
    if(!gui) return;
    core::objectmodel::BaseObject *objectPicked=nullptr;
    if (body.body)
    {
        QTreeWidgetItem* item=gui->simulationGraph->getListener()->items[body.body];
//        gui->simulationGraph->ensureItemVisible(item);
        gui->simulationGraph->scrollToItem(item);
        gui->simulationGraph->clearSelection();
//        gui->simulationGraph->setSelected(item,true);
        gui->simulationGraph->setCurrentItem(item);
        objectPicked=body.body;
    }
    else if (body.mstate)
    {
        QTreeWidgetItem* item=gui->simulationGraph->getListener()->items[body.mstate];
//        gui->simulationGraph->ensureItemVisible(item);
        gui->simulationGraph->scrollToItem(item);
        gui->simulationGraph->clearSelection();
//        gui->simulationGraph->setSelected(item,true);
        gui->simulationGraph->setCurrentItem(item);
        objectPicked=body.mstate;
    }
    else
        gui->simulationGraph->clearSelection();

    if (objectPicked)
    {
        QString messagePicking;
        const simulation::Node *n=static_cast<simulation::Node*>(objectPicked->getContext());
        messagePicking=QString("Index ") + QString::number(body.indexCollisionElement)
                + QString(" of  ")
                + QString(n->getPathName().c_str())
                + QString("/") + QString(objectPicked->getName().c_str())
                + QString(" : ") + QString(objectPicked->getClassName().c_str());
        if (!objectPicked->getTemplateName().empty())
            messagePicking += QString("<") + QString(objectPicked->getTemplateName().c_str()) + QString(">");
        gui->statusBar()->showMessage(messagePicking,3000); //display message during 3 seconds
    }
}

ColourPickingRenderCallBack::ColourPickingRenderCallBack()
    :_viewer(nullptr)
{
}

ColourPickingRenderCallBack::ColourPickingRenderCallBack(viewer::SofaViewer* viewer)
    :_viewer(viewer)
{
}

void ColourPickingRenderCallBack::render(ColourPickingVisitor::ColourCode code)
{
    if(_viewer)
    {
        _viewer->drawColourPicking(code);
    }

}

} // namespace sofa::gui::qt

/****************************************************************************
* MeshLab                                                           o o     *
* A versatile mesh processing toolbox                             o     o   *
*                                                                _   O  _   *
* Copyright(C) 2005-2008                                           \/)\/    *
* Visual Computing Lab                                            /\/|      *
* ISTI - Italian National Research Council                           |      *
*                                                                    \      *
* All rights reserved.                                                      *
*                                                                           *
* This program is free software; you can redistribute it and/or modify      *   
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation; either version 2 of the License, or         *
* (at your option) any later version.                                       *
*                                                                           *
* This program is distributed in the hope that it will be useful,           *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                         *
*                                                                           *
****************************************************************************/

#include "edit_manipulators_factory.h"
#include "edit_manipulators.h"

EditManipulatorsFactory::EditManipulatorsFactory()
{
	editManipulators = new QAction(QIcon(":/images/icon_manipulators.png"),"Manipulators Tool", this);

	actionList.push_back(editManipulators);
	
	foreach(QAction *editAction, actionList)
		editAction->setCheckable(true); 	
}

QString EditManipulatorsFactory::pluginName() const
{
	return "EditManipulators";
}

//get the edit tool for the given action
EditTool* EditManipulatorsFactory::getEditTool(const QAction *action)
{
	if(action == editManipulators) {
		return new EditManipulatorsPlugin();
	}
	else
		assert(0); //should never be asked for an action that isn't here
	return NULL;
}

QString EditManipulatorsFactory::getEditToolDescription(const QAction *)
{
	return EditManipulatorsPlugin::info();
}

MESHLAB_PLUGIN_NAME_EXPORTER(EditManipulatorsFactory)

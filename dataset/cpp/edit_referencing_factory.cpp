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

#include "edit_referencing_factory.h"
#include "edit_referencing.h"

EditReferencingFactory::EditReferencingFactory()
{
	editReferencing = new QAction(QIcon(":/images/icon_referencing.png"),"Reference scene", this);
	
	actionList.push_back(editReferencing);
	
	foreach(QAction *editAction, actionList)
		editAction->setCheckable(true); 	
}

QString EditReferencingFactory::pluginName() const
{
	return "EditReferencing";
}

//get the edit tool for the given action
EditTool* EditReferencingFactory::getEditTool(const QAction *action)
{
	if(action == editReferencing) {
		return new EditReferencingPlugin();
	}
	else
		assert(0); //should never be asked for an action that isn't here
	return nullptr;
}

QString EditReferencingFactory::getEditToolDescription(const QAction *)
{
	return EditReferencingPlugin::info();
}

MESHLAB_PLUGIN_NAME_EXPORTER(EditReferencingFactory)

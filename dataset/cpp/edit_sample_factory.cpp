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

#include "edit_sample_factory.h"
#include "edit_sample.h"

SampleEditFactory::SampleEditFactory()
{
	editSample = new QAction(QIcon(":/images/icon_info.png"),"Get Info", this);
	
	actionList.push_back(editSample);
	
	foreach(QAction *editAction, actionList)
		editAction->setCheckable(true); 	
}

QString SampleEditFactory::pluginName() const
{
	return "EditSample";
}

//get the edit tool for the given action
EditTool* SampleEditFactory::getEditTool(const QAction *action)
{
	if(action == editSample)
	{
		return new SampleEditPlugin();
	} else assert(0); //should never be asked for an action that isn't here
	return nullptr;
}

QString SampleEditFactory::getEditToolDescription(const QAction *)
{
	return SampleEditPlugin::info();
}

MESHLAB_PLUGIN_NAME_EXPORTER(SampleEditFactory)

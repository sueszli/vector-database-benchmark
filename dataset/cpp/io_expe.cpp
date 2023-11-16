/****************************************************************************
* MeshLab                                                           o o     *
* An extendible mesh processor                                    o     o   *
*                                                                _   O  _   *
* Copyright(C) 2005, 2006                                          \/)\/    *
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

#include <Qt>

#include "io_expe.h"

#include <wrap/io_trimesh/export.h>
#include <wrap/io_trimesh/io_mask.h>

#include "import_expe.h"
#include "import_xyz.h"
#include "export_xyz.h"
// #include "export_expe.h"

using namespace std;
using namespace vcg;



void ExpeIOPlugin::open(const QString &formatName, const QString &fileName, MeshModel &m, int& mask, const RichParameterList & /*parlst*/, CallBackPos *cb)
{
	// initializing mask
	mask = 0;
	
	// initializing progress bar status
	if (cb != NULL)		(*cb)(0, "Loading...");
	
	QString errorMsgFormat = "Error encountered while loading file:\n\"%1\"\n\nError details: %2";
	QString error_2MsgFormat = "Error encountered while loading file:\n\"%1\"\n\n File with more than a mesh.\n Load only the first!";
	
	string filename = QFile::encodeName(fileName).constData ();
	bool useXYZ=false;
	if ( (formatName.toLower() == tr("pts")) || (formatName.toLower() == tr("apts")) )
	{
		int loadMask;
		if (!vcg::tri::io::ImporterExpePTS<CMeshO>::LoadMask(filename.c_str(),loadMask))
		{
			useXYZ=true;
			if (!vcg::tri::io::ImporterXYZ<CMeshO>::LoadMask(filename.c_str(),loadMask)){
				throw MLException("Error while loading [A]PTS mask.");
			}
		}
		m.enable(loadMask);
		int result;
		if(useXYZ) {
			result = vcg::tri::io::ImporterXYZ<CMeshO>::Open(m.cm, filename.c_str(), mask, cb);
			if (result != 0)
			{
				throw MLException(errorMsgFormat.arg(fileName, vcg::tri::io::ImporterXYZ<CMeshO>::ErrorMsg(result)));
			}
		}
		else 
		{
			result = vcg::tri::io::ImporterExpePTS<CMeshO>::Open(m.cm, filename.c_str(), mask, cb);
			if (result != 0)
			{
				throw MLException(errorMsgFormat.arg(fileName, vcg::tri::io::ImporterExpePTS<CMeshO>::ErrorMsg(result)));
			}
		}
		
	}
	else if (formatName.toLower() == tr("xyz"))
	{
		int loadMask;
		if (!vcg::tri::io::ImporterXYZ<CMeshO>::LoadMask(filename.c_str(),loadMask)) {
			throw MLException("Error while loading XYZ mask.");
		}
		m.enable(loadMask);
		
		
		int result = vcg::tri::io::ImporterXYZ<CMeshO>::Open(m.cm, filename.c_str(), mask, cb);
		if (result != 0)
		{
			throw MLException(errorMsgFormat.arg(fileName, vcg::tri::io::ImporterXYZ<CMeshO>::ErrorMsg(result)));
		}
	}
	
	vcg::tri::UpdateBounding<CMeshO>::Box(m.cm);					// updates bounding box
	
	if (cb != NULL)
		(*cb)(99, "Done");
}

void ExpeIOPlugin::save(const QString &formatName, const QString &fileName, MeshModel &m, const int mask, const RichParameterList &, vcg::CallBackPos * /*cb*/)
{
	QString errorMsgFormat = "Error encountered while exporting file %1:\n%2";
	string filename = QFile::encodeName(fileName).constData ();
	string ex = formatName.toUtf8().data();
	
	// 	if( formatName.toUpper() == tr("GTS") )
	// 	{
	// 		int result = vcg::tri::io::ExporterGTS<CMeshO>::Save(m.cm,filename.c_str(),mask);
	// 		if(result!=0)
	// 		{
	// 			QMessageBox::warning(parent, tr("Saving Error"), errorMsgFormat.arg(fileName, vcg::tri::io::ExporterGTS<CMeshO>::ErrorMsg(result)));
	// 			return false;
	// 		}
	// 		return true;
	// 	}
	
	if(formatName.toLower() == tr("xyz"))
	{
		int result = vcg::tri::io::ExporterXYZ<CMeshO>::Save(m.cm,filename.c_str(),mask);
		if(result!=0)
		{
			throw MLException("Saving Error: " + errorMsgFormat.arg(fileName, vcg::tri::io::ExporterXYZ<CMeshO>::ErrorMsg(result)));
		}
	}
	else {
		wrongSaveFormat(formatName);
	}
}

/*
	returns the list of the file's type which can be imported
*/
QString ExpeIOPlugin::pluginName() const
{
	return "IOExpe";
}

std::list<FileFormat> ExpeIOPlugin::importFormats() const
{
	std::list<FileFormat> formatList = {
		FileFormat("Expe's point set (binary)" ,tr("pts")),
		FileFormat("Expe's point set (ascii)" ,tr("apts")),
		FileFormat("XYZ Point Cloud (with or without normal)",tr("xyz"))
	};
	return formatList;
}

/*
	returns the list of the file's type which can be exported
*/
std::list<FileFormat> ExpeIOPlugin::exportFormats() const
{
	std::list<FileFormat> formatList = {
		// FileFormat("Expe's point set (binary)" ,tr("pts")),
		// FileFormat("Expe's point set (ascii)" ,tr("apts")),
		FileFormat("XYZ Point Cloud (with or without normal)" ,tr("xyz"))
	};
	return formatList;
}

/*
	returns the mask on the basis of the file's type.
	otherwise it returns 0 if the file format is unknown
*/
void ExpeIOPlugin::exportMaskCapability(const QString &format, int &capability, int &defaultBits) const
{
	// 	if(format.toLower() == tr("apts")){capability=defaultBits= vcg::tri::io::ExporterExpeAPTS<CMeshO>::GetExportMaskCapability();}
	// 	if(format.toLower() == tr("pts")){capability=defaultBits= vcg::tri::io::ExporterExpePTS<CMeshO>::GetExportMaskCapability();}
	if(format.toLower() == tr("xyz")){capability=defaultBits= vcg::tri::io::ExporterXYZ<CMeshO>::GetExportMaskCapability();}
	return;
}

MESHLAB_PLUGIN_NAME_EXPORTER(ExpeIOPlugin)

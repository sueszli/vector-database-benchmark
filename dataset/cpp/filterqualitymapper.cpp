/****************************************************************************
* MeshLab                                                           o o     *
* A versatile mesh processing toolbox                             o     o   *
*                                                                _   O  _   *
* Copyright(C) 2005                                                \/)\/    *
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
/****************************************************************************
History
Revision 1.0 2008/02/20 Alessandro Maione, Federico Bellucci
FIRST RELEASE

****************************************************************************/
#include <math.h>
#include <stdlib.h>
#include "filterqualitymapper.h"

QualityMapperFilter::QualityMapperFilter() 
{ 
	typeList = {FP_QUALITY_MAPPER};

	for(ActionIDType tt: types())
		actionList.push_back(new QAction(filterName(tt), this));
}

QString QualityMapperFilter::pluginName() const
{
	return "FilterQuality";
}

QString QualityMapperFilter::filterName(ActionIDType filterId) const
{
	switch(filterId) {
	case FP_QUALITY_MAPPER :  return QString("Quality Mapper applier");
	default : assert(0); return QString();
	}
}

QString QualityMapperFilter::pythonFilterName(ActionIDType f) const
{
	switch(f) {
	case FP_QUALITY_MAPPER :
		return QString("compute_color_from_scalar_using_transfer_function_per_vertex");
	default : assert(0); return QString();
	}
}


QString QualityMapperFilter::filterInfo(ActionIDType filterId) const
{
	switch(filterId) {
	case FP_QUALITY_MAPPER :
		return QString(
				"The filter maps quality levels into colors using a colorband "
				"built from a transfer function (may be loaded from an external "
				"file) and colorizes the mesh vertices. The minimum, medium and "
				"maximum quality values can be set by user to obtain a custom "
				"quality range for mapping");
	default : assert(0);
	}
	return QString("");
}

FilterPlugin::FilterClass QualityMapperFilter::getClass(const QAction *a) const
{
	switch(ID(a))
	{
	case FP_QUALITY_MAPPER : return FilterPlugin::Quality;
	default : assert(0);     return FilterPlugin::Generic;
	}
}

RichParameterList QualityMapperFilter::initParameterList(const QAction *action,const MeshModel &m)
{
	RichParameterList parlst;
	switch(ID(action)) {
	case FP_QUALITY_MAPPER :
	{
		_meshMinMaxQuality = tri::Stat<CMeshO>::ComputePerVertexQualityMinMax(m.cm);

		parlst.addParam(RichFloat("minQualityVal", _meshMinMaxQuality.minV, "Minimum mesh quality","The specified quality value is mapped in the <b>lower</b> end of the chosen color scale. Default value: the minimum quality value found on the mesh."  ));
		parlst.addParam(RichFloat("maxQualityVal", _meshMinMaxQuality.maxV, "Maximum mesh quality","The specified quality value is mapped in the <b>upper</b> end of the chosen color scale. Default value: the maximum quality value found on the mesh." ));
		parlst.addParam(RichFloat("midHandlePos", 50, "Gamma biasing (0..100)", "Defines a gamma compression of the quality values, by setting the position of the middle of the color scale. Value is defined as a percentage (0..100). Default value is 50, that corresponds to a linear mapping." ));
		parlst.addParam(RichFloat("brightness", 1.0f, "Mesh brightness", "must be between 0 and 2. 0 represents a completely dark mesh, 1 represents a mesh colorized with original colors, 2 represents a completely bright mesh"));
		//setting default transfer functions names
		TransferFunction::defaultTFs[GREY_SCALE_TF] = "Grey Scale";
		TransferFunction::defaultTFs[MESHLAB_RGB_TF] = "Meshlab RGB";
		TransferFunction::defaultTFs[RGB_TF] = "RGB";
		TransferFunction::defaultTFs[FRENCH_RGB_TF] = "French RGB";
		TransferFunction::defaultTFs[RED_SCALE_TF] = "Red Scale";
		TransferFunction::defaultTFs[GREEN_SCALE_TF] = "Green Scale";
		TransferFunction::defaultTFs[BLUE_SCALE_TF] = "Blue Scale";
		TransferFunction::defaultTFs[FLAT_TF] = "Flat";
		TransferFunction::defaultTFs[SAW_4_TF] = "Saw 4";
		TransferFunction::defaultTFs[SAW_8_TF] = "Saw 8";

		QStringList tfList;
		tfList << "Custom Transfer Function File";
		for (int i=0; i<NUMBER_OF_DEFAULT_TF; i++ )
			//fetching and adding default TFs to TFList
			tfList << TransferFunction::defaultTFs[(STARTUP_TF_TYPE + i)%NUMBER_OF_DEFAULT_TF];

		parlst.addParam(RichEnum( "TFsList", 1, tfList, "Transfer Function type to apply to filter", "Choose the Transfer Function to apply to the filter" ));
		parlst.addParam(RichString("csvFileName", "", "Custom TF Filename", "Filename of the transfer function to be loaded, used only if you have chosen the Custom Transfer Function. Write the full path of the qmap file, or save the file in the same folder of the current mesh, and write only the name of the qmap file. Only the RGB mapping will be imported from the qmap file" ));
	}
		break;
	default: break; // do not add any parameter for the other filters
	}
	return parlst;
}

std::map<std::string, QVariant> QualityMapperFilter::applyFilter(
		const QAction *filter,
		const RichParameterList & par,
		MeshDocument &md,
		unsigned int& /*postConditionMask*/,
		vcg::CallBackPos *)
{
	if (ID(filter) == FP_QUALITY_MAPPER) {
		MeshModel &m=*(md.mm());
		m.updateDataMask(MeshModel::MM_VERTCOLOR);

		EQUALIZER_INFO eqData;
		eqData.minQualityVal = par.getFloat("minQualityVal");
		eqData.midQualityPercentage = par.getFloat("midHandlePos");
		eqData.maxQualityVal = par.getFloat("maxQualityVal");
		eqData.brightness = par.getFloat("brightness");
		TransferFunction *transferFunction = 0;

		if ( par.getEnum("TFsList") != 0 )
		{
			//default TF
			//building new TF object from default Tf type
			int tfId = ((par.getEnum("TFsList")-1) + STARTUP_TF_TYPE) % NUMBER_OF_DEFAULT_TF;
			transferFunction = new TransferFunction( (DEFAULT_TRANSFER_FUNCTIONS) tfId );
		}
		else
		{
			//text TF
			QString csvFileName = par.getString("csvFileName");
			if ( csvFileName != "" &&  loadEqualizerInfo(csvFileName, &eqData) > 0 )
			{
				/*par.setValue("minQualityVal", FloatValue(eqData.minQualityVal) );
					par.setValue("maxQualityVal", FloatValue(eqData.maxQualityVal) );
					par.setValue("midHandlePos", FloatValue(_meshMinMaxQuality.minV + ((_meshMinMaxQuality.maxV-_meshMinMaxQuality.minV)/eqData.midQualityPercentage)));
					par.setValue("brightness", FloatValue(eqData.brightness));*/

				//building new TF object from external file
				transferFunction = new TransferFunction( par.getString("csvFileName") );
			}
			else
			{
				throw MLException("Something went wrong while trying to open the specified transfer function file");
			}
		}
		// Applying colors
		applyColorByVertexQuality(m, transferFunction, par.getFloat("minQualityVal"), par.getFloat("maxQualityVal"), par.getFloat("midHandlePos")/100.0, par.getFloat("brightness"));

		//all done, deleting transfer function object
		if ( transferFunction )
		{
			delete transferFunction;
			transferFunction = 0;
		}
	}
	else {
		wrongActionCalled(filter);
	}

	return std::map<std::string, QVariant>();
}

int QualityMapperFilter::getPreConditions(const QAction* a) const
{
	switch(ID(a))
	{
	case FP_QUALITY_MAPPER :
		return MeshModel::MM_VERTQUALITY;
	default :
		assert(0);
		return MeshModel::MM_NONE;
	}
}

int QualityMapperFilter::postCondition(const QAction* a) const
{
	switch(ID(a))
	{
	case FP_QUALITY_MAPPER :
		return MeshModel::MM_VERTCOLOR;
	default :
		assert(0);
		return MeshModel::MM_NONE;
	}
}
MESHLAB_PLUGIN_NAME_EXPORTER(QualityMapperFilter)

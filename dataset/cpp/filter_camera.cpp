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

#include "filter_camera.h"

#include <vcg/complex/algorithms/clean.h>


using namespace std;
using namespace vcg;


// Constructor
FilterCameraPlugin::FilterCameraPlugin()
{
	typeList  = {
		FP_SET_MESH_CAMERA,
		FP_SET_RASTER_CAMERA,
		FP_QUALITY_FROM_CAMERA,
		FP_CAMERA_ROTATE,
		FP_CAMERA_SCALE,
		FP_CAMERA_TRANSLATE,
		FP_CAMERA_TRANSFORM,
		FP_ORIENT_NORMALS_WITH_CAMERAS
	};
	
	for(ActionIDType tt : types())
		actionList.push_back(new QAction(filterName(tt), this));
}

QString FilterCameraPlugin::pluginName() const
{
	return "FilterCamera";
}

QString FilterCameraPlugin::pythonFilterName(ActionIDType f) const
{
	switch (f) {
	case FP_SET_MESH_CAMERA :           return QString("set_camera_per_mesh");
	case FP_SET_RASTER_CAMERA :         return QString("set_camera_per_raster");
	case FP_QUALITY_FROM_CAMERA :       return QString("compute_scalar_from_camera_per_vertex");
	case FP_CAMERA_ROTATE :             return QString("apply_cameras_rotation");
	case FP_CAMERA_SCALE :              return QString("apply_cameras_scaling");
	case FP_CAMERA_TRANSLATE :          return QString("apply_cameras_translation");
	case FP_CAMERA_TRANSFORM :          return QString("apply_cameras_extrinsics_transformation");
	case FP_ORIENT_NORMALS_WITH_CAMERAS:return QString("compute_normal_from_cameras_per_vertex");
	default : assert(0); return QString();
	}
}

QString FilterCameraPlugin::filterName(ActionIDType filterId) const
{
	switch(filterId) {
	case FP_SET_MESH_CAMERA :           return QString("Set Mesh Camera");
	case FP_SET_RASTER_CAMERA :         return QString("Set Raster Camera");
	case FP_QUALITY_FROM_CAMERA :       return QString("Vertex Quality from Camera");
	case FP_CAMERA_ROTATE :             return QString("Transform: Rotate Camera or set of cameras");
	case FP_CAMERA_SCALE :              return QString("Transform: Scale Camera or set of cameras");
	case FP_CAMERA_TRANSLATE :          return QString("Transform: Translate Camera or set of cameras");
	case FP_CAMERA_TRANSFORM :          return QString("Transform the camera extrinsics, or all the cameras of the project");
	case FP_ORIENT_NORMALS_WITH_CAMERAS:return QString("Re-Orient vertex normals using cameras");
	default : assert(0); return QString();
	}
}

// Info() return the longer string describing each filtering action
QString FilterCameraPlugin::filterInfo(ActionIDType filterId) const
{
	switch(filterId) {
	case FP_SET_MESH_CAMERA :           return QString("This filter allows one to set a shot for the current mesh");
	case FP_SET_RASTER_CAMERA :         return QString("This filter allows one to set a shot for the current mesh");
	case FP_QUALITY_FROM_CAMERA :       return QString("Compute vertex quality using the camera definition, according to viewing angle or distance");
	case FP_CAMERA_ROTATE :             return QString("Rotate the camera, or all the cameras of the project. The selected raster is the reference if viewpoint rotation is selected.");
	case FP_CAMERA_SCALE :              return QString("Scale the camera, or all the cameras of the project. The selected raster is the reference if viewpoint scaling is selected.");
	case FP_CAMERA_TRANSLATE :          return QString("Translate the camera, or all the cameras of the project.");
	case FP_CAMERA_TRANSFORM :          return QString("Transform the camera extrinsics, or all the cameras of the project.");
	case FP_ORIENT_NORMALS_WITH_CAMERAS:return QString("Reorient vertex normals using cameras. For this  filter to work the mesh needs to have the attribute 'correspondences' which is only created when loading Bundler files (.out projects)");
	default : assert(0);
	}
	return NULL;
}

// This function define the needed parameters for each filter.
RichParameterList FilterCameraPlugin::initParameterList(const QAction *action, const MeshDocument &/*m*/)
{
	RichParameterList parlst;
	Shotm defShot;
	switch(ID(action))
	{
	case FP_CAMERA_ROTATE :
	{
		QStringList shotType;
		shotType.push_back("Raster Camera");
		shotType.push_back("Mesh Camera");
		parlst.addParam(RichEnum("camera", 0, shotType, tr("Camera type"), tr("Choose the camera to scale")));
		QStringList rotMethod;
		rotMethod.push_back("X axis");
		rotMethod.push_back("Y axis");
		rotMethod.push_back("Z axis");
		rotMethod.push_back("custom axis");
		parlst.addParam(RichEnum("rotAxis", 0, rotMethod, tr("Rotation on:"), tr("Choose a method")));
		QStringList rotCenter;
		rotCenter.push_back("origin");
		rotCenter.push_back("camera viewpoint");
		rotCenter.push_back("custom point");
		parlst.addParam(RichEnum("rotCenter", 0, rotCenter, tr("Center of rotation:"), tr("Choose a method")));
		parlst.addParam(RichDynamicFloat("angle",0,-360,360,"Rotation Angle","Angle of rotation (in <b>degree</b>). If snapping is enabled this value is rounded according to the snap value"));
		parlst.addParam(RichDirection("customAxis",Point3m(0,0,0),"Custom axis","This rotation axis is used only if the 'custom axis' option is chosen."));
		parlst.addParam(RichPosition("customCenter",Point3m(0,0,0),"Custom center","This rotation center is used only if the 'custom point' option is chosen."));
		parlst.addParam(RichBool ("toallRaster", false, "Apply to all active Raster layers", "Apply the same scaling to all the active Raster layers: it is taken into account only if 'Raster Camera' is selected"));
		parlst.addParam(RichBool ("toall", false, "Apply to all active Raster and visible Mesh layers", "Apply the same scaling to all the layers, including any visible 3D layer"));
	}
		break;
	case FP_CAMERA_SCALE :
	{
		QStringList shotType;
		shotType.push_back("Raster Camera");
		shotType.push_back("Mesh Camera");
		parlst.addParam(RichEnum("camera", 0, shotType, tr("Camera type"), tr("Choose the camera to scale")));
		QStringList scaleCenter;
		scaleCenter.push_back("origin");
		scaleCenter.push_back("camera viewpoint");
		scaleCenter.push_back("custom point");
		parlst.addParam(RichEnum("scaleCenter", 0, scaleCenter, tr("Center of scaling:"), tr("Choose a method")));
		parlst.addParam(RichPosition("customCenter",Point3m(0,0,0),"Custom center","This scaling center is used only if the 'custom point' option is chosen."));
		parlst.addParam(RichFloat("scale", 1.0, "Scale factor", "The scale factor that has to be applied to the camera"));
		parlst.addParam(RichBool ("toallRaster", false, "Apply to all active Raster layers", "Apply the same scaling to all the active Raster layers: it is taken into account only if 'Raster Camera' is selected"));
		parlst.addParam(RichBool ("toall", false, "Apply to all active Raster and visible Mesh layers", "Apply the same scaling to all the layers, including any visible 3D layer"));
	}
		break;
	case FP_CAMERA_TRANSLATE :
	{
		QStringList shotType;
		shotType.push_back("Raster Camera");
		shotType.push_back("Mesh Camera");
		parlst.addParam(RichEnum("camera", 0, shotType, tr("Camera type"), tr("Choose the camera to scale")));
		parlst.addParam(RichDynamicFloat("axisX",0,-1000,1000,"X Axis","Absolute translation amount along the X axis"));
		parlst.addParam(RichDynamicFloat("axisY",0,-1000,1000,"Y Axis","Absolute translation amount along the Y axis"));
		parlst.addParam(RichDynamicFloat("axisZ",0,-1000,1000,"Z Axis","Absolute translation amount along the Z axis"));
		parlst.addParam(RichBool("centerFlag",false,"translate viewpoint position to the origin","If selected, the camera viewpoint is translated to the origin"));
		parlst.addParam(RichBool ("toallRaster", false, "Apply to all active Raster layers", "Apply the same scaling to all the active Raster layers: it is taken into account only if 'Raster Camera' is selected"));
		parlst.addParam(RichBool ("toall", false, "Apply to all active Raster and Mesh layers", "Apply the same scaling to all the layers, including any visible 3D layer"));
	}
		break;
		
	case FP_CAMERA_TRANSFORM :
	{
		QStringList shotType;
		shotType.push_back("Raster Camera");
		shotType.push_back("Mesh Camera");
		QStringList behaviour;
		behaviour.push_back("The matrix is the transformation to apply to the extrinsics");
		behaviour.push_back("The matrix represent the new extrinsics");
		
		parlst.addParam(RichMatrix44f("TransformMatrix",Matrix44m::Identity(),""));
		parlst.addParam(RichEnum("camera", 0, shotType, tr("Camera type"), tr("Choose the camera to scale")));
		parlst.addParam(RichEnum("behaviour", 0, behaviour, tr("Matrix semantic"), tr("What the matrix is used for")));
		parlst.addParam(RichBool ("toallRaster", false, "Apply to all active Raster layers", "Apply the same scaling to all the active Raster layers: it is taken into account only if 'Raster Camera' is selected"));
		parlst.addParam(RichBool ("toall", false, "Apply to all active Raster and visible Mesh layers", "Apply the same scaling to all the layers, including any visible 3D layer"));
		
	}
		break;  case FP_SET_RASTER_CAMERA :
		parlst.addParam(RichShotf ("Shot", defShot, "New shot", "This filter allows one to set a shot for the current raster."));
		break;
		
	case FP_SET_MESH_CAMERA :
		parlst.addParam(RichShotf ("Shot", defShot, "New shot", "This filter allows one to set a shot for the current mesh."));
		break;
	case FP_QUALITY_FROM_CAMERA :
		parlst.addParam(RichBool ("Depth", true, "Depth", "Use depth as a factor."));
		parlst.addParam(RichBool ("Facing", false, "ViewAngle", "Use cosine of viewing angle as a factor."));
		parlst.addParam(RichBool ("Clip", false,  "Clipping", "clip values outside the viewport to zero."));
		parlst.addParam(RichBool("normalize",false,"normalize","if checked normalize all quality values in range [0..1]"));
		parlst.addParam(RichBool("map",false,"map into color", "if checked map quality generated values into per-vertex color"));
		break;
	default: break; // do not add any parameter for the other filters
	}
	return parlst;
}

// Core Function doing the actual mesh processing.
std::map<std::string, QVariant> FilterCameraPlugin::applyFilter(const QAction *filter, const RichParameterList & par, MeshDocument &md, unsigned int& /*postConditionMask*/, vcg::CallBackPos * /*cb*/)
{
	MeshModel* currentMesh = md.mm();
	CMeshO* cm = NULL;
	if (currentMesh != NULL)
		cm = &(currentMesh->cm);
	RasterModel *currentRaster = md.rm();
	switch(ID(filter))
	{
	case FP_CAMERA_ROTATE :
	{
		Matrix44m trRot; trRot.SetIdentity();
		Point3m axis, tranVec;
		Matrix44m trTran,trTranInv;
		
		switch(par.getEnum("rotAxis"))
		{
		case 0: 
			axis=Point3m(1,0,0); 
			break;
		case 1: 
			axis=Point3m(0,1,0);
			break;
		case 2: 
			axis=Point3m(0,0,1);
			break;
		case 3: 
			axis=par.getPoint3m("customAxis");
			break;
		}
		switch(par.getEnum("rotCenter"))
		{
		case 0: 
			tranVec=Point3m(0,0,0); 
			break;
		case 1: 
		{
			switch(par.getEnum("camera"))
			{
			case 0:
				if (currentRaster == NULL)
				{
					throw MLException("You need a Raster Model to apply this filter!");
				}
				tranVec=currentRaster->shot.Extrinsics.Tra();
				break;
			case 1: 	
				if (cm == NULL)
				{
					throw MLException("You need a Mesh Model to apply this filter!");
				}
				tranVec=cm->shot.Extrinsics.Tra();
				break;
			}
		}
			break;
		case 2: 
			tranVec=par.getPoint3m("customCenter");
			break;
		}
		
		Scalarm angleDeg= par.getDynamicFloat("angle");
		
		trRot.SetRotateDeg(angleDeg,axis);
		trTran.SetTranslate(tranVec);
		trTranInv.SetTranslate(-tranVec);
		//cm.Tr=trTran*trRot*trTranInv;
		Matrix44m transf=trTran*trRot*trTranInv;
		if (par.getBool("toall")) {
			for (MeshModel& mm : md.meshIterator()) {
				if (mm.isVisible()) {
					mm.cm.Tr=transf * mm.cm.Tr;
					tri::UpdatePosition<CMeshO>::Matrix(mm.cm, mm.cm.Tr);
					//tri::UpdateNormal<CMeshO>::PerVertexMatrix(md.meshList[i]->cm,md.meshList[i]->cm.Tr);
					//tri::UpdateNormal<CMeshO>::PerFaceMatrix(md.meshList[i]->cm,md.meshList[i]->cm.Tr);
					tri::UpdateBounding<CMeshO>::Box(mm.cm);
					mm.cm.shot.ApplyRigidTransformation(transf);
					mm.cm.Tr.SetIdentity();
				}
				
			}
			for (RasterModel& rm : md.rasterIterator()) {
				if (rm.isVisible())
					rm.shot.ApplyRigidTransformation(transf);
			}
		}
		else if (par.getBool("toallRaster") && (par.getEnum("camera")==0)) {
			for (RasterModel& rm : md.rasterIterator()) {
				if (rm.isVisible())
					rm.shot.ApplyRigidTransformation(transf);
			}
		}
		else switch(par.getEnum("camera"))
		{
		case 0:
		{
			if (currentRaster == NULL)
			{
				throw MLException("You need a Raster Model to apply this filter!");
			}
			currentRaster->shot.ApplyRigidTransformation(transf);
			break;
		}
		case 1:
		{
			if (cm == NULL)
			{
				throw MLException("You need a Mesh Model to apply this filter!");
			}
			cm->shot.ApplyRigidTransformation(transf);
			break;
		}
		}
	}
		md.documentUpdated();
		break;
	case FP_CAMERA_SCALE :
	{
		Matrix44m trScale; trScale.SetIdentity();
		Point3m tranVec(0,0,0);
		Matrix44m trTran,trTranInv;
		
		Scalarm Scale= par.getFloat("scale");
		trScale.SetScale(Scale,Scale,Scale);
		
		switch(par.getEnum("scaleCenter"))
		{
		case 0: 
			tranVec=Point3m(0,0,0); 
			break;
		case 1:
		{
			switch(par.getEnum("camera"))
			{
			case 0:
				if (currentRaster == NULL)
				{
					throw MLException("You need a Raster Model to apply this filter!");
				}
				tranVec=currentRaster->shot.Extrinsics.Tra();
				break;
			case 1:
				if (cm == NULL)
				{
					throw MLException("You need a Mesh Model to apply this filter!");
				}
				tranVec=cm->shot.Extrinsics.Tra();
				break;
			}
		}
			break;
		case 2: 
			tranVec=par.getPoint3m("customCenter");
			break;
		}
		
		trTran.SetTranslate(tranVec);
		trTranInv.SetTranslate(-tranVec);
		if (par.getBool("toall")) {
			for (MeshModel& mm : md.meshIterator()) {
				if (mm.isVisible()) {
					mm.cm.Tr=trTran*trScale*trTranInv;
					tri::UpdatePosition<CMeshO>::Matrix(mm.cm, mm.cm.Tr);
					//tri::UpdateNormal<CMeshO>::PerVertexMatrix(md.meshList[i]->cm,md.meshList[i]->cm.Tr);
					//tri::UpdateNormal<CMeshO>::PerFaceMatrix(md.meshList[i]->cm,md.meshList[i]->cm.Tr);
					tri::UpdateBounding<CMeshO>::Box(mm.cm);
					mm.cm.Tr.SetIdentity();
					mm.cm.shot.ApplyRigidTransformation(trTran);
					mm.cm.shot.RescalingWorld(trScale[0][0], false);
					mm.cm.shot.ApplyRigidTransformation(trTranInv);
				}
				
			}
			for (RasterModel& rm : md.rasterIterator()) {
				if (rm.isVisible()) {
					rm.shot.ApplyRigidTransformation(trTran);
					rm.shot.RescalingWorld(trScale[0][0], false);
					rm.shot.ApplyRigidTransformation(trTranInv);
				}
			}
		}
		else if (par.getBool("toallRaster") && (par.getEnum("camera")==0)) {
			for (RasterModel& rm : md.rasterIterator()) {
				if (rm.isVisible()) {
					rm.shot.ApplyRigidTransformation(trTran);
					rm.shot.RescalingWorld(trScale[0][0], false);
					rm.shot.ApplyRigidTransformation(trTranInv);
				}
			}
		}
		else switch(par.getEnum("camera"))
		{
		case 0:
		{
			if (currentRaster == NULL) {
				throw MLException("You need a Raster Model to apply this filter!");
			}
			currentRaster->shot.ApplyRigidTransformation(trTran);
			currentRaster->shot.RescalingWorld(Scale, false);
			currentRaster->shot.ApplyRigidTransformation(trTranInv);
			break;
		}
		case 1:
		{
			if (cm == NULL) {
				throw MLException("You need a Mesh Model to apply this filter!");
			}
			cm->shot.ApplyRigidTransformation(trTran);
			cm->shot.RescalingWorld(Scale, false);
			cm->shot.ApplyRigidTransformation(trTranInv);
			break;
		}
		}
	}
		md.documentUpdated();
		break;
	case FP_CAMERA_TRANSLATE :
	{
		Matrix44m trTran; trTran.SetIdentity();
		
		Scalarm xScale= par.getDynamicFloat("axisX");
		Scalarm yScale= par.getDynamicFloat("axisY");
		Scalarm zScale= par.getDynamicFloat("axisZ");
		
		trTran.SetTranslate(xScale,yScale,zScale);
		if(par.getBool("centerFlag"))
		{
			switch(par.getEnum("camera"))
			{
			case 0:     
				if (currentRaster == NULL)
				{
					throw MLException("You need a Raster Model to apply this filter!");
				}
				trTran.SetTranslate(-currentRaster->shot.Extrinsics.Tra());
				break;
			case 1: 	
				if (cm == NULL)
				{
					throw MLException("You need a Mesh Model to apply this filter!");
				}
				trTran.SetTranslate(-cm->shot.Extrinsics.Tra());
				break;
			}
		}
		if (par.getBool("toall"))
		{
			for (MeshModel& mm : md.meshIterator())
			{
				if (mm.isVisible())
				{
					mm.cm.Tr=trTran;
					tri::UpdatePosition<CMeshO>::Matrix(mm.cm, mm.cm.Tr);
					//tri::UpdateNormal<CMeshO>::PerVertexMatrix(md.meshList[i]->cm,md.meshList[i]->cm.Tr);
					//tri::UpdateNormal<CMeshO>::PerFaceMatrix(md.meshList[i]->cm,md.meshList[i]->cm.Tr);
					tri::UpdateBounding<CMeshO>::Box(mm.cm);
					mm.cm.Tr.SetIdentity();
					mm.cm.shot.ApplyRigidTransformation(trTran);
				}
			}
			for (RasterModel& rm : md.rasterIterator()) {
				if (rm.isVisible())
					rm.shot.ApplyRigidTransformation(trTran);
				
			}
		}
		else if (par.getBool("toallRaster") && (par.getEnum("camera")==0)) {
			for (RasterModel& rm : md.rasterIterator()) {
				if (rm.isVisible())
					rm.shot.ApplyRigidTransformation(trTran);
			}
		}
		else switch(par.getEnum("camera"))
		{
		case 0:
		{
			if (currentRaster == NULL)
			{
				throw MLException("You need a Raster Model to apply this filter!");
			}
			currentRaster->shot.ApplyRigidTransformation(trTran);
			break;
		}
		case 1:
		{
			if (cm == NULL)
			{
				throw MLException("You need a Mesh Model to apply this filter!");
			}
			cm->shot.ApplyRigidTransformation(trTran);
			break;
		}
		}
	}
		md.documentUpdated();
		break;
	case FP_CAMERA_TRANSFORM :
	{
		Matrix44m mat,inv;
		inv.SetIdentity();
		Point3m tra;
		mat = par.getMatrix44("TransformMatrix");
		if(par.getEnum("behaviour") == 1)
		{
			if (currentRaster == NULL)
			{
				throw MLException("You need a Raster Model to apply this filter!");
			}
			inv = currentRaster->shot.Extrinsics.Rot();
			tra =  inv * currentRaster->shot.Extrinsics.Tra();
			inv[0][3] = -tra[0];
			inv[1][3] = -tra[1];
			inv[2][3] = -tra[2];
			mat = mat * inv;
			mat = inv;
		}
		
		if (par.getBool("toall"))
		{
			for (MeshModel& mm : md.meshIterator())
			{
				if (mm.isVisible())
				{
					mm.cm.Tr = mat;
					tri::UpdatePosition<CMeshO>::Matrix(mm.cm, mm.cm.Tr);
					//tri::UpdateNormal<CMeshO>::PerFaceMatrix(md.meshList[i]->cm,md.meshList[i]->cm.Tr);
					tri::UpdateBounding<CMeshO>::Box(mm.cm);
					mm.cm.Tr.SetIdentity();
					mm.cm.shot.ApplySimilarity(mat);
				}
			}
			for (RasterModel& rm : md.rasterIterator()) {
				if (rm.isVisible())
					rm.shot.ApplySimilarity(mat);
				
			}
		}
		else if (par.getBool("toallRaster") && (par.getEnum("camera")==0)) {
			for (RasterModel& rm : md.rasterIterator()) {
				if (rm.isVisible())
					rm.shot.ApplySimilarity(mat);
			}
		}
		else switch(par.getEnum("camera"))
		{
		case 0:
		{
			if (currentRaster == NULL)
			{
				throw MLException("You need a Raster Model to apply this filter!");
			}
			currentRaster->shot.ApplyRigidTransformation(mat);
			break;
		}
		case 1:
		{
			if (cm == NULL)
			{
				throw MLException("You need a Mesh Model to apply this filter!");
			}
			cm->shot.ApplyRigidTransformation(mat);
			break;
		}
		}
	}
		md.documentUpdated();
		break;
		
	case FP_SET_RASTER_CAMERA :
	{
		if (currentRaster == NULL)
		{
			throw MLException("You need a Raster Model to apply this filter!");
		}
		Shotm shotGot=par.getShotf("Shot");
		currentRaster->shot = shotGot;
		float ratio=(float)currentRaster->currentPlane->image.height()/(float)shotGot.Intrinsics.ViewportPx[1];
		currentRaster->shot.Intrinsics.ViewportPx[0]=currentRaster->currentPlane->image.width();
		currentRaster->shot.Intrinsics.ViewportPx[1]=currentRaster->currentPlane->image.height();
		currentRaster->shot.Intrinsics.PixelSizeMm[1]/=ratio;
		currentRaster->shot.Intrinsics.PixelSizeMm[0]/=ratio;
		currentRaster->shot.Intrinsics.CenterPx[0]=(int)((float)currentRaster->shot.Intrinsics.ViewportPx[0]/2.0);
		currentRaster->shot.Intrinsics.CenterPx[1]=(int)((float)currentRaster->shot.Intrinsics.ViewportPx[1]/2.0);
	}
		break;
	case FP_SET_MESH_CAMERA :
		if (cm == NULL)
		{
			throw MLException("You need a Mesh Model to apply this filter!");
		}
		cm->shot = par.getShotf("Shot");
		break;
	case FP_QUALITY_FROM_CAMERA :
	{
		if (cm == NULL)
		{
			throw MLException("You need a Mesh Model to apply this filter!");
		}
		if(!cm->shot.IsValid())
		{
			throw MLException("Mesh has not a valid camera");
		}
		md.mm()->updateDataMask(MeshModel::MM_VERTQUALITY + MeshModel::MM_VERTCOLOR);
		bool clipFlag = par.getBool("Clip");
		bool depthFlag = par.getBool("Depth");
		bool facingFlag = par.getBool("Facing");
		CMeshO::VertexIterator vi;
		float deltaN = cm->bbox.Diag()/100.0f;
		for(vi=cm->vert.begin();vi!=cm->vert.end();++vi)
		{
			Point2m pp = cm->shot.Project( (*vi).P());
			float depth = cm->shot.Depth((*vi).P());
			Point3m pc = cm->shot.ConvertWorldToCameraCoordinates((*vi).P());
			Point3m pn = cm->shot.ConvertWorldToCameraCoordinates((*vi).P()+(*vi).N()*deltaN);
			float q=1.0;
			
			if(depthFlag) q*=depth;
			if(facingFlag) q*=pn[2]-pc[2];
			if(clipFlag)
			{
				if(pp[0]<0 || pp[1]<0 ||
						pp[0]>cm->shot.Intrinsics.ViewportPx[0] || pp[1]>cm->shot.Intrinsics.ViewportPx[1])
					q=0;
			}
			(*vi).Q() = q;
		}
		if(par.getBool("normalize")) tri::UpdateQuality<CMeshO>::VertexNormalize(*cm);
		if(par.getBool("map")) tri::UpdateColor<CMeshO>::PerVertexQualityRamp(*cm);
		
	}
		break;
		
	case FP_ORIENT_NORMALS_WITH_CAMERAS:
	{
		if (cm == NULL)
		{
			throw MLException("You need a Mesh Model to apply this filter!");
		}
		struct Correspondence{unsigned int id_img;float padding[3];};
		typedef  vector<Correspondence> CorrVec;
		
		CMeshO::PerVertexAttributeHandle<CorrVec> ch =
				vcg::tri::Allocator<CMeshO>::FindPerVertexAttribute<CorrVec>(*cm,"correspondences");
		if(!vcg::tri::Allocator<CMeshO>::IsValidHandle<CorrVec>(*cm,ch)){
			throw MLException("Vertices have no associated camera.\n This filter works only for point clouds loaded within a Bundler project (.out)");
		}
		for(CMeshO::VertexIterator vi= cm->vert.begin(); vi != cm->vert.end();++vi) {
			unsigned int camera_id = ch[*vi][0].id_img;
			auto it = md.rasterBegin();
			std::advance(it, camera_id);
			if (it != md.rasterEnd()) {
				Point3m n=(*it).shot.GetViewPoint()-(*vi).P() ;
				if( n*(*vi).cN()<0)
					(*vi). N()=-(*vi).cN();
			}
		}
	}
		md.documentUpdated();
		break;
	default:
		wrongActionCalled(filter);
	}
	return std::map<std::string, QVariant>();
}

int FilterCameraPlugin::postCondition(const QAction * filter) const
{
	switch (ID(filter))
	{
	case FP_SET_MESH_CAMERA:
		return MeshModel::MM_CAMERA;
	case FP_CAMERA_ROTATE:
	case FP_CAMERA_TRANSLATE:
	case FP_CAMERA_TRANSFORM:
	case FP_CAMERA_SCALE:
		return MeshModel::MM_TRANSFMATRIX + MeshModel::MM_VERTCOORD + MeshModel::MM_VERTNORMAL + MeshModel::MM_FACENORMAL;
	case FP_QUALITY_FROM_CAMERA:
		return MeshModel::MM_VERTQUALITY + MeshModel::MM_VERTCOLOR;
	default: return MeshModel::MM_ALL;
	}
}

FilterCameraPlugin::FilterClass FilterCameraPlugin::getClass(const QAction *a) const
{
	switch(ID(a))
	{
	case FP_CAMERA_ROTATE :
	case FP_CAMERA_SCALE :
	case FP_CAMERA_TRANSLATE :
	case FP_CAMERA_TRANSFORM:
	case FP_SET_RASTER_CAMERA :
		return FilterClass (FilterPlugin::Camera + FilterPlugin::RasterLayer) ;
	case FP_QUALITY_FROM_CAMERA:
		return FilterClass(FilterPlugin::Camera + FilterPlugin::RasterLayer + FilterPlugin::Quality);
	case FP_ORIENT_NORMALS_WITH_CAMERAS:
		return FilterClass(FilterPlugin::Camera + FilterPlugin::Normal);
	case FP_SET_MESH_CAMERA:
		return FilterClass(FilterPlugin::Camera + FilterPlugin::Layer);
	}
	assert(0);
	return FilterPlugin::Camera;
}

int FilterCameraPlugin::getPreConditions(const QAction * a) const
{
	switch (ID(a))
	{
	case FP_CAMERA_ROTATE:
	case FP_CAMERA_SCALE:
	case FP_CAMERA_TRANSLATE:
	case FP_CAMERA_TRANSFORM:
	case FP_QUALITY_FROM_CAMERA:
	case FP_SET_RASTER_CAMERA:
	case FP_SET_MESH_CAMERA:
		return MeshModel::MM_NONE;
	case FP_ORIENT_NORMALS_WITH_CAMERAS:
		return MeshModel::MM_VERTNORMAL;
	}
	assert(0);
	return 0;
}

FilterPlugin::FilterArity FilterCameraPlugin::filterArity(const QAction* act ) const
{
	switch (ID(act))
	{
	case FP_CAMERA_ROTATE:
	case FP_CAMERA_SCALE:
	case FP_CAMERA_TRANSLATE:
	case FP_CAMERA_TRANSFORM:
	case FP_QUALITY_FROM_CAMERA:
	case FP_SET_RASTER_CAMERA:
	case FP_SET_MESH_CAMERA:
	case FP_ORIENT_NORMALS_WITH_CAMERAS:
		return SINGLE_MESH;
	}
	return SINGLE_MESH;
}

MESHLAB_PLUGIN_NAME_EXPORTER(FilterCameraPlugin)

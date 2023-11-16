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

#include "VisibleSet.h"
#include "VisibilityCheck.h"
#include <wrap/gl/shot.h>
#include <cmath>




VisibleSet::VisibleSet(glw::Context &ctx,
		MLPluginGLContext* plugctx,
		int meshid,
		CMeshO &mesh,
		std::list<RasterModel*>& rasterList,
		int weightMask) :
	m_Mesh(mesh),
	m_FaceVis(mesh.fn),
	m_WeightMask(weightMask)
{
	VisibilityCheck &visibility = *VisibilityCheck::GetInstance( ctx );
	visibility.setMesh(meshid,&mesh );
	visibility.m_plugcontext = plugctx;


	float depthMin =  std::numeric_limits<float>::max();
	m_DepthMax = -std::numeric_limits<float>::max();

	for(RasterModel *rm: rasterList) {
		CMeshO::ScalarType zNear, zFar;
		GlShot< Shotm >::GetNearFarPlanes( rm->shot, mesh.bbox, zNear, zFar );

		if( zNear < depthMin )
			depthMin = zNear;
		if( zFar > m_DepthMax )
			m_DepthMax = zFar;
	}

	if( depthMin < 0.0001f )
		depthMin = 0.1f;
	if( m_DepthMax < depthMin )
		m_DepthMax = depthMin + 1000.0f;

	m_DepthRangeInv = 1.0f / (m_DepthMax-depthMin);


	for( RasterModel *rm : rasterList)
	{
		visibility.setRaster( rm );
		visibility.checkVisibility();

		for( int f=0; f<mesh.fn; ++f ){
			if( visibility.isFaceVisible(f) ) {
				float w = getWeight( rm, mesh.face[f] );
				if( w >= 0.0f )
					m_FaceVis[f].add( w, rm );
			}
		}
	}

	VisibilityCheck::ReleaseInstance();
}


float VisibleSet::getWeight( const RasterModel *rm, CFaceO &f )
{
    Point3m centroid = (f.V(0)->P() +
                             f.V(1)->P() +
                             f.V(2)->P()) / 3.0f;

    float weight = 1.0f;

    if( m_WeightMask & W_ORIENTATION )
      weight *= (rm->shot.GetViewPoint()-centroid).Normalize() * f.N();

    if( (m_WeightMask & W_DISTANCE) && weight>0.0f )
    {
        weight *= (m_DepthMax - (rm->shot.GetViewPoint()-centroid).Norm()) * m_DepthRangeInv;
    }

    if( (m_WeightMask & W_IMG_BORDER) && weight>0.0f )
    {
        Point2m cam = rm->shot.Project( centroid );
        weight *= 1.0f - std::max( std::abs(2.0f*cam.X()/rm->shot.Intrinsics.ViewportPx.X()-1.0f),
                                   std::abs(2.0f*cam.Y()/rm->shot.Intrinsics.ViewportPx.Y()-1.0f) );
    }

    if( (m_WeightMask & W_IMG_ALPHA) && weight>0.0f )
    {
        float alpha[3];
        for(int i=0;i<3;++i)
        {
          Point2m ppoint = rm->shot.Project( f.V(i)->P() );
          if(ppoint[0] < 0 ||
             ppoint[1] < 0 ||
             ppoint[0] >= rm->currentPlane->image.width() ||
             ppoint[1] >= rm->currentPlane->image.height())
            alpha[i] = 0;
          else
            alpha[i] = qAlpha(rm->currentPlane->image.pixel(ppoint[0],rm->shot.Intrinsics.ViewportPx[1] - ppoint[1]));
        }

        int minAlpha = vcg::math::Min(alpha[0],alpha[1],alpha[2]);
        float aweight = float(minAlpha)/255.0f;

        // if alpha weight is zero, that image part should not be used at all,
        // so, we set the weight below zero in order to force the visibility check to fail
        // just setting weight to zero would result passing visibility check with a 0 weight,
        // making the piece usable if nothing else is available
        if(aweight == 0.0)
          weight = -1.0;
        else
          weight *= aweight;
    }

    return weight;
}

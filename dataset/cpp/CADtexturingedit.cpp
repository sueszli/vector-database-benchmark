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
$Log: meshedit.cpp,v $
****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <meshlab/glarea.h>
#include "CADtexturingedit.h"
//#include <wrap/gl/pick.h>
#include<wrap/qt/gl_label.h>
#include<wrap/gl/shot.h>
#include <wrap/glw/glw.h>
#include<vcg/complex/algorithms/update/flag.h>
#include "image_edge_detection.h"

using namespace std;
using namespace vcg;
using namespace glw;

//struct EVertType;
//struct EEdgeType;
//
//struct UT : public vcg::UsedTypes<vcg::Use<EVertType>::AsVertexType, vcg::Use<EEdgeType   >::AsEdgeType >{};
//class EVertType : public vcg::Vertex< UT,
//	vcg::vertex::InfoOcf,           /*  4b */
//	vcg::vertex::Coord3m,           /* 12b */
//	vcg::vertex::BitFlags,          /*  4b */
//	vcg::vertex::Normal3m,          /* 12b */
//	vcg::vertex::Qualityf,          /*  4b */
//	vcg::vertex::Color4b,           /*  4b */
//	vcg::vertex::VEAdj,          /*  0b */
//	vcg::vertex::MarkOcf,           /*  0b */
//	vcg::vertex::TexCoordfOcf      /*  0b */
//>{
//};
//
//class EEdgeType : public vcg::Edge<UT,
//	vcg::edge::BitFlags,          /*  4b */
//	vcg::edge::EVAdj,
//	vcg::edge::EEAdj
//>{
//};
//
//class EMesh : public vcg::tri::TriMesh< vcg::vertex::vector_ocf<EVertType>, std::vector<EEdgeType> >
//{};


struct EdgeEval{
	EdgeEval(){};
	const bool operator()(CMeshO::FaceType & f, int e){
		return !f.IsF(e);
	}

};

template < class EEVal>
void AddMeshEdges(CMeshO & m, EEVal e){
	for (CMeshO::FaceIterator fi = m.face.begin(); fi != m.face.end(); ++fi) if (!(*fi).IsD())
	for (int i = 0; i < (*fi).VN(); ++i) if (e(*fi, i))
		vcg::tri::Allocator<CMeshO>::AddEdge(m, (*fi).V(i), (*fi).V((i + 1) % (*fi).VN()));
}



void FaceFauxSignedSilouette(CMeshO &m, const vcg::Matrix44f &mvpm)
{
	std::vector<CMeshO::CoordType> opos;
	opos.resize(m.vn);
	int i = 0;
	for (CMeshO::VertexIterator vi = m.vert.begin(); vi != m.vert.end(); ++vi) if (!((*vi).IsD())){
		opos[i++] = (*vi).P();
		vcg::Point4f p4 = vcg::Point4f((*vi).cP()[0], (*vi).cP()[1], (*vi).cP()[2], 1.0);
		p4 = mvpm*p4;
		p4 = p4 / p4[3];
		(*vi).P() = *(CMeshO::CoordType*)&p4[0];
	}

	// Then mark faux only if the signed angle is the range.
	for (CMeshO::FaceIterator fi = m.face.begin(); fi != m.face.end(); ++fi) if (!(*fi).IsD())
	{
		for (int z = 0; z<(*fi).VN(); ++z)
		{
			if (!face::IsBorder(*fi, z))
			{
				CMeshO::CoordType N0 = vcg::Normal((*fi).P(0), (*fi).P(1), (*fi).P(2) );
				CMeshO::CoordType N1 = vcg::Normal((*fi).FFp(z)->P(0), (*fi).FFp(z)->P(1), (*fi).FFp(z)->P(2));

				if ( N0.Z()>0 ^ N1.Z() > 0 )
					(*fi).ClearF(z);
			}
		}
	}
	i = 0;
	for (CMeshO::VertexIterator vi = m.vert.begin(); vi != m.vert.end(); ++vi) if (!((*vi).IsD())){
		(*vi).P() = opos[i++];
	}

}

CADtexturingEditPlugin::CADtexturingEditPlugin() 
{
	//qFont.setFamily("Helvetica");
	//qFont.setPixelSize(12);

	//haveToPick = false;
	//pickmode = 0; // 0 face 1 vertex
	//curFacePtr = 0;
	//curVertPtr = 0;
	//pIndex = 0;
}

const QString CADtexturingEditPlugin::Info() 
{
	return tr("Return detailed info about a picked face or vertex of the model.");
}
 
void CADtexturingEditPlugin::mouseReleaseEvent(QMouseEvent * event, MeshModel &/*m*/, GLArea * gla)
{
	gla->update();
	//cur = event->pos();
	//haveToPick = true;
	//curFacePtr = 0;
	//curVertPtr = 0;
	//pIndex = 0;
}


void CADtexturingEditPlugin::ComputeNearFar(const vcg::Shotf &  s,float & nearplane, float & farplane )
{
	float sx, dx, bt, tp,nr;
	s.Intrinsics.GetFrustum(sx, dx, bt, tp, nr);
	GLfloat fAspect = fabs((dx - sx) / (tp - bt));

	vcg::Box3f b = this->meshmodel->cm.bbox;
	vcg::Box3f bb;
	for (int i = 0; i < 8; ++i) bb.Add(s.ConvertWorldToCameraCoordinates(b.P(i)));
	nearplane = bb.min.Z();
	farplane = bb.max.Z();
}



bool first;

void CADtexturingEditPlugin::renderEdges(GLArea * gla){



	//glarea = gla;
	if (gla->mvc() == NULL)
		return;
	MLSceneGLSharedDataContext* shared = gla->mvc()->sharedDataContext();
	if (shared == NULL)
		return;

	MLSceneGLSharedDataContext::PerMeshRenderingDataMap dt;
	vcg::Point2i cp;
	Context ctx;
	RenderbufferHandle hDepth;
	Texture2DHandle    hColor;
	FramebufferHandle  hFramebuffer;
	GLint vp[4];
	QImage image;
	QImage depth;

	if (saveRenderingTrigger)
	{
		cp = gla->mvc()->meshDoc.rm()->shot.Intrinsics.ViewportPx;
		ctx.acquire();

		hDepth = createRenderbuffer(ctx, GL_DEPTH_COMPONENT24, cp[0], cp[1]);
		hColor = createTexture2D(ctx, GL_RGBA8, cp[0], cp[1], GL_RGBA, GL_UNSIGNED_BYTE);
		hFramebuffer = createFramebuffer(ctx, renderbufferTarget(hDepth), texture2DTarget(hColor));


		glGetIntegerv(GL_VIEWPORT, vp);

		glViewport(0, 0, cp[0], cp[1]);

		ctx.bindReadDrawFramebuffer(hFramebuffer);
		GLW_CHECK_GL_READ_DRAW_FRAMEBUFFER_STATUS;

		shared->getRenderInfoPerMeshView(gla->context(), dt);


		image = QImage(int(cp[0]), int(cp[1]), QImage::Format_ARGB32);
		depth = QImage(int(cp[0]), int(cp[1]), QImage::Format_Grayscale8);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (saveRenderingTrigger){
		glViewport(0, 0, cp[0], cp[1]);
		glPushMatrix();


		/* set camera*/
		float np, fp;
		ComputeNearFar(gla->mvc()->meshDoc.rm()->shot, np, fp);
		GlShot<Shotm>::SetView(gla->mvc()->meshDoc.rm()->shot, (np>0.01)?np:0.01, fp);
	}
	/**/


	{
		MLSceneGLSharedDataContext* datacont = gla->mvc()->sharedDataContext();
		if (datacont == NULL)
			return;

		foreach(MeshModel * mp, gla->md()->meshList){
			MLRenderingData curr;
			datacont->getRenderInfoPerMeshView(mp->id(), gla->context(), curr);
			MLPerViewGLOptions opts;
			curr.get(opts);
			//if (curr.get(opts) == false)
			//	throw MLException(QString("GLArea: invalid MLPerViewGLOptions"));
			//gla->setLightingColors(opts);


			glEnable(GL_CULL_FACE);
			datacont->setMeshTransformationMatrix(mp->id(), mp->cm.Tr);

			if (first ){
				vcg::Matrix44f mvpm, pm,mv4;
				glGetFloatv(GL_PROJECTION_MATRIX, &pm[0][0]);
				pm.transposeInPlace();

				glGetFloatv(GL_MODELVIEW_MATRIX, &mvpm[0][0]);
				mvpm.transposeInPlace();

				FaceFauxSignedSilouette(mp->cm, pm* mvpm);
				first = false;
			}


			if (mp->cm.fn){
				glPolygonMode(GL_FRONT, GL_FILL);
				glDisable(GL_LIGHTING);

				glCullFace(GL_BACK);
				glEnable(GL_POLYGON_OFFSET_FILL);
				glPolygonOffset(1.0, 1);
				drawer.DrawFill<vcg::GLW::NormalMode::NMNone, vcg::GLW::ColorMode::CMPerMesh, vcg::GLW::TextureMode::TMNone>();
				
			 	glCullFace(GL_BACK);
				glDisable(GL_POLYGON_OFFSET_FILL);
				glColor3f(1.0, 0.0, 0.0);
				drawer.DrawWirePolygonal<vcg::GLW::NormalMode::NMNone, vcg::GLW::ColorMode::CMNone>();
				glEnable(GL_LIGHTING);
			}
		}
	}
	if(saveRenderingTrigger){

		GlShot<Shotm>::UnsetView();

		glReadPixels(0, 0,  cp[0],  cp[1], GL_RGBA, GL_UNSIGNED_BYTE, image.bits());
		glReadPixels(0, 0, cp[0], cp[1], GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, depth.bits());

		image.mirrored().save("crease_angles.jpg");
		//depth.save("depth.jpg");

		ctx.unbindReadDrawFramebuffer();
		ctx.release();
		
		glViewport(vp[0], vp[1], vp[2], vp[3]);

		/*save edges of the raster*/
		Plane * pl = gla->mvc()->meshDoc.rm()->currentPlane;

		if (!pl->IsInCore()) pl->Load();

		detect_edges(pl->image,"imageedges.jpg");
		/**************************/

		saveRenderingTrigger = false;

	}

	return;

}
void CADtexturingEditPlugin::on_renderEdges(int state){
	drawEdgesTrigger = (state == Qt::Checked);
	glarea->update();
}

void CADtexturingEditPlugin::on_saverendering(){
	saveRenderingTrigger = true;
	glarea->update();
}

void CADtexturingEditPlugin::Decorate(MeshModel &m, GLArea * gla, QPainter *p)
{
	if (drawEdgesTrigger || saveRenderingTrigger){
		renderEdges(gla);
	}
}
void CADtexturingEditPlugin::keyReleaseEvent(QKeyEvent *e, MeshModel &m, GLArea *gla)
{
	//if (e->key() == Qt::Key_Space) // toggle pick mode
	//{
	//	pickmode = (pickmode + 1) % 2;
	//	curFacePtr = 0;
	//	curVertPtr = 0;
	//	pIndex = 0;
	//	gla->update();
	//}

	//if ((e->key() == Qt::Key_Q) && ((curFacePtr != 0) || (curVertPtr != 0))) // tab to next selected mode
	//{
	//	if (pickmode == 0)
	//	{
	//		pIndex = (pIndex + 1) % NewFaceSel.size();
	//		curFacePtr = NewFaceSel[pIndex];
	//	}
	//	else if (pickmode == 1)
	//	{
	//		pIndex = (pIndex + 1) % NewVertSel.size();
	//		curVertPtr = NewVertSel[pIndex];
	//	}
	//	gla->update();
	//}

	//if ((e->key() == Qt::Key_P) && ((curFacePtr != 0) || (curVertPtr != 0))) // tab to next selected mode
	//{
	//	// print on log
	//	if (pickmode == 0)
	//	{
	//		this->Log(GLLogStream::FILTER, "------");
	//		this->Log(GLLogStream::FILTER, "face# %i : vert# (%i %i %i)", tri::Index(m.cm, curFacePtr), tri::Index(m.cm, curFacePtr->V(0)), tri::Index(m.cm, curFacePtr->V(1)), tri::Index(m.cm, curFacePtr->V(2)));
	//		if (m.hasDataMask(MeshModel::MM_FACECOLOR))
	//			this->Log(GLLogStream::FILTER, "face color (%i %i %i %i)", curFacePtr->C()[0], curFacePtr->C()[1], curFacePtr->C()[2], curFacePtr->C()[3]);
	//		for (int i = 0; i < 3; ++i)
	//		{
	//			this->Log(GLLogStream::FILTER, "face vert %i : vert# %i", i, tri::Index(m.cm, curFacePtr->V(i)));
	//			this->Log(GLLogStream::FILTER, "position [%f %f %f]", curFacePtr->V(i)->P()[0], curFacePtr->V(i)->P()[1], curFacePtr->V(i)->P()[2]);
	//			this->Log(GLLogStream::FILTER, "normal [%f %f %f]", curFacePtr->V(i)->N()[0], curFacePtr->V(i)->N()[1], curFacePtr->V(i)->N()[2]);
	//			if (m.hasDataMask(MeshModel::MM_VERTQUALITY))
	//				this->Log(GLLogStream::FILTER, "quality %f", curFacePtr->V(i)->Q());
	//			if (m.hasDataMask(MeshModel::MM_VERTCOLOR))
	//				this->Log(GLLogStream::FILTER, "color (%f %f %f %f)", curFacePtr->V(i)->C()[0], curFacePtr->V(i)->C()[1], curFacePtr->V(i)->C()[2], curFacePtr->V(i)->C()[3]);
	//			if (m.hasDataMask(MeshModel::MM_WEDGTEXCOORD))
	//				this->Log(GLLogStream::FILTER, "wedge UV (%f %f) texID %i)", curFacePtr->WT(i).U(), curFacePtr->WT(i).V(), curFacePtr->WT(i).N());
	//			if (m.hasDataMask(MeshModel::MM_VERTTEXCOORD))
	//				this->Log(GLLogStream::FILTER, "vertex UV (%f %f) texID %i)", curFacePtr->V(i)->T().U(), curFacePtr->V(i)->T().U(), curFacePtr->V(i)->T().N());
	//		}
	//		this->Log(GLLogStream::FILTER, "------");
	//	}
	//	else if (pickmode == 1)
	//	{
	//		this->Log(GLLogStream::FILTER, "------");
	//		this->Log(GLLogStream::FILTER, "vertex# %i", tri::Index(m.cm, curVertPtr));
	//		this->Log(GLLogStream::FILTER, "position [%f %f %f]", curVertPtr->P()[0], curVertPtr->P()[1], curVertPtr->P()[2]);
	//		this->Log(GLLogStream::FILTER, "normal [%f %f %f]", curVertPtr->N()[0], curVertPtr->N()[1], curVertPtr->N()[2]);
	//		if (m.hasDataMask(MeshModel::MM_VERTQUALITY))
	//			this->Log(GLLogStream::FILTER, "quality %f", curVertPtr->Q());
	//		if (m.hasDataMask(MeshModel::MM_VERTCOLOR))
	//			this->Log(GLLogStream::FILTER, "color (%f %f %f %f)", curVertPtr->C()[0], curVertPtr->C()[1], curVertPtr->C()[2], curVertPtr->C()[3]);
	//		if (m.hasDataMask(MeshModel::MM_VERTTEXCOORD))
	//			this->Log(GLLogStream::FILTER, "vertex UV (%f %f) texID %i)", curVertPtr->T().U(), curVertPtr->T().U(), curVertPtr->T().N());
	//		this->Log(GLLogStream::FILTER, "------");
	//	}
	//	gla->update();
	//}
}



bool CADtexturingEditPlugin::StartEdit(MeshModel & m , GLArea * gla, MLSceneGLSharedDataContext* /*cont*/)
{
	GLenum err = glewInit();
	if (err != GLEW_OK)
		return false;

	gla->setCursor(QCursor(QPixmap(":/images/cur_info.png"),1,1));	
	connect(this, SIGNAL(suspendEditToggle()), gla, SLOT(suspendEditToggle()));

	dock = new QDockWidget(gla->window());
	control = new CADtexturingControl(dock);
	meshmodel = &m;
	glarea = gla;
	dock->setAllowedAreas(Qt::NoDockWidgetArea);
	dock->setWidget(control);
	QPoint p = gla->mapToGlobal(QPoint(0, 0));
	dock->setGeometry(5 + p.x(), p.y() + 5, control->width(), control->height() );
	dock->setFloating(true);
	dock->setVisible(true);

	QObject::connect(control, SIGNAL(saverenderingClicked()), this, SLOT(on_saverendering()));
	QObject::connect(control, SIGNAL(renderedgesChanged(int)),	this, SLOT(on_renderEdges(int)));
	drawEdgesTrigger = false;
	saveRenderingTrigger = false;

	m.updateDataMask(MeshModel::MM_FACEFACETOPO);
	vcg::tri::UpdateFlags<CMeshO>::FaceClearF(m.cm);
	vcg::tri::UpdateTopology<CMeshO>::FaceFace(m.cm);	
	vcg::tri::UpdateFlags<CMeshO>::FaceFauxSignedCrease(m.cm, -M_PI*0.5, M_PI*0.5);

	AddMeshEdges(m.cm,EdgeEval());
	vcg::tri::UpdateNormal<CMeshO>::PerFace(m.cm);

	drawer.m = &m.cm;
	m.cm.C().SetGrayShade(0.5);
	first = true;
	return true;
}

void CADtexturingEditPlugin::EndEdit(MeshModel &/*m*/, GLArea * /*parent*/, MLSceneGLSharedDataContext* /*cont*/){

	dock->setVisible(false);
	delete control;
	delete dock;

};

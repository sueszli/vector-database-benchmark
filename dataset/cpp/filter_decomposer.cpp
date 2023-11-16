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
#include <math.h>
#include "filter_decomposer.h"
#include <QtScript>
#include <vcg/complex/algorithms/geodesic.h>
#include <vcg/simplex/face/topology.h>
#include "graph_builder.h"

class CutEdge;
class CutFace;
class CutVertex;
struct CutUsedTypes : public vcg::UsedTypes<	vcg::Use<CutVertex>   ::AsVertexType,
        vcg::Use<CutEdge>     ::AsEdgeType,
        vcg::Use<CutFace>     ::AsFaceType>{};

class CutVertex  : public vcg::Vertex<CutUsedTypes, vcg::vertex::Coord3f, vcg::vertex::VFAdj, vcg::vertex::Qualityf, vcg::vertex::Color4b, vcg::vertex::BitFlags  >{};
class CutFace    : public vcg::Face<CutUsedTypes, vcg::face::VFAdj, vcg::face::VertexRef, vcg::face::FFAdj, vcg::face::FEAdj, vcg::face::Qualityf, vcg::face::Normal3f, vcg::face::Color4b, vcg::face::BitFlags, vcg::face::Mark > {};
class CutEdge    : public vcg::Edge<CutUsedTypes>{};
class CutMesh    : public vcg::tri::TriMesh< std::vector<CutVertex>, std::vector<CutFace> , std::vector<CutEdge>  > {};
typedef vcg::GridStaticPtr<CutFace, CutMesh::ScalarType> TriMeshGrid;


template <class MeshType>
class WeightFunctor
{
    typedef typename MeshType::FaceType FaceType;
    typedef typename MeshType::FacePointer FacePointer;

private: float _ambient_weight, _dihedral_weight, _elength_weight, _geodesic_weight, _max, _maxq, _geomin, _geomax;
    FacePointer _start,_end;
    typename MeshType:: template PerFaceAttributeHandle<float> _gfH1;
    double _totmax;
public:
    WeightFunctor(float aow, float dw, float geodesic, float elength, FacePointer start, FacePointer end,  typename MeshType:: template PerFaceAttributeHandle<float> gfH1, float max, float geomin, float geomax, float maxq)
    {
        if(max<0 || maxq<0 || elength<0 || start==NULL ||
                end==NULL ||aow<0 || dw<0 || geodesic <0 || geomin<0 || geomax<0)
            assert(0);

        _max = max;
        _maxq = maxq;
        _geomin = geomin;
        _geomax = geomax;
        _gfH1 = gfH1;
        _start = start;
        _end = end;
        _elength_weight = elength;
        _ambient_weight = aow;
        _dihedral_weight = dw;
        _geodesic_weight = geodesic;
        qDebug("_max1 %f",_max);
        qDebug("_geomin %f _geomax %f",_geomin,_geomax);
        qDebug("_elength_weight %f",_elength_weight);
        qDebug("_ambient_weight %f",_ambient_weight);
        qDebug("_dihedral_weight %f",_dihedral_weight);
        qDebug("_geodesic_weight %f",_geodesic_weight);
    }

    /* Computes the weight W of the edge f1->f2 using the float parameters as coefficients for the expression:
     * W = alpha * (f1->Q()+f2->Q())/2 + beta * f1\/f2 + delta * f1<-->start/end

     * f1= pos.f ; f2= pos.FFlip ;
     * f1->Q() = ambient occlusion value for the face f1
     ** The final weight of the edge should be lower for faces with higher ambient occlusion value.
     * f1\/f2 = dihedral angle between face f1 and f2: concave edges are priviledged.
     ** The final weight of the edge should be higher for convex edges and lower for concave ones.
     * f1<-->start = geodesic distance "dist" from the current face to the start face chosen by the user.
                     Using the "geomin" and "geomax" coefficients, edges that are at a distance
                     lower than geomin*dist or higher geomax*dist are given a higher weight
                     ..this will prevent trivial cuts
     ** weighted also on the length of the edge shared by f1 and f2.
    */
    double operator() (const vcg::face::Pos<CutFace> pos)
    {
        //sub-weights are normalized using previously computed max values
        return ((_elength_weight * vcg::Norm(pos.V()->P() - pos.VFlip()->P()))
                + (_geodesic_weight * GeodesicDistance(pos))
                + (_ambient_weight * (pos.f->Q() + pos.FFlip()->Q())/(2.0f*_maxq))
                + (_dihedral_weight * ( (M_PI + vcg::face::DihedralAngleRad<CutFace>(*(pos.f), pos.E())) / (2*M_PI))) );
    }

    double GeodesicDistance(const vcg::face::Pos<CutFace> pos)
    {
        //if the current face is too close to either the startFace or the endFace
        //float::max is returned, which makes sure the final weight will be higher
        //and the edge won't be considered for a minimum cut.
        //this is useful to avoid trivial cuts.
        float dist = (_gfH1[pos.f] + _gfH1[pos.FFlip()])/2.0f;

        assert(dist<=_max);

        if((dist  <= _gfH1[_end]*_geomin || dist >= _gfH1[_end]*_geomax))
            return std::numeric_limits<float>::max();
        return    dist/_max;
    }
};


// Constructor usually performs only two simple tasks of filling the two lists
//  - typeList: with all the possible id of the filtering actions
//  - actionList with the corresponding actions. If you want to add icons to your filtering actions you can do here by construction the QActions accordingly
ExtraSamplePlugin::ExtraSamplePlugin()
{
    typeList << FP_DECOMPOSER;

    foreach(FilterIDType tt , types())
        actionList << new QAction(filterName(tt), this);
}

// ST() must return the very short string describing each filtering action
// (this string is used also to define the menu entry)
QString ExtraSamplePlugin::filterName(FilterIDType filterId) const
{
    switch(filterId) {
    case FP_DECOMPOSER : return QString("Decompose mesh");
    default : assert(0);
    }
    return QString();
}

// Info() must return the longer string describing each filtering action
// (this string is used in the About plugin dialog)
QString ExtraSamplePlugin::filterInfo(FilterIDType filterId) const
{
    switch(filterId) {
    case FP_DECOMPOSER : return QString("Mesh segmentation in two sub parts using a maxflow/mincut approach. Kolmogorov's maxflow algorithm is applied on the mesh dual graph.\n\nFor more details see:\nYuri Boykov and Vladimir Kolmogorov,'An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision.\nIn IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), September 2004");
    default : assert(0);
    }
    return QString("Unknown Filter");
}


// The FilterClass describes in which generic class of filters it fits.
// This choice affect the submenu in which each filter will be placed
// More than a single class can be choosen.
ExtraSamplePlugin::FilterClass ExtraSamplePlugin::getClass(QAction *a)
{
    switch(ID(a))
    {
    case FP_DECOMPOSER : return MeshFilterInterface::Remeshing;
    default : assert(0);
    }
    return MeshFilterInterface::Generic;
}

// This function define the needed parameters for each filter. Return true if the filter has some parameters
// it is called every time, so you can set the default value of parameters according to the mesh
// For each parameter you need to define,
// - the name of the parameter,
// - the string shown in the dialog
// - the default value
// - a possibly long string describing the meaning of that parameter (shown as a popup help in the dialog)

//farsi passare due Point3f sulla mesh dai quali risalire alle facce sorgente/pozzo
void ExtraSamplePlugin::initParameterSet(QAction *action,MeshModel &m, RichParameterSet & parlst)
{
    switch(ID(action))	 {
    case FP_DECOMPOSER :
        parlst.addParam(new RichPoint3f("upperPoint",m.cm.bbox.min,"Start Surf. Pos","This face will be connected to the SOURCE in the graph over which the mincut algorithm will be applied."));
        parlst.addParam(new RichPoint3f("lowerPoint",m.cm.bbox.min,"End Surf. Pos","This face will be connected to the SINK in the graph over which the mincut algorithm will be applied."));
        parlst.addParam(new RichDynamicFloat("geodesic", 0.2f, 0.0f, 1.0f,"Geodesic Weight Factor","Factor to model the geodesic distance impact on the edge weight: The geodesic distance is inteded from a certain face to the Start or End faces."));
        parlst.addParam(new RichDynamicFloat("geo-factor-min", 0.25f, 0.0f, 1.0f,"Min Dist. Factor","Factor to model the desidered minimum distance of the cut from the Start pos."));
        parlst.addParam(new RichDynamicFloat("geo-factor-max", 0.75f, 0.0f, 1.0f,"Max Dist. Factor","Factor to model the desidered maximum distance of the cut from the Start pos."));
        parlst.addParam(new RichDynamicFloat("dihedral", 0.2f, 0.0f, 1.0f,"Dihedral Weight Factor","Factor to model the dihedral angle impact on the edge weight: Concave angles between faces will be privileged."));
        parlst.addParam(new RichDynamicFloat("ambient", 0.2f, 0.0f, 1.0f,"Ambient Weight Factor","Factor to model the ambient occlusion impact on the edge weight: Faces with low value of ambient occlusion will be privileged."));
        parlst.addParam(new RichDynamicFloat("e-length", 0.2f, 0.0f, 1.0f,"Edge Length Factor","Factor to model the edge length impact on the edge weight: shorter edges will be privileged."));
        break;
    default : assert(0);
    }
}

// The Real Core Function doing the actual mesh processing.
// Move Vertex of a random quantity
bool ExtraSamplePlugin::applyFilter(QAction *action, MeshDocument &md, RichParameterSet & par, vcg::CallBackPos *cb)
{
    switch(ID(action))	 {

    case FP_DECOMPOSER :
    {
        CMeshO &m = md.mm()->cm;
        CutMesh cm;
        vcg::tri::Append<CutMesh,CMeshO>::MeshCopy(cm,m);
        vcg::tri::Allocator<CutMesh>::CompactEveryVector(cm);
        vcg::tri::UpdateTopology<CutMesh>::FaceFace(cm);
        vcg::tri::UpdateTopology<CutMesh>::VertexFace(cm);
        vcg::tri::UpdateNormal<CutMesh>::PerFace(cm);

        //Get Parameters
        vcg::Point3f upperPoint = par.getPoint3f("upperPoint");
        vcg::Point3f lowerPoint = par.getPoint3f("lowerPoint");
        float dihedral = par.getDynamicFloat("dihedral");
        float ambient = par.getDynamicFloat("ambient");
        float geodesic = par.getDynamicFloat("geodesic");
        float geomin = par.getDynamicFloat("geo-factor-min");
        float geomax = par.getDynamicFloat("geo-factor-max");
        float elength = par.getDynamicFloat("e-length");
        CutMesh::FacePointer startFace = NULL;
        CutMesh::FacePointer endFace = NULL;



        //getting the closest faces to the passed point parameters
        TriMeshGrid grid;
        grid.Set(cm.face.begin(),cm.face.end());
        vcg::Point3<CutMesh::ScalarType> closest;
        float maxDist = cm.bbox.Diag();
        float minDist;
        startFace = vcg::tri::GetClosestFaceBase(cm, grid, upperPoint, maxDist, minDist, closest);
        endFace = vcg::tri::GetClosestFaceBase(cm, grid, lowerPoint, maxDist, minDist, closest);

        assert(startFace!=NULL && endFace!=NULL);

        std::vector<CutVertex::VertexPointer> seedVec;
        for(int i=0; i<3; i++)
            seedVec.push_back(startFace->V(i));

        //getting the geodesic distance from the start/end face and some other data
        //that will be used in the weight computation
        typename  CutMesh::template PerFaceAttributeHandle <float>  gfH1
                = vcg::tri::Allocator<CutMesh>::template AddPerFaceAttribute<float>(cm, std::string("GeoDis1"));
        vcg::tri::Geodesic<CutMesh>::Compute(cm, seedVec);

        float max1=0.0f;
        float maxq=0.0f;
        //getting also the maximum perFaceQuality value
        for(CutMesh::FaceIterator fit= cm.face.begin(); fit!=cm.face.end(); fit++){
            if(fit->Q()>=maxq)
                maxq=fit->Q();
            gfH1[fit] = (fit->V(0)->Q()+fit->V(1)->Q()+fit->V(2)->Q())/3.0f;
            if(gfH1[fit]>=max1)
                max1=gfH1[fit];
        }


        //this functor will be used to compute edge weights on the dual graph
        WeightFunctor<CutMesh> wf(ambient, dihedral, geodesic, elength, startFace, endFace, gfH1, max1, geomin, geomax, maxq);
        //start actual computation
        vcg::tri::GraphBuilder<CutMesh, WeightFunctor<CutMesh> >::Compute(cm, wf, startFace, endFace);
        vcg::tri::Allocator<CutMesh>::DeletePerFaceAttribute(cm,gfH1);
        vcg::tri::Append<CMeshO,CutMesh>::MeshCopy(m,cm);
        return true;
    }
    }
    return false;
}


QString ExtraSamplePlugin::filterScriptFunctionName( FilterIDType filterID )
{
    switch(filterID) {
    case FP_DECOMPOSER : return QString("meshDecomposer");
    default : assert(0);
    }
    return QString();
}

Q_EXPORT_PLUGIN(ExtraSamplePlugin)

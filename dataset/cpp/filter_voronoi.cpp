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

#include "filter_voronoi.h"

#include<vcg/complex/algorithms/voronoi_processing.h>
#include<vcg/complex/algorithms/update/curvature.h>
#include<vcg/complex/algorithms/update/curvature_fitting.h>
#include<vcg/complex/algorithms/update/quality.h>
#include<vcg/complex/algorithms/smooth.h>
#include<vcg/complex/algorithms/voronoi_volume_sampling.h>
#include<vcg/complex/algorithms/polygon_support.h>

using namespace vcg;
using namespace vcg::tri;

FilterVoronoiPlugin::FilterVoronoiPlugin()
{ 
	typeList = {
		VORONOI_SAMPLING,
		VOLUME_SAMPLING,
		VORONOI_SCAFFOLDING,
		BUILD_SHELL,
//		CROSS_FIELD_CREATION,
//		CROSS_FIELD_SMOOTHING,
	};

	for (ActionIDType tt : types())
		actionList.push_back(new QAction(filterName(tt), this));
}

QString FilterVoronoiPlugin::pluginName() const
{
	return "FilterVoronoi";
}

QString FilterVoronoiPlugin::filterName(ActionIDType filterId) const
{
	switch(filterId) {
	case VORONOI_SAMPLING :
		return "Voronoi Sampling";
	case VOLUME_SAMPLING:
		return "Volumetric Sampling";
	case VORONOI_SCAFFOLDING:
		return "Voronoi Scaffolding";
	case BUILD_SHELL:
		return "Create Solid Wireframe";
//	case CROSS_FIELD_CREATION:
//		return "Cross Field Creation";
//	case CROSS_FIELD_SMOOTHING:
//		return "Cross Field Smoothing";
	default :
		assert(0);
		return "";
	}
}

QString FilterVoronoiPlugin::pythonFilterName(ActionIDType f) const
{
	switch(f) {
	case VORONOI_SAMPLING :
		return "generate_sampling_voronoi";
	case VOLUME_SAMPLING:
		return "generate_sampling_volumetric";
	case VORONOI_SCAFFOLDING:
		return "generate_voronoi_scaffolding";
	case BUILD_SHELL:
		return "generate_solid_wireframe";
		//	case CROSS_FIELD_CREATION:
		//		return "Cross Field Creation";
		//	case CROSS_FIELD_SMOOTHING:
		//		return "Cross Field Smoothing";
	default :
		assert(0);
		return "";
	}
}


QString FilterVoronoiPlugin::filterInfo(ActionIDType filterId) const
{
	switch(filterId) {
	case VORONOI_SAMPLING :
		return "Compute a point sampling over a mesh and perform a Lloyd relaxation. The filter selects the vertices of the starting mesh that corresponds to the sampled points. <br>"
			   "Two additional layers containing a voronoi tassellation are created, one as a mesh and one as a polyline. "
			   "To save the sampled vertices in a different layer just use the 'move selected vertices to a new layer' filter";
	case VOLUME_SAMPLING:
		return "Compute a volumetric sampling over a watertight mesh.";
	case VORONOI_SCAFFOLDING:
		return "Compute a volumetric sampling over a watertight mesh.";
	case BUILD_SHELL:
		return "";
//	case CROSS_FIELD_CREATION:
//		return "";
//	case CROSS_FIELD_SMOOTHING:
//		return "";
	default :
		assert(0);
		return "";
	}
}

FilterVoronoiPlugin::FilterClass FilterVoronoiPlugin::getClass(const QAction* a) const
{
	switch(ID(a)) {
	case VORONOI_SAMPLING :
	case VOLUME_SAMPLING:
	case VORONOI_SCAFFOLDING:
		return FilterPlugin::Sampling;
	case BUILD_SHELL:
		return FilterPlugin::Remeshing;
//	case CROSS_FIELD_CREATION:
//		return FilterPlugin::Normal;
//	case CROSS_FIELD_SMOOTHING:
//		return MeshFilterInterface::Smoothing;
	default :
		assert(0);
		return FilterPlugin::Generic;
	}
}

FilterPlugin::FilterArity FilterVoronoiPlugin::filterArity(const QAction* a) const
{
	switch(ID(a)) {
	case VORONOI_SAMPLING :
	case VORONOI_SCAFFOLDING :
//	case CROSS_FIELD_CREATION :
//	case CROSS_FIELD_SMOOTHING :
		return SINGLE_MESH;
	case VOLUME_SAMPLING :
	case BUILD_SHELL :
		return VARIABLE;
	default :
		assert(0);
		return NONE;
	}
}

RichParameterList FilterVoronoiPlugin::initParameterList(const QAction* action, const MeshModel& m)
{
	RichParameterList par;
	switch(ID(action))	 {
	case VORONOI_SAMPLING :
		par.addParam(RichInt("iterNum", 10, "Iteration", "number of iterations"));
		par.addParam(RichInt("sampleNum", 10, "Sample Num.", "Number of samples"));
		par.addParam(RichFloat("radiusVariance", 1, "Radius Variance", "The distance metric will vary along the surface between 1/x and x, linearly according to the scalar field specified by the quality."));
		par.addParam(RichEnum("colorStrategy", 1, {"None", "Seed Distance", "Border Distance", "Region Area"}, "Color Strategy"));
		par.addParam(RichEnum("distanceType", 0, {"Euclidean", "Quality Weighted", "Anisotropic"}, "Distance Type"));
		par.addParam(RichBool("preprocessFlag", false, "Preprocessing"));
		par.addParam(RichInt("refineFactor", 10, "Refinement Factor", "To ensure good convergence the mesh should be more complex than the voronoi partitioning. This number affect how much the mesh is refined according to the required number of samples."));
		par.addParam(RichFloat("perturbProbability", 0, "Perturbation Probability", "To ensure good convergence the mesh should be more complex than the voronoi partitioning. This number affect how much the mesh is refined according to the required number of samples."));
		par.addParam(RichFloat("perturbAmount", 0.001, "Perturbation Amount", "To ensure good convergence the mesh should be more complex than the voronoi partitioning. This number affect how much the mesh is refined according to the required number of samples."));
		par.addParam(RichInt("randomSeed", 0, "Random seed", "To ensure repeatability you can specify the random seed used. If 0 the random seed is tied to the current clock."));
		par.addParam(RichEnum("relaxType", 1, {"Geodesic", "Squared Distance", "Restricted"}, "Relax Type",
									"At each relaxation step we search for each voronoi region the new position of the seed. "
									"According to the classical LLoyd relaxation strategy it should have been placed onto the "
									"barycenter of the region. Over a surface we have two different strategies: <ul>"
									"<li> Geodesic: the seed is placed onto the vertex that maximize the geodesic distance from the border of the region </li>"
									"<li> Squared Distance: the seed is placed in the vertex that minimize the squared sum of the distances from all the pints of the region.</li>"
									"<li> Restricted: the seed is placed in the barycenter of current voronoi region. Even if it is outside the surface. During the relaxation process the seed is free to move off the surface in a continuous way. Re-association to vertex is done at the end..</li>"
									"</ul>"));
		break;
	case VOLUME_SAMPLING:
		par.addParam(RichAbsPerc("sampleSurfRadius", m.cm.bbox.Diag() / 500.0, 0, m.cm.bbox.Diag(),"Surface Sampling Radius", "Surface Sampling is used only as an optimization."));
		par.addParam(RichInt("sampleVolNum", 200000, "Volume Sample Num.", "Number of volumetric samples scattered inside the mesh and used for choosing the voronoi seeds and performing the Lloyd relaxation for having a centroidal voronoi diagram."));
		par.addParam(RichBool("poissonFiltering", true, "Poisson Filtering", "If true the base montecarlo sampling of the volume is filtered to get a poisson disk volumetric distribution."));
		par.addParam(RichAbsPerc("poissonRadius", m.cm.bbox.Diag() / 100.0, 0, m.cm.bbox.Diag(), "Poisson Radius", "Number of voxel per side in the volumetric representation."));
		break;
	case VORONOI_SCAFFOLDING:
		par.addParam(RichAbsPerc("sampleSurfRadius", m.cm.bbox.Diag() / 100.0, 0, m.cm.bbox.Diag(), "Surface Sampling Radius", "Surface Sampling is used only as an optimization."));
		par.addParam(RichInt("sampleVolNum", 100000, "Volume Sample Num.", "Number of volumetric samples scattered inside the mesh and used for choosing the voronoi seeds and performing the Lloyd relaxation for having a centroidal voronoi diagram."));
		par.addParam(RichInt("voxelRes", 50, "Volume Side Resolution", "Number of voxel per side in the volumetric representation."));
		par.addParam(RichFloat("isoThr", 1, "Width of the entity (in voxel)", "Number of voxel per side in the volumetric representation."));
		par.addParam(RichInt("smoothStep", 3, "Smooth Step", "Number of voxel per side in the volumetric representation."));
		par.addParam(RichInt("relaxStep", 5, "Lloyd Relax Step", "Number of Lloyd relaxation step to get a better distribution of the voronoi seeds."));
		par.addParam(RichBool("surfFlag", true, "Add original surface", "Number of voxel per side in the volumetric representation."));
		par.addParam(RichEnum("elemType", 1, {"Seed", "Edge", "Face"}, "Voronoi Element"));
		break;
	case BUILD_SHELL:
		par.addParam(RichBool("edgeCylFlag", true, "Edge -> Cyl.", "If True all the edges are converted into cylinders."));
		par.addParam(RichAbsPerc("edgeCylRadius", m.cm.bbox.Diag() / 100.0, 0, m.cm.bbox.Diag(), "Edge Cylinder Rad.", "The radius of the cylinder replacing each edge."));
		par.addParam(RichBool("vertCylFlag", false, "Vertex -> Cyl.", "If True all the vertices are converted into cylinders."));
		par.addParam(RichAbsPerc("vertCylRadius", m.cm.bbox.Diag() / 100.0, 0, m.cm.bbox.Diag(), "Vertex Cylinder Rad.", "The radius of the cylinder replacing each vertex."));
		par.addParam(RichBool("vertSphFlag", true, "Vertex -> Sph.", "If True all the vertices are converted into sphere."));
		par.addParam(RichAbsPerc("vertSphRadius", m.cm.bbox.Diag() / 100.0, 0, m.cm.bbox.Diag(), "Vertex Sphere Rad.", "The radius of the sphere replacing each vertex."));
		par.addParam(RichBool("faceExtFlag", true, "Face -> Prism", "If True all the faces are converted into prism."));
		par.addParam(RichAbsPerc("faceExtHeight", m.cm.bbox.Diag() / 200.0, 0, m.cm.bbox.Diag(), "Face Prism Height", "The Height of the prism that is substituted with each face."));
		par.addParam(RichAbsPerc("faceExtInset", m.cm.bbox.Diag() / 200.0, 0, m.cm.bbox.Diag(), "Face Prism Inset", "The inset radius of each prism, e.g. how much it is moved toward the inside each vertex on the border of the prism."));
		par.addParam(RichBool("edgeFauxFlag", true, "Ignore faux edges", "If true only the Non-Faux edges will be considered for conversion."));
		par.addParam(RichInt("cylinderSideNum", 16, "Cylinder Side", "Number of sides of the cylinder (both edge and vertex)."));
		break;
//	case CROSS_FIELD_CREATION:
//		par.addParam(RichEnum("crossType", 0, {"Linear Y", "Radial", "Curvature"}, "Cross Type", ""));
//		break;
//	case CROSS_FIELD_SMOOTHING:
//		par.addParam(RichBool("preprocessFlag", true, "Preprocessing"));
//		break;
	default :
		assert(0);
	}
	return par;
}

int FilterVoronoiPlugin::getPreConditions(const QAction* action) const
{
	switch(ID(action))	 {
	case VORONOI_SAMPLING :
	case VOLUME_SAMPLING:
	case VORONOI_SCAFFOLDING:
	case BUILD_SHELL:
//	case CROSS_FIELD_CREATION:
		return MeshModel::MM_NONE;
//	case CROSS_FIELD_SMOOTHING:
//		return MeshModel::MM_VERTCURVDIR;
	default :
		assert(0);
		return 0;
	}
}

std::map<std::string, QVariant> FilterVoronoiPlugin::applyFilter(
		const QAction * action,
		const RichParameterList & par,
		MeshDocument &md,
		unsigned int& /*postConditionMask*/,
		vcg::CallBackPos *cb)
{
	switch(ID(action))	 {
	case VORONOI_SAMPLING :
		voronoiSampling(
					md, cb,
					par.getInt("iterNum"), par.getInt("sampleNum"), par.getFloat("radiusVariance"),
					par.getEnum("distanceType"), par.getInt("randomSeed"), par.getEnum("relaxType"),
					par.getEnum("colorStrategy"), par.getInt("refineFactor"), par.getFloat("perturbProbability"),
					par.getFloat("perturbAmount"), par.getBool("preprocessFlag"));
		break;
	case VOLUME_SAMPLING:
		volumeSampling(
					md, cb,
					par.getFloat("sampleSurfRadius"),
					par.getInt("sampleVolNum"),
					par.getBool("poissonFiltering"),
					par.getFloat("poissonRadius"));
		break;
	case VORONOI_SCAFFOLDING:
		voronoiScaffolding(
					md, cb,
					par.getFloat("sampleSurfRadius"), par.getInt("sampleVolNum"),
					par.getInt("voxelRes"), par.getFloat("isoThr"), par.getInt("smoothStep"),
					par.getInt("relaxStep"), par.getBool("surfFlag"), par.getInt("elemType"));
		break;
	case BUILD_SHELL:
		createSolidWireframe(
					md,
					par.getBool("edgeCylFlag"), par.getFloat("edgeCylRadius"),
					par.getBool("vertCylFlag"), par.getFloat("vertCylRadius"),
					par.getBool("vertSphFlag"), par.getFloat("vertSphRadius"),
					par.getBool("faceExtFlag"), par.getFloat("faceExtHeight"),
					par.getFloat("faceExtInset"), par.getBool("edgeFauxFlag"),
					par.getInt("cylinderSideNum"));
		break;
//	case CROSS_FIELD_CREATION:
//		crossFieldCreation(md, par.getEnum("crossType"));
//		break;
//	case CROSS_FIELD_SMOOTHING:
//		return crossFieldSmoothing(md, par.getBool("preprocessFlag"));
	default :
		wrongActionCalled(action);
		break;
	}
	return std::map<std::string, QVariant>();
}

int FilterVoronoiPlugin::postCondition(const QAction* action) const
{
	switch(ID(action))	 {
	case VORONOI_SAMPLING :
	case VOLUME_SAMPLING:
	case VORONOI_SCAFFOLDING:
		return MeshModel::MM_VERTCOLOR | MeshModel::MM_VERTQUALITY;
	case BUILD_SHELL:
		return MeshModel::MM_NONE;
//	case CROSS_FIELD_CREATION:
//		return MeshModel::MM_VERTCURVDIR;
//	case CROSS_FIELD_SMOOTHING:
//		return MeshModel::MM_VERTCOLOR;
	default :
		assert(0);
		return 0;
	}
}

void FilterVoronoiPlugin::voronoiSampling(
		MeshDocument &md,
		vcg::CallBackPos* cb,
		int iterNum,
		int sampleNum,
		Scalarm radiusVariance,
		int distanceType,
		int randomSeed,
		int relaxType,
		int colorStrategy,
		int refineFactor,
		Scalarm perturbProbability,
		Scalarm perturbAmount,
		bool preprocessingFlag)
{
	MeshModel *om=md.addOrGetMesh("voro", "voro", false);
	MeshModel *poly=md.addOrGetMesh("poly", "poly", false);
	
	om->updateDataMask(MeshModel::MM_VERTCOLOR);

	MeshModel &m=*md.mm();

	m.updateDataMask(MeshModel::MM_FACEFACETOPO);
	m.updateDataMask(MeshModel::MM_VERTFACETOPO);
	m.updateDataMask(MeshModel::MM_VERTCURV);
	m.updateDataMask(MeshModel::MM_VERTCURVDIR);
	m.updateDataMask(MeshModel::MM_VERTCOLOR);
	tri::UpdateCurvature<CMeshO>::PerVertexBasicRadialCrossField(m.cm);

	//tri::EuclideanDistance<CMeshO> dd;
	std::vector<CVertexO *> seedVec;
	//tri::ClusteringSampler<CMeshO> cs(seedVec);
	//tri::SurfaceSampling<CMeshO, vcg::tri::ClusteringSampler<CMeshO> >::SamplingRandomGenerator().initialize(randSeed);
	std::vector<Point3m> pointVec;
	std::vector<bool> fixedVec;
	CMeshO::ScalarType radius=0;

	tri::PoissonSampling<CMeshO>(m.cm,pointVec,sampleNum,radius,radiusVariance,0,randomSeed);

	tri::VoronoiProcessingParameter vpp;
	vpp.geodesicRelaxFlag = (relaxType==0);
	vpp.colorStrategy = colorStrategy;
	vpp.deleteUnreachedRegionFlag=true;
	vpp.refinementRatio=refineFactor;
	vpp.seedPerturbationAmount = perturbAmount;
	vpp.seedPerturbationProbability = perturbProbability;
	if(preprocessingFlag) {
		tri::VoronoiProcessing<CMeshO>::PreprocessForVoronoi(m.cm,radius,vpp);
	}

	tri::VoronoiProcessing<CMeshO>::SeedToVertexConversion(m.cm,pointVec,seedVec);
	fixedVec.resize(pointVec.size(),false);
	QList<int> meshlist; meshlist << m.id();

	// Uniform Euclidean Distance
	if(distanceType==0)  { 
		EuclideanDistance<CMeshO> dd;
		for(int i=0;i<iterNum;++i) {
			cb(100*i/iterNum, "Relaxing...");
			if(relaxType==2) {
				tri::VoronoiProcessing<CMeshO, EuclideanDistance<CMeshO> >::RestrictedVoronoiRelaxing(m.cm, pointVec, fixedVec, 10,vpp);
				tri::VoronoiProcessing<CMeshO>::SeedToVertexConversion(m.cm,pointVec,seedVec);
				tri::VoronoiProcessing<CMeshO>::ComputePerVertexSources(m.cm,seedVec,dd);
			}
			else {
				tri::VoronoiProcessing<CMeshO, EuclideanDistance<CMeshO> >::VoronoiRelaxing(m.cm, seedVec, 1,dd,vpp);
			}
			// md.updateRenderStateMeshes(meshlist,int(MeshModel::MM_VERTCOLOR));
			//if (intteruptreq)
			//	return true;
		}
		om->updateDataMask(MeshModel::MM_FACEFACETOPO);
		tri::VoronoiProcessing<CMeshO>::ConvertVoronoiDiagramToMesh(m.cm,om->cm,poly->cm,seedVec, vpp);
	}
    //  Quality Weighted Distance
	if(distanceType==1) {
		IsotropicDistance<CMeshO> id(m.cm,radiusVariance);
		for(int i=0;i<iterNum;++i) {
			cb(100*i/iterNum, "Relaxing...");
			tri::VoronoiProcessing<CMeshO, IsotropicDistance<CMeshO> >::VoronoiRelaxing(m.cm, seedVec, 1,id,vpp);
			//md.updateRenderStateMeshes(meshlist,int(MeshModel::MM_VERTCOLOR));
			//if (intteruptreq)
			//	return true;
		}
		om->updateDataMask(MeshModel::MM_FACEFACETOPO);
		tri::VoronoiProcessing<CMeshO>::ConvertVoronoiDiagramToMesh(m.cm,om->cm,poly->cm,seedVec, vpp);
	}
	// Anisotropic distance 
	if(distanceType==2) { 
		for(int i=0;i<iterNum;++i) {
			cb(100*i/iterNum, "Relaxing...");
			BasicCrossFunctor<CMeshO> bcf(m.cm);
			AnisotropicDistance<CMeshO> ad(m.cm,bcf);
			tri::VoronoiProcessing<CMeshO, AnisotropicDistance<CMeshO> >::VoronoiRelaxing(m.cm, seedVec, 1, ad, vpp);
			//md.updateRenderStateMeshes(meshlist,int(MeshModel::MM_VERTCOLOR));
			//if (intteruptreq)
			//	return true;
		}
		//tri::VoronoiProcessing<CMeshO, AnisotropicDistance<CMeshO> >::ConvertVoronoiDiagramToMesh(m.cm,om->cm,seedVec, ad, vpp);
	}

	//md.updateRenderStateMeshes(meshlist,int(MeshModel::MM_VERTCOLOR));
	//if (intteruptreq)
	//	return true;

	tri::UpdateSelection<CMeshO>::VertexClear(m.cm);
	for(auto vi =seedVec.begin();vi!=seedVec.end();++vi)
		(*vi)->SetS();

	om->updateBoxAndNormals();
}

void FilterVoronoiPlugin::volumeSampling(
		MeshDocument& md,
		vcg::CallBackPos* cb,
		Scalarm sampleSurfRadius,
		int sampleVolNum,
		bool poissonFiltering,
		Scalarm poissonRadius)
{
	MeshModel* m= md.mm();
	m->updateDataMask(MeshModel::MM_FACEMARK);

	MeshModel* mcVm= md.addOrGetMesh("Montecarlo Volume","Montecarlo Volume",false);
	//MeshModel* pVm = nullptr;
	if (poissonFiltering)
		/*pVm =*/ md.addOrGetMesh("Poisson Sampling","Poisson Sampling",false);
	MeshModel* pSm= md.addOrGetMesh("Surface Sampling","Surface Sampling",false);

	mcVm->updateDataMask(MeshModel::MM_VERTCOLOR | MeshModel::MM_VERTQUALITY);
	pSm->updateDataMask(MeshModel::MM_VERTCOLOR | MeshModel::MM_VERTQUALITY);
	VoronoiVolumeSampling<CMeshO> vvs(m->cm);
	log("Sampling Surface at a radius %f ",sampleSurfRadius);
	cb(1, "Init");
	vvs.Init(sampleSurfRadius);
	cb(30, "Sampling Volume...");
	vvs.BuildVolumeSampling(sampleVolNum,poissonRadius,0);
	tri::Append<CMeshO,CMeshO>::MeshCopy(mcVm->cm,vvs.montecarloVolumeMesh);
	tri::UpdateColor<CMeshO>::PerVertexQualityRamp(mcVm->cm);
	//vvs.ThicknessEvaluator();
	tri::Append<CMeshO,CMeshO>::MeshCopy(pSm->cm,vvs.psd.poissonSurfaceMesh);

	//TODO: compute poisson filtered volume mesh
}

void FilterVoronoiPlugin::voronoiScaffolding(
		MeshDocument& md,
		vcg::CallBackPos* cb,
		Scalarm sampleSurfRadius,
		int sampleVolNum,
		int voxelRes,
		Scalarm isoThr,
		int smoothStep,
		int relaxStep,
		bool surfFlag,
		int elemType)
{
	MeshModel *m= md.mm();
	m->updateDataMask(MeshModel::MM_FACEMARK);
	MeshModel   *pm= md.addOrGetMesh("Poisson-disk Samples","Poisson-disk Samples",false);
	MeshModel *mcVm= md.addOrGetMesh("Montecarlo Volume","Montecarlo Volume",false);
	/*MeshModel  *vsm= */md.addOrGetMesh("Voronoi Seeds","Voronoi Seeds",false);
	MeshModel   *sm= md.addOrGetMesh("Scaffolding","Scaffolding",false);

	pm->updateDataMask(m);
	cb(10, "Sampling Surface...");

	VoronoiVolumeSampling<CMeshO> vvs(m->cm);
	VoronoiVolumeSampling<CMeshO>::Param par;

	log("Sampling Surface at a radius %f ",sampleSurfRadius);
	vvs.Init(sampleSurfRadius);
	cb(30, "Sampling Volume...");
	CMeshO::ScalarType poissonVolumeRadius=0;
	vvs.BuildVolumeSampling(sampleVolNum,poissonVolumeRadius,0);
	log("Base Poisson volume sampling at a radius %f ",poissonVolumeRadius);

	cb(40, "Relaxing Volume...");
	vvs.BarycentricRelaxVoronoiSamples(relaxStep);

	cb(50, "Building Scaffloding Volume...");
	par.isoThr = isoThr;
	par.surfFlag = surfFlag;
	par.elemType = elemType;
	par.voxelSide = voxelRes;
	vvs.BuildScaffoldingMesh(sm->cm,par);
	cb(90, "Final Smoothing...");
	tri::Smooth<CMeshO>::VertexCoordLaplacian(sm->cm, smoothStep);
	sm->updateBoxAndNormals();
	tri::Append<CMeshO,CMeshO>::MeshCopy(mcVm->cm,vvs.montecarloVolumeMesh);
	tri::Append<CMeshO,CMeshO>::MeshCopy(pm->cm,vvs.psd.poissonSurfaceMesh);
}

void FilterVoronoiPlugin::createSolidWireframe(
		MeshDocument& md,
		bool edgeCylFlag,
		Scalarm edgeCylRadius,
		bool vertCylFlag,
		Scalarm vertCylRadius,
		bool vertSphFlag,
		Scalarm vertSphRadius,
		bool faceExtFlag,
		Scalarm faceExtHeight,
		Scalarm faceExtInset,
		bool /*edgeFauxFlag*/,
		int cylinderSideNum)
{
	//ToDo: consider edge faux flag

	MeshModel *m= md.mm();
	m->updateDataMask(MeshModel::MM_FACEFACETOPO);
	MeshModel *sm= md.addOrGetMesh("Shell Mesh","Shell Mesh",false);

	sm->cm.Clear();
	sm->updateDataMask(MeshModel::MM_FACEFACETOPO);
	tri::RequireFFAdjacency(sm->cm);
	tri::UpdateNormal<CMeshO>::PerVertexNormalizedPerFace(m->cm);
	tri::Clean<CMeshO>::RemoveUnreferencedVertex(m->cm);
	tri::Allocator<CMeshO>::CompactEveryVector(m->cm);
	if(edgeCylFlag) tri::BuildCylinderEdgeShell(m->cm,sm->cm,edgeCylRadius,cylinderSideNum);
	if(vertCylFlag) tri::BuildCylinderVertexShell(m->cm,sm->cm,vertCylRadius,edgeCylRadius,cylinderSideNum);
	if(vertSphFlag) tri::BuildSphereVertexShell(m->cm,sm->cm,vertSphRadius);
	if(faceExtFlag) tri::BuildPrismFaceShell(m->cm,sm->cm,faceExtHeight,faceExtInset);

	sm->updateBoxAndNormals();
}

void FilterVoronoiPlugin::crossFieldCreation(
		MeshDocument& md,
		int crossType)
{
	MeshModel &m=*md.mm();
	m.updateDataMask(MeshModel::MM_VERTCURVDIR + MeshModel::MM_FACECURVDIR + MeshModel::MM_FACEQUALITY + MeshModel::MM_FACECOLOR);

	if(crossType == 0) { //Linear Y
		float range = m.cm.bbox.DimY();
		for(size_t i=0;i<m.cm.vert.size();++i) {
			CMeshO::ScalarType q01 = .25f+(m.cm.vert[i].P().Y() - m.cm.bbox.min.Y())/(2.0f*range);
			m.cm.vert[i].PD1().Import(Point3m(1,0,0)*q01);
			m.cm.vert[i].PD2().Import(Point3m(0,1,0)*(sqrt(1-q01*q01)));
		}
	}

	if(crossType == 1) { // Radial
		tri::UpdateCurvature<CMeshO>::PerVertexBasicRadialCrossField(m.cm,2.0);
	}

	if(crossType == 2) { // Curvature
		m.updateDataMask(MeshModel::MM_VERTFACETOPO);
		m.updateDataMask(MeshModel::MM_FACEFACETOPO);
		//      tri::FieldSmoother<CMeshO>::InitByCurvature(m.cm);
		//      tri::FieldSmoother<CMeshO>::SmoothParam par;
		//      tri::FieldSmoother<CMeshO>::SmoothDirections(m.cm,par);
	}
}

void FilterVoronoiPlugin::crossFieldColoring(MeshDocument& md)
{
	//ToDo: Filter not used
	MeshModel &m=*md.mm();
	m.updateDataMask(MeshModel::MM_VERTCOLOR);
	m.updateDataMask(MeshModel::MM_VERTQUALITY);
	for(CMeshO::VertexIterator vi=m.cm.vert.begin();vi!=m.cm.vert.end();++vi) {
		vi->Q() = Norm(vi->PD1()) / Norm(vi->PD2());
	}

	tri::UpdateColor<CMeshO>::PerVertexQualityRamp(m.cm);
}

//bool FilterVoronoiPlugin::crossFieldSmoothing(
//		MeshDocument& md,
//		bool preprocessFlag)
//{
//
//}

MESHLAB_PLUGIN_NAME_EXPORTER(FilterVoronoiPlugin)

/******************************************************************************
File:  AtlaExport.cpp

Author:   Nick Chirkov
Copyright (C) 2001 Nick Chirkov

Comments:
Maya interface for export
******************************************************************************/
#define LOG
//NOTE_IT

#define CM2M_SCALE 0.01f

#include <maya/MFnPlugin.h>
#include <maya/MComputation.h>
#include <maya/MPxCommand.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MPointArray.h>
#include <maya/MFloatArray.h>
#include <maya/MFnPhongShader.h>
#include <maya/MFnTransform.h>
#include <maya/MMatrix.h>
#include <maya/MEulerRotation.h>
#include <maya/MTime.h>
#include <maya/MAnimControl.h>
#include <maya/MFloatPointArray.h>

#include"AtlaExport.h"
const float R2Y = 0.299f;
const float G2Y = 0.587f;
const float B2Y = 0.114f;

#include <math.h>

#include "GEOM_LIB\rdf_exp.h"
#include <vector>
#include <maya/MDagPathArray.h>
#include <maya/MItGeometry.h>


bool MakePathForTraceMaker(char * buf)
{
	const char * cl = ::GetCommandLine();
	if(cl[0] != '"') return false;
	for(long i = 1; cl[i] && cl[i] != '"'; i++) buf[i - 1] = cl[i];
	if(cl[i] != '"') return false;
	for(i--; buf[i] != '\\' && i >= 0; i--);
	i++;
	const char * fileName = "TraceMaker.exe";
	for(long j = 0; fileName[j]; i++, j++) buf[i] = fileName[j];
	buf[i] = 0;
	return true;
}


using namespace std;
EXPORT_STATIC *rexp;
FILE *LogFile=0;
void Log(const char *fmt,...)
{
#ifdef LOG	
	va_list args;
	va_start(args,fmt);
	vfprintf(LogFile, fmt, args);
	va_end(args);
#endif
}
MStatus status;
#include <maya/MItDependencyGraph.h>
#include <maya/MFnStringData.h>
vector<string> locators;
bool GlobAnimation;

bool Result(std::string &s)
{
	MString command = "text -edit -label \"";
	command += s.c_str();
	command += "\" MyLable";
	MGlobal::executeCommand(command,true);
	//MGlobal::executeCommand("showWindow StatusWin",true);
	return true;
}
bool Error(std::string &s)
{
	MString command = "text -edit -label \"";
	command += s.c_str();
	command += "\" MyLable";
	MGlobal::executeCommand(command,true);
	FILE *errfl = fopen("error.txt", "w+a");
	fprintf(errfl, "%s\n", s.c_str());
	fclose(errfl);
	return true;
}

#include <maya/MItDependencyNodes.h>
#include <maya/MFnSkinCluster.h>

struct BONEWEIGHT
{
	long idBone[4];
	float weight[4];
};
MSelectionList mJointsList;
void FillJoints(MDagPath &path)
{
	MItDag dagIterator( MItDag::kBreadthFirst, MFn::kJoint, &status);
	if(!status)	return;

	for ( ; !dagIterator.isDone(); dagIterator.next() ) 
	{
		MDagPath dagPath;
		status = dagIterator.getPath(dagPath);  // Get DagPath
		if(!status) return;

		MObject object = dagIterator.item(&status);     // Get MObject
		if(!status) return;

		mJointsList.add(dagPath, object, false);
		//Log( "JOINT_ADDED: %s\n", dagPath.fullPathName().asChar());
	}
}

void SaveAnimation(BONEWEIGHT *ani, MDagPath &path, long nverts)
{
 	memset(ani, 0, sizeof(BONEWEIGHT)*nverts);

	MStatus status;
	Log( "\n\n\nobject: %s\n", path.fullPathName().asChar());
	FillJoints(path);

	MItDependencyNodes iter(MFn::kSkinClusterFilter); // Create SkinClusters iterator
  if( iter.isDone() ) return; // No SkinClusters

  for ( ; !iter.isDone(); iter.next() )
  {
    MObject object = iter.item();
		MFnSkinCluster fnSkinCluster;
    status = fnSkinCluster.setObject(object);       // Get MFnSkinCluster interface
    if(!status) continue;

    int numConnections = fnSkinCluster.numOutputConnections(&status);
    if(numConnections==0) continue;

    for(int con=0; con<numConnections; con++)           // Check Output connections
    {
      int PlugIndex = fnSkinCluster.indexForOutputConnection(con,&status);
      if(!status) continue;

			MDagPath SkinDagPath;
      status = fnSkinCluster.getPathAtIndex(PlugIndex, SkinDagPath);	// Check connectoin to current shape
			SkinDagPath.pop();
      if( !(SkinDagPath==path) ) continue;

			MDagPathArray InfsObjectsArray;
      fnSkinCluster.influenceObjects(InfsObjectsArray,&status);
      if(!status)	return;

      MItGeometry gIter(SkinDagPath);
      for (  ; !gIter.isDone(); gIter.next() )  // cycle for all vertexes in geometry --
      {
        MObject component = gIter.component(&status); 
        if(!status)	return;

				MFloatArray WeightArray;
				unsigned int numInfl;
        status = fnSkinCluster.getWeights(SkinDagPath,component,WeightArray, numInfl);
        if(!status)	return;

        int idxVrt = gIter.index();
        int count = 0;
        for(long i=0; i<numInfl; i++)
        {
          if(WeightArray[i]==0.0f) continue;

          long J_Index = -1;
          for(long joint=0; joint<mJointsList.length(); joint++)
          {
						MDagPath dagPath;
            mJointsList.getDagPath(joint, dagPath);
            if( !(InfsObjectsArray[i] == dagPath) ) continue;
            J_Index = joint;
            break;
          }
          if(J_Index<0) return;

          if(count>=2)                  // More then 4 objects
          {
            long j = 0;
						if(ani[idxVrt].weight[1] < ani[idxVrt].weight[0])	j = 1;

            if( WeightArray[i] > max(ani[idxVrt].weight[0], ani[idxVrt].weight[1]) )
            {
              ani[idxVrt].weight[j]  = WeightArray[i];
              ani[idxVrt].idBone[j]   = J_Index;
            }
          }
          else                        // Less then 4 objects
          {
						ani[idxVrt].weight[count] = WeightArray[i];
						ani[idxVrt].idBone[count]  = J_Index;
					}
					count++;
				}


				//normalize value
				float normw = 1.0f/(ani[idxVrt].weight[0] + ani[idxVrt].weight[1]);
				ani[idxVrt].weight[0] *= normw;
				ani[idxVrt].weight[1] *= normw;

				for(long w=0; w<4; w++)
						Log( "%f, %d, ", ani[idxVrt].weight[w], ani[idxVrt].idBone[w]);
				Log( "\n");

			}//all geometry
		}//connections
	}
};


AtlaExport::AtlaExport()
{
}

AtlaExport::~AtlaExport()
{
}

void MulMtx(float *matrix, float *m1, float *m2)
{
	matrix[0] = m2[0]*m1[0] + m2[4]*m1[1] + m2[8]*m1[2] + m2[12]*m1[3];
	matrix[1] = m2[1]*m1[0] + m2[5]*m1[1] + m2[9]*m1[2] + m2[13]*m1[3];
	matrix[2] = m2[2]*m1[0] + m2[6]*m1[1] + m2[10]*m1[2] + m2[14]*m1[3];
	matrix[3] = m2[3]*m1[0] + m2[7]*m1[1] + m2[11]*m1[2] + m2[15]*m1[3];

	matrix[4] = m2[0]*m1[4] + m2[4]*m1[5] + m2[8]*m1[6] + m2[12]*m1[7];
	matrix[5] = m2[1]*m1[4] + m2[5]*m1[5] + m2[9]*m1[6] + m2[13]*m1[7];
	matrix[6] = m2[2]*m1[4] + m2[6]*m1[5] + m2[10]*m1[6] + m2[14]*m1[7];
	matrix[7] = m2[3]*m1[4] + m2[7]*m1[5] + m2[11]*m1[6] + m2[15]*m1[7];

	matrix[8] = m2[0]*m1[8] + m2[4]*m1[9] + m2[8]*m1[10] + m2[12]*m1[11];
	matrix[9] = m2[1]*m1[8] + m2[5]*m1[9] + m2[9]*m1[10] + m2[13]*m1[11];
	matrix[10] = m2[2]*m1[8] + m2[6]*m1[9] + m2[10]*m1[10] + m2[14]*m1[11];
	matrix[11] = m2[3]*m1[8] + m2[7]*m1[9] + m2[11]*m1[10] + m2[15]*m1[11];

	matrix[12] = m2[0]*m1[12] + m2[4]*m1[13] + m2[8]*m1[14] + m2[12]*m1[15];
	matrix[13] = m2[1]*m1[12] + m2[5]*m1[13] + m2[9]*m1[14] + m2[13]*m1[15];
	matrix[14] = m2[2]*m1[12] + m2[6]*m1[13] + m2[10]*m1[14] + m2[14]*m1[15];
	matrix[15] = m2[3]*m1[12] + m2[7]*m1[13] + m2[11]*m1[14] + m2[15]*m1[15];
}

bool GetFloatValues(MObject &obj, const char *name, float &f0, float &f1)
{
	MFnDependencyNode NodeFnDn;
	NodeFnDn.setObject(obj);
	MStatus status;
	MPlug rpuv = NodeFnDn.findPlug(name,&status);
	if(!status)	return false;
	MObject val;
	rpuv.getValue(val);
	MFnNumericData numFn(val);
	numFn.getData(f0, f1);
	//Log( "%s, %f, %f\n", name, f0, f1);
	return true;
}

bool GetDoubleValue(MObject &obj, const char *name, double &d)
{
	MFnDependencyNode NodeFnDn;
	NodeFnDn.setObject(obj);
	MStatus status;
	MPlug rpuv = NodeFnDn.findPlug(name,&status);
	if(!status)	return false;
	MObject val;
	rpuv.getValue(d);
	return true;
}

MMatrix GetLocator(MDagPath &path, MMatrix &localRotation)
{
	//adjust local position
	double dbvx, dbvy, dbvz;
	MDagPath shape = path;
	shape.push(path.child(0));
	GetDoubleValue(shape.node(), "localPositionX", dbvx);
	GetDoubleValue(shape.node(), "localPositionY", dbvy);
	GetDoubleValue(shape.node(), "localPositionZ", dbvz);
	MVector translate;
	translate.x = dbvx;	translate.y = dbvy;	translate.z = dbvz;
	MTransformationMatrix mtrs;
	mtrs.setTranslation(translate, MSpace::kWorld);

	//rotation component will be stored
	GetDoubleValue(path.node(), "rotateX", dbvx);
	GetDoubleValue(path.node(), "rotateY", dbvy);
	GetDoubleValue(path.node(), "rotateZ", dbvz);
	MEulerRotation rt;
	rt.x = dbvx;	rt.y = dbvy;	rt.z = dbvz;
	MTransformationMatrix mtrr;
	mtrr.rotateTo(rt);
	localRotation = mtrr.asMatrix();

	MMatrix mtx = mtrs.asMatrix()*path.inclusiveMatrix();
	return mtx;
}

bool ExportCharacterPatch;
MStatus AtlaExport::writer(const MFileObject &file,  const MString &optionsString, FileAccessMode mode)
{
  MStringArray    optionList;
	MStringArray    theOption;
  optionsString.split(';', optionList);

	bool scale = false;
	bool bsp = false;
	bool animation = false;
	bool product = false;
	ExportCharacterPatch = false;

  for(DWORD i = 0; i < optionList.length (); ++i)
  {
    theOption.clear();
    optionList[i].split ('=', theOption);
    if (theOption.length () <= 1) continue;

    if(theOption[0] == MString("Check1"))
    {
        if( theOption[1].asInt() ) scale = true;
    }
    if(theOption[0] == MString("Check2"))
    {
        if( theOption[1].asInt() ) bsp = true;
    }
    if(theOption[0] == MString("Check3"))
    {
        if( theOption[1].asInt() ) animation = true;
    }
    if(theOption[0] == MString("ExportCharacterPatch"))
    {
        if( theOption[1].asInt() ) ExportCharacterPatch = true;
    }
    /*if(theOption[0] == MString("radioGrp1"))
    {
        if( theOption[1].asInt()==2 ) product = true;
        else	product = false;
    }*/

    //----------------------------------//
  }// next i(element list)

	if(ExportCharacterPatch)	bsp = true;
	GlobAnimation = animation;
	if(GlobAnimation)	bsp = false;

  MString  fname = file.fullName();
	ExportAll(fname.asChar(), scale, bsp, product);
  return (MS::kSuccess);
}
bool AtlaExport::haveWriteMethod () const
{
  return (true);
}
MString AtlaExport::defaultExtension () const
{
  return MString("gm");
}

MPxFileTranslator::MFileKind   AtlaExport::identifyFile( const MFileObject &file, const char *buffer,short size) const
{
  const char *name = file.name().asChar();
  int nameLength = strlen(name);

  return (kIsMyFileType);
	//return (kNotMyFileType);
}

void *AtlaExport::creator()
{
	return new AtlaExport();
}

// command for adding extra parameters
class ExtraCommand : public MPxCommand
{
public:

	MStatus doIt( const MArgList& )
	{
		MItDag itDag;
		MDagPath dagPath;

		while(!itDag.isDone())
		{
			itDag.getPath(dagPath);
			itDag.next();
			if(!dagPath.hasFn(MFn::kMesh))	continue;

			MFnDependencyNode fnDep(dagPath.node());

			if(dagPath.hasFn(MFn::kTransform))
			{
				MFnNumericAttribute attr;

				MObject rdfattrib = attr.create("visible", "visible", MFnNumericData::kBoolean, 1);
				fnDep.addAttribute(rdfattrib);
				rdfattrib = attr.create("static_light", "static_light", MFnNumericData::kBoolean, 1);
				fnDep.addAttribute(rdfattrib);
				rdfattrib = attr.create("dynamic_light", "dynamic_light", MFnNumericData::kBoolean, 1);
				fnDep.addAttribute(rdfattrib);
				rdfattrib = attr.create("merge", "merge", MFnNumericData::kBoolean, 1);
				fnDep.addAttribute(rdfattrib);
				rdfattrib = attr.create("collision", "collision", MFnNumericData::kBoolean, 1);
				fnDep.addAttribute(rdfattrib);
				rdfattrib = attr.create("cast_shadows", "cast_shadows", MFnNumericData::kBoolean, 1);
				fnDep.addAttribute(rdfattrib);

			}
			/*else
			{
				MObject att;
				att = fnDep.attribute("visible", &status);	if(status==MS::kSuccess)	fnDep.removeAttribute(att);
				att = fnDep.attribute("static_light", &status);	if(status==MS::kSuccess)	fnDep.removeAttribute(att);
				att = fnDep.attribute("dynamic_light", &status);	if(status==MS::kSuccess)	fnDep.removeAttribute(att);
				att = fnDep.attribute("merge", &status);	if(status==MS::kSuccess)	fnDep.removeAttribute(att);
				att = fnDep.attribute("collision", &status);	if(status==MS::kSuccess)	fnDep.removeAttribute(att);
				att = fnDep.attribute("geometry_access", &status);	if(status==MS::kSuccess)	fnDep.removeAttribute(att);
				att = fnDep.attribute("cast_shadows", &status);	if(status==MS::kSuccess)	fnDep.removeAttribute(att);
			}//*/

		}
		return MS::kSuccess;
	}
	static void *creator()
	{
		return new ExtraCommand();
	}
};

//****************************************************************
MStatus initializePlugin (MObject obj)
{
	MStatus status;
	MFnPlugin plugin (obj, "Nick Chirkov (C) 2001", "1.0");
	status = plugin.registerFileTranslator("SD2GeoExport", "", 	AtlaExport::creator,"SD2GeoExportMel");
	plugin.registerCommand( "addsex", ExtraCommand::creator );
	return (status);
}

MStatus uninitializePlugin (MObject obj)
{
	MFnPlugin plugin (obj);
	plugin.deregisterFileTranslator ("SD2GeoExport");
	plugin.deregisterCommand( "addsex" );
	return (MS::kSuccess);
}


struct MATERIAL
{
	long ntex;
	MObject *MO;
	char matname[256];
	char texname[4][256];
	float offset[4][2], repeat[4][2];
	double rotate[4];

	EXPORT_STATIC::BRDF_TYPE brdf;
	float diffuse, specular, gloss, selfIllum, transparency, refraction_id, reflection;
};

long nummat;
MATERIAL material[1024];
long matref[1024];

long curobject, totobjects;
long retobj;

void PreTraverse(MDagPath &path)
{
	//geometry object
	if(path.hasFn(MFn::kMesh) && path.hasFn(MFn::kTransform))
		retobj++;

	//recurse all children
	long nc = path.childCount(&status);
	for(long c=0; c<nc; c++)
	{
		MObject chlo = path.child(c, &status);
		MDagPath cl = path;
		cl.push(chlo);

		//export locator
		PreTraverse(cl);
	}
}


float locator_mtx[4][4];
static long depth = -1;
void ExportLocator(MDagPath &cl)
{
	//search if selected
	MSelectionList slist;
	MGlobal::getActiveSelectionList(slist);
	MItSelectionList iter(slist);
	MItDag dagIterator( MItDag::kDepthFirst, MFn::kInvalid, &status);
	for( ; !iter.isDone(); iter.next() ) 
	{
		MDagPath objectPath;
		status = iter.getDagPath( objectPath );
		if(objectPath==cl)
			break;
	}
	//if locator is not selected do not export
	if(iter.isDone()==true)	return;

	MDagPath gpath(cl);
	MMatrix localRotation;
	MMatrix locator_m = GetLocator(gpath, localRotation);

	gpath.pop();
	MFnDagNode fdagnodeg(gpath);
	MFnDagNode fdagnode(cl);

	EXPORT_STATIC::LABEL loc;
	loc.group_name = string(fdagnodeg.name().asChar());
	loc.name = string(fdagnode.name().asChar());
	loc.flags = 0;
	memset(&loc.bones[0], 0, sizeof(loc.bones));
	memset(&loc.weight[0], 0, sizeof(loc.weight));

  for(long joint=0; joint<mJointsList.length(); joint++)
  {
		MDagPath dagPath;
    mJointsList.getDagPath(joint, dagPath);
    if(gpath == dagPath) break;
  }
	//-----------------------------------------------
	//get matrix for locator
	float mtx[4][4];
	MMatrix glob_parent(locator_mtx);
	MMatrix local_loc;
	//joint is correct
  if(joint<mJointsList.length())
	{
		loc.bones[0] = joint;
		Log("locator_bone: %d\n", joint);
/*		MDagPath jointPath;
    mJointsList.getDagPath(joint, jointPath);
		MMatrix jm = jointPath.inclusiveMatrix().transpose();
		local_loc = locator_m * jm;
Log( "bone Matrix: \n");
for(long ry=0; ry<4; ry++)
{
for(long rx=0; rx<4; rx++)
Log( "%f, ", jm[ry][rx]);
Log( "\n\n");
}*/
	}//else
	local_loc = locator_m * glob_parent;

	if(GlobAnimation)
	{
		local_loc[3][0] *= 0.01f;	local_loc[3][1] *= 0.01f;	local_loc[3][2] *= 0.01f;
	}
	else
	{
		local_loc[0][0] = local_loc[1][1] = local_loc[2][2] = local_loc[3][3] = 1.0f;
		local_loc[0][1] = local_loc[0][2] = local_loc[0][3] = 0.0f;
		local_loc[1][0] = local_loc[1][2] = local_loc[1][3] = 0.0f;
		local_loc[2][0] = local_loc[2][1] = local_loc[2][3] = 0.0f;
		local_loc[3][0] *= -0.01f;	local_loc[3][1] *= 0.01f;	local_loc[3][2] *= 0.01f;
		local_loc = localRotation.transpose()*local_loc;
	}

	//store to locator
	local_loc.get(mtx);
	memcpy(&loc.vector[0][0], &mtx[0][0], sizeof(loc.vector));

	//memset(loc.vector, 0, sizeof loc.vector);	loc.vector[0][0] = loc.vector[1][1] = loc.vector[2][2] = loc.vector[3][3] = 1.0f;

Log( "LOCATOR ADDED TO HIERARCHY: %s\n", loc.name.c_str());
for(long ry=0; ry<4; ry++)
{
for(long rx=0; rx<4; rx++)
Log( "%f, ", loc.vector[ry][rx]);
Log( "\n");
}
	for(long l=0; l<locators.size(); l++)
		if(locators[l]==loc.name)
		{
			loc.group_name = string("geometry");
			break;
		}

	Log( "group_name: %s\n", loc.group_name.c_str());
	Log( "name: %s\n\n", loc.name.c_str());

	//------------------------------------------------------------------------
	//add locator
	//------------------------------------------------------------------------
	try
	{
		rexp->AddLabel(loc);
	}
	catch(char* errmsg)
	{
		MString command = "text -edit -label \"";
		command += errmsg;
		command += "\" MyLable";
		MGlobal::executeCommand(command,true);
		MGlobal::executeCommand("showWindow StatusWin",true);
		Sleep(2000);
	}
}

void Traverse(MDagPath &path)
{
	depth++;
	//--------------------------------------------------------
	//geometry object
	//--------------------------------------------------------
	if(path.hasFn(MFn::kMesh) && path.hasFn(MFn::kTransform))
	{
		MString command = "text -edit -label \"Export object ";
		char ee[256];
		sprintf(ee, "%d / %d", curobject+1, totobjects);
		command += ee;
		command += "\" MyLable";
		MGlobal::executeCommand(command,true);
		MGlobal::executeCommand("showWindow StatusWin",true);
		curobject++;

		EXPORT_STATIC::OBJECT     exp_obj;
		EXPORT_STATIC::MATERIAL   exp_mtl;
		EXPORT_STATIC::TRIANGLE   exp_trg;
		//------------------------------------------------------------------------
		//object
		MFnMesh fnMesh(path, &status);
		if(!status)	goto SkipObj;//throw "error fnMesh";

		//group & object name
		MDagPath gpath(path);
		gpath.pop();
		MFnDagNode fdagnodeg(gpath);
		MFnDagNode fdagnode(path);
		exp_obj.group_name = string(fdagnodeg.name(&status).asChar());
		exp_obj.name = string(fdagnode.name(&status).asChar());

		//------------------------------------------------------------------------
		//extra attributes
		exp_obj.flags = 0;

		bool vis, stlt, dnlt, mg, col, cs;
		fdagnode.findPlug("visible", &status).getValue(vis);
		if(!status)
		{
			exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::VISIBLE;
			exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::STATIC_LIGHT_ENABLE;
			exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::DINAMIC_LIGHT_ENABLE;
			exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::MERGE;
			exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::COLLISION_ENABLE;
			exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::CAST_SHADOWS_ENABLE;
		}
		else
		{
			fdagnode.findPlug("static_light").getValue(stlt);
			fdagnode.findPlug("dynamic_light").getValue(dnlt);
			fdagnode.findPlug("merge").getValue(mg);
			fdagnode.findPlug("collision").getValue(col);
			fdagnode.findPlug("cast_shadows").getValue(cs);

			if(vis)	exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::VISIBLE;
			if(stlt)	exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::STATIC_LIGHT_ENABLE;
			if(dnlt)	exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::DINAMIC_LIGHT_ENABLE;
			if(mg)	exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::MERGE;
			if(col)	exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::COLLISION_ENABLE;
			if(cs)	exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::CAST_SHADOWS_ENABLE;
		}

		//------------------------------------------------------------------------
		//double side
		bool doublesided =false;
		MPlug pdside = fnMesh.findPlug("doubleSided",&status);
		if(status) pdside.getValue( doublesided );
		if(!doublesided)	exp_obj.flags |= EXPORT_STATIC::OBJECT_FLAGS::CULLENABLE;

		//------------------------------------------------------------------------
		//opposite
		bool opposite = false;
		MPlug poppos = fnMesh.findPlug("opposite",&status);
		if(status) poppos.getValue(opposite);

		//------------------------------------------------------------------------
		//object info

		//get uvset names
		long uvsets = fnMesh.numUVSets();
		MStringArray msArray;
		fnMesh.getUVSetNames(msArray);
		long uvsettype[8];
		//calculate UVsets
		bool color_present = false;
		for(long s=0; s<uvsets; s++)
			if(msArray[s]=="color")
				color_present = true;

		for(s=0; s<uvsets; s++)
		{
			uvsettype[s] = -1;

			//base texture
			if(msArray[s]=="bump")	uvsettype[s] = 1;
			else
				if(msArray[s]=="color")	uvsettype[s] = 0;
				else
					if(!color_present && msArray[s]=="map1")	uvsettype[s] = 0;
		}

		long InstaceNumber = path.instanceNumber(&status);
		if(!status)
		{
			goto SkipObj;//throw "error instanceNumber";
		}
		
		MObjectArray      ArrayOfSharers;
		MIntArray         ArrayOfIndex;
		status = fnMesh.getConnectedShaders(InstaceNumber, ArrayOfSharers, ArrayOfIndex);
		if(!status)
		{
			Error(std::string("error getConnectedShaders"));
			goto SkipObj;//throw "error getConnectedShaders";
		}

		//------------------------------------------------------------------------
		//material
		//------------------------------------------------------------------------
		//reference
		for(unsigned long i=0; i<ArrayOfSharers.length(); i++)
		{
			for(long m=0; m<nummat; m++)
				//if(ArrayOfSharers[i]==material[m].MO)	break;// && uvsets==material[m].ntex)	break;
				if(ArrayOfSharers[i]==*material[m].MO)	break;

			if(m==nummat)
			{
				//optical params
				MFnDependencyNode fnNode;
				fnNode.setObject(ArrayOfSharers[i]);
				MPlug shaderPlug = fnNode.findPlug( "surfaceShader", &status );
				if(status)
				{
					MPlugArray connectedPlugs;
					shaderPlug.connectedTo( connectedPlugs, true, false );
					if(connectedPlugs.length()==1)
					{
						MObject ShaderObject = connectedPlugs[0].node();
						switch ( ShaderObject.apiType() )
						{
							case MFn::kPhong:
							{
								MFnPhongShader fnShader( ShaderObject );
								material[m].gloss = fnShader.cosPower();
								material[m].specular = fnShader.reflectivity();
								MColor sp_color = fnShader.specularColor(&status);
								if(status) material[m].specular = sp_color.r*R2Y +sp_color.g*G2Y +sp_color.b*B2Y;
								material[m].brdf = EXPORT_STATIC::PHONG;	//brdf
								material[m].diffuse = fnShader.diffuseCoeff();	//diffuse
								MColor si_color = fnShader.incandescence(&status);	//selfillumination
								if(status) material[m].selfIllum = R2Y*si_color.r +G2Y*si_color.g +B2Y*si_color.b;
								MColor tr_color = fnShader.transparency(&status);	//transparency
								if(status) material[m].transparency = R2Y*tr_color.r +G2Y*tr_color.g +B2Y*tr_color.b;
								material[m].refraction_id  = fnShader.refractiveIndex(&status);	//refraction
								MPlug reflectPlug = fnShader.findPlug( "reflectionSpecularity", &status );	//reflection
								if(status) reflectPlug.getValue(material[m].reflection);

							}
							break;
							case MFn::kLambert:
							{
								MFnLambertShader fnShader( ShaderObject );
								material[m].brdf = EXPORT_STATIC::PHONG;	//brdf
								material[m].diffuse = fnShader.diffuseCoeff();	//diffuse
								MColor si_color = fnShader.incandescence(&status);	//selfillumination
								if(status) material[m].selfIllum = R2Y*si_color.r +G2Y*si_color.g +B2Y*si_color.b;
								MColor tr_color = fnShader.transparency(&status);	//transparency
								if(status) material[m].transparency = R2Y*tr_color.r +G2Y*tr_color.g +B2Y*tr_color.b;
								material[m].refraction_id  = fnShader.refractiveIndex(&status);	//refraction
								MPlug reflectPlug = fnShader.findPlug( "reflectionSpecularity", &status );	//reflection
								if(status) reflectPlug.getValue(material[m].reflection);
								material[m].gloss = 2.0f;
								material[m].specular = 0.0f;
								material[m].specular = 0.0f;
							}
							break;
						}
					}
				}


				material[m].MO = new MObject(ArrayOfSharers[i]);
				material[m].ntex = 0;
				//shader's name
				MFnDependencyNode fndn(ArrayOfSharers[i]);
				sprintf(material[m].matname, "%s", fndn.name().asChar());
				material[m].texname[0][0] = 0;
				material[m].texname[1][0] = 0;
				material[m].texname[2][0] = 0;
				material[m].texname[3][0] = 0;

				//------------------------------------------------------------------------
				//find lambert
				MItDependencyGraph DpGraph(  ArrayOfSharers[i],
																		 MFn::kLambert,
																		 MItDependencyGraph::kUpstream,
																		 MItDependencyGraph::kBreadthFirst,
																		 MItDependencyGraph::kNodeLevel, &status );
				DpGraph.disablePruningOnFilter();
				MObject LambertNode = DpGraph.thisNode();
	
				//------------------------------------------------------------------------
				status = DpGraph.resetTo( LambertNode, MFn::kLayeredTexture,
												MItDependencyGraph::kUpstream,
												MItDependencyGraph::kBreadthFirst,
												MItDependencyGraph::kNodeLevel);
				DpGraph.enablePruningOnFilter();

				//if layered texture found
				if(status)
				{
					status = DpGraph.resetTo(DpGraph.thisNode(), MFn::kFileTexture,
													MItDependencyGraph::kUpstream,
													MItDependencyGraph::kBreadthFirst,
													MItDependencyGraph::kNodeLevel);
					//all layers
					if(status)
					{
						while(!DpGraph.isDone())
						{
							long curlayer=-1;

							//search associated textures
							for(long s=0; s<uvsets; s++)
							{
								MObjectArray texArr;
								fnMesh.getAssociatedUVSetTextures(msArray[s], texArr);

								//curlayer = -1 if no textures for this set
								//if(texArr.length()==0)	break;

								//if map1 uvset and 2 textures associated so
								if(uvsets==1 && texArr.length()==2)	DpGraph.next();

								//for all textures in this uvset
								for(long t=0; t<texArr.length(); t++)
									if(DpGraph.thisNode()==texArr[t])	break;

								if(t<texArr.length())
								{
									curlayer = uvsettype[s];
									break;
								}
							}

							//if color or bump
							if(curlayer>=0)
							{
								MFnDependencyNode NodeFnDn;
								NodeFnDn.setObject( DpGraph.thisNode() );
								MPlug plugToFile = NodeFnDn.findPlug("fileTextureName",&status);
								MObject fnameValue;
								plugToFile.getValue( fnameValue );
								MFnStringData stringFn( fnameValue );
								MString nm = stringFn.string();

								strcpy(material[m].texname[curlayer], stringFn.string().asChar());
								GetFloatValues(DpGraph.thisNode(), "repeatUV", material[m].repeat[curlayer][0], material[m].repeat[curlayer][1]);
								GetFloatValues(DpGraph.thisNode(), "offset", material[m].offset[curlayer][0], material[m].offset[curlayer][1]);
								GetDoubleValue(DpGraph.thisNode(), "rotateFrame", material[m].rotate[curlayer]);

								material[m].ntex++;
							}

							DpGraph.next();
						}
					}
				}
				else	//try to find file texture
				{
					//base texture---------------------------------------
					status = DpGraph.resetTo(LambertNode, MFn::kFileTexture,
													MItDependencyGraph::kUpstream,
													MItDependencyGraph::kBreadthFirst,
													MItDependencyGraph::kNodeLevel);
					DpGraph.enablePruningOnFilter();
					if(status && !DpGraph.isDone())
					{
						MFnDependencyNode NodeFnDn;
						NodeFnDn.setObject( DpGraph.thisNode() );
						MPlug plugToFile = NodeFnDn.findPlug("fileTextureName",&status);
						MObject fnameValue;
						plugToFile.getValue( fnameValue );
						MFnStringData stringFn( fnameValue );
						MString nm = stringFn.string();

						strcpy(material[m].texname[0], stringFn.string().asChar());
						GetFloatValues(DpGraph.thisNode(), "repeatUV", material[m].repeat[0][0], material[m].repeat[0][1]);
						GetFloatValues(DpGraph.thisNode(), "offset", material[m].offset[0][0], material[m].offset[0][1]);
						GetDoubleValue(DpGraph.thisNode(), "rotateFrame", material[m].rotate[0]);

						material[m].ntex++;
					}

					//bump texture---------------------------------------
					status = DpGraph.resetTo(LambertNode, MFn::kBump,
													MItDependencyGraph::kUpstream,
													MItDependencyGraph::kBreadthFirst,
													MItDependencyGraph::kNodeLevel);

					DpGraph.disablePruningOnFilter();
					if(status && !DpGraph.isDone())
					{
						status = DpGraph.resetTo(DpGraph.thisNode(), MFn::kFileTexture,
														MItDependencyGraph::kUpstream,
														MItDependencyGraph::kBreadthFirst,
														MItDependencyGraph::kNodeLevel);
						//texture file
						DpGraph.enablePruningOnFilter();
						if(status && !DpGraph.isDone())
						{
							MFnDependencyNode NodeFnDn;
							NodeFnDn.setObject( DpGraph.thisNode() );
							MPlug plugToFile = NodeFnDn.findPlug("fileTextureName",&status);
							MObject fnameValue;
							plugToFile.getValue( fnameValue );
							MFnStringData stringFn( fnameValue );
							MString nm = stringFn.string();
							strcpy(material[m].texname[1], stringFn.string().asChar());
							GetFloatValues(DpGraph.thisNode(), "repeatUV", material[m].repeat[1][0], material[m].repeat[1][1]);
							GetFloatValues(DpGraph.thisNode(), "offset", material[m].offset[1][0], material[m].offset[1][1]);
							GetDoubleValue(DpGraph.thisNode(), "rotateFrame", material[m].rotate[1]);

							material[m].ntex++;
						}
					}
				}
				if(material[m].ntex>0)
				{
					nummat++;
					if(nummat==1024)
					{
						Error(std::string("Internal error: too much materials"));
						depth--;
						return;
					}
				}

				Log( "%s\n", material[m].matname);
				Log( "%f\n", material[m].selfIllum);
				Log( "%f\n", material[m].diffuse);
				Log( "%f\n", material[m].gloss);
				Log( "%f\n", material[m].specular);
				Log( "%d\n", material[m].brdf);
				Log( "%f\n", material[m].reflection);
				Log( "%f\n", material[m].refraction_id);
				Log( "%f\n", material[m].transparency);
				for(long ti=0; ti<4; ti++)
				{
					if(material[m].texname[ti][0]==0)	continue;
					Log( "%s\n", material[m].texname[ti]);
				}
				Log( "\n\n");//*/
			}

			exp_mtl.brdf       = material[m].brdf;
			exp_mtl.diffuse    = material[m].diffuse;
			exp_mtl.specular   = material[m].specular;
			exp_mtl.gloss      = material[m].gloss;
			exp_mtl.selfIllum  = material[m].selfIllum;
			exp_mtl.transparency = material[m].transparency;
			exp_mtl.refraction_id= material[m].refraction_id;
			exp_mtl.reflection   = material[m].reflection;
			exp_mtl.group_name = std::string( "unknown material group" );
			exp_mtl.name       = string(material[m].matname);

			//fill material
			for(long ti=0; ti<4; ti++)
			{
				exp_mtl.texture[ti].id   = 0;
				exp_mtl.texture[ti].type = EXPORT_STATIC::NONE;
				if(material[m].texname[ti][0]==0)	continue;

				switch(ti)
				{
					case 0:
						exp_mtl.texture[0].type = EXPORT_STATIC::BASE;
					break;
					case 1:
						exp_mtl.texture[1].type = EXPORT_STATIC::NORMAL;
					break;
					case 2:
						exp_mtl.texture[2].type = EXPORT_STATIC::BASE;
					break;
					case 3:
						exp_mtl.texture[3].type = EXPORT_STATIC::BASE;
					break;
				}

				char *ctemp = strrchr(material[m].texname[ti],'\\');
				if(ctemp==0)	ctemp = strrchr(material[m].texname[ti],'/');
				if(ctemp==0)	ctemp = material[m].texname[ti];
				else ctemp++;

				exp_mtl.texture[ti].name = string(ctemp);
			}

			//add material
			//if(material[m].ntex>0)//NOTE_IT
			{
				exp_obj.mtl.push_back(exp_mtl);
				matref[i] = m;
			}
		}

		//------------------------------------------------------------------------
		//transformation
		//------------------------------------------------------------------------
		//mtx * path_mtx * INVERSE(locator_mtx)
		//vrt = v*mtx -> vertex is reletive to locator
		MMatrix locator_pos(locator_mtx);
		MMatrix m = path.inclusiveMatrix() * locator_pos;
		float mtx[4][4];
		status = m.get(mtx);


		MTransformationMatrix mtrans = MTransformationMatrix(m);
		double scale[3];
		mtrans.getScale(scale, MSpace::kWorld);
		if(scale[0]*scale[1]*scale[2]<0.0f)	opposite = !opposite;
/*Log( "OBJECT_TRANSFORM: %s::%s\n", fdagnodeg.name(&status).asChar(), fdagnode.name(&status).asChar());
for(long ry=0; ry<4; ry++)
{
	for(long rx=0; rx<4; rx++)
		Log( "%f, ", mtx[ry][rx]);
	Log( "\n");
}
Log( "\n");//*/

		//------------------------------------------------------------------------
		//polygon
		//------------------------------------------------------------------------
		//get vertex coord
		MPointArray mesh_vrt;
		fnMesh.getPoints(mesh_vrt);

		MFloatVectorArray meshNorm;
		fnMesh.getNormals(meshNorm, MSpace::kWorld);

		BONEWEIGHT *ani = new BONEWEIGHT[fnMesh.numVertices()];
		if(GlobAnimation)	SaveAnimation(ani, path, fnMesh.numVertices());
		//get colors
		MColorArray mesh_col;
		fnMesh.getFaceVertexColors(mesh_col);

		//get points
		MFloatPointArray vertexArray;
		fnMesh.getPoints(vertexArray);

		//polygon iterator
		MItMeshPolygon itPoly(path);

		for(long p=0; p<fnMesh.numPolygons(); p++)
		{
			//get triangles
			if(itPoly.hasValidTriangulation()==false)
			{
				itPoly.next();
				continue;
			}
			exp_trg.material = ArrayOfIndex[p];
			long mmt = matref[exp_trg.material];

			MIntArray vrtId;
			itPoly.getVertices(vrtId);
			
			MPointArray points;
			MIntArray vertexList;
			int nTrgs;
			itPoly.numTriangles(nTrgs);
			for(int nt=0; nt<nTrgs; nt++)
			{
				itPoly.getTriangle(nt, points, vertexList);
				for(long v=0; v<3; v++)
				{

					for(int faceRelIndex = 0; faceRelIndex<vrtId.length(); faceRelIndex++)
						if(vrtId[faceRelIndex]==vertexList[v])
							break;

					//color-per-vertex
					int colidx;
					fnMesh.getFaceVertexColorIndex(p, v, colidx);
					if(mesh_col[colidx].a<0.0f)
						//no colors here
						exp_trg.vrt[v].color = 0xFF7F7F7F;
					else
					{
						long a = long(255.0f*mesh_col[colidx].a);
						long r = min(127, long(127.0f*mesh_col[colidx].r));
						long g = min(127, long(127.0f*mesh_col[colidx].g));
						long b = min(127, long(127.0f*mesh_col[colidx].b));
						exp_trg.vrt[v].color = (a<<24)|(r<<16)|(g<<8)|b;
						//Log( "%lu\n", exp_trg.vrt[v].color);
					}

					//bones and weights
					exp_trg.vrt[v].boneid[0] = ani[vertexList[v]].idBone[0];
					exp_trg.vrt[v].boneid[1] = ani[vertexList[v]].idBone[1];
					exp_trg.vrt[v].boneid[2] = ani[vertexList[v]].idBone[2];
					exp_trg.vrt[v].boneid[3] = ani[vertexList[v]].idBone[3];
					exp_trg.vrt[v].weight[0] = ani[vertexList[v]].weight[0];
					exp_trg.vrt[v].weight[1] = ani[vertexList[v]].weight[1];
					exp_trg.vrt[v].weight[2] = ani[vertexList[v]].weight[2];
					exp_trg.vrt[v].weight[3] = ani[vertexList[v]].weight[3];
					//Log( "%f, %f, %f, %f\n", exp_trg.vrt[v].weight[0], exp_trg.vrt[v].weight[1], exp_trg.vrt[v].weight[2], exp_trg.vrt[v].weight[3]);

					//position
					float dx = mesh_vrt[vertexList[v]].x;
					float dy = mesh_vrt[vertexList[v]].y;
					float dz = mesh_vrt[vertexList[v]].z;
					exp_trg.vrt[v].x = dx*mtx[0][0] + dy*mtx[1][0] + dz*mtx[2][0] + mtx[3][0];
					exp_trg.vrt[v].y = dx*mtx[0][1] + dy*mtx[1][1] + dz*mtx[2][1] + mtx[3][1];
					exp_trg.vrt[v].z = dx*mtx[0][2] + dy*mtx[1][2] + dz*mtx[2][2] + mtx[3][2];
					exp_trg.vrt[v].x *= CM2M_SCALE;
					exp_trg.vrt[v].y *= CM2M_SCALE;
					exp_trg.vrt[v].z *= CM2M_SCALE;

					//normal
					MVector norm;
					itPoly.getNormal(faceRelIndex, norm, MSpace::kWorld);
					exp_trg.vrt[v].nx = norm.x;
					exp_trg.vrt[v].ny = norm.y;
					exp_trg.vrt[v].nz = norm.z;

					if(!GlobAnimation)
					{
						exp_trg.vrt[v].x *= -1.0f;
						exp_trg.vrt[v].nx *= -1.0f;
					}

					/*if(opposite)	//reverce normal
					{
						exp_trg.vrt[v].nx *= -1.0f;
						exp_trg.vrt[v].ny *= -1.0f;
						exp_trg.vrt[v].nz *= -1.0f;
					}//*/

					// UV set
					for(long s=0; s<uvsets; s++)
						if(uvsettype[s]!=-1)
						{
							float U,V;
							float uvp[2];
							status = itPoly.getUV( faceRelIndex, uvp, &msArray[s]);
							if(!status)
							{
								Log("can't get UV[%d] for uvSet: %d\n", uvsettype[s], s);
								exp_trg.vrt[v].tu[uvsettype[s]] = 0.62109375f;
								exp_trg.vrt[v].tv[uvsettype[s]] = 0.62109375f;
								continue;
							}
							U = uvp[0];
							V = uvp[1];

							float ang = -material[mmt].rotate[uvsettype[s]];
							float tU,tV;
 							tU = (U-0.5f)*cosf(ang) + (V-0.5f)*sinf(ang) + 0.5f;
							tV = (V-0.5f)*cosf(ang) - (U-0.5f)*sinf(ang) + 0.5f;
							tU *= material[mmt].repeat[uvsettype[s]][0];
							tV *= material[mmt].repeat[uvsettype[s]][1];
							tU += material[mmt].offset[uvsettype[s]][0];
							tV += material[mmt].offset[uvsettype[s]][1];

							switch(uvsettype[s])
							{
								case 0:
									exp_trg.vrt[v].tu[0] = tU;
									exp_trg.vrt[v].tv[0] = -tV;
								break;
								case 1:
									exp_trg.vrt[v].tu[1] = tU;
									exp_trg.vrt[v].tv[1] = -tV;
								break;
							}

							if(uvsets==1)
								for(long nt=1; nt<material[mmt].ntex; nt++)
								{
									float ang = material[mmt].rotate[nt];
 									tU = (U-0.5f)*cosf(ang) + (V-0.5f)*sinf(ang) + 0.5f;
									tV = (V-0.5f)*cosf(ang) - (U-0.5f)*sinf(ang) + 0.5f;
									tU *= material[mmt].repeat[nt][0];
									tV *= material[mmt].repeat[nt][1];
									tU += material[mmt].offset[nt][0];
									tV += material[mmt].offset[nt][1];

									exp_trg.vrt[v].tu[nt] = tU;
									exp_trg.vrt[v].tv[nt] = -tV;
								}
						}//*/

				}
				//we use CCW, Maya uses CW back-face culling
				if(!opposite)
				{
					EXPORT_STATIC::VERTEX tvrt = exp_trg.vrt[0];
					exp_trg.vrt[0] = exp_trg.vrt[1];
					exp_trg.vrt[1] = tvrt;
				}
				exp_obj.trg.push_back(exp_trg);
			}

			itPoly.next();
		}
		delete ani;
		//------------------------------------------------------------------------
		//add object
		//------------------------------------------------------------------------
	  try
		{
			rexp->AddObject(exp_obj);
		}
	  catch(char* errmsg)
	  {
			MString command = "text -edit -label \"";
			command += errmsg;
			command += "\" MyLable";
			MGlobal::executeCommand(command,true);
			MGlobal::executeCommand("showWindow StatusWin",true);
		  Sleep(2000);
		}
	}
SkipObj:;
	//--------------------------------------------------------
	//recurse all children
	//--------------------------------------------------------
	long nc = path.childCount(&status);
	for(long c=0; c<nc; c++)
	{
		MObject chlo = path.child(c, &status);
		MDagPath cl = path;
		cl.push(chlo);

		//export locator
		if(cl.hasFn(MFn::kLocator) && cl.hasFn(MFn::kTransform))
		{
			ExportLocator(cl);
		}
		else	Traverse(cl);
	}

	//if animation export we needs to link all other locators to the root
	if(GlobAnimation && path.hasFn(MFn::kLocator) && path.hasFn(MFn::kTransform))
	{
		MDagPath gpath(path);
		gpath.pop();

		//if parent of this locator is a joint - skip this
		//if(!gpath.hasFn(MFn::kJoint))
		MFnDagNode fdagnodeg(gpath);
		if(strcmpi(fdagnodeg.name().asChar(), "world")==0)
		{
			Log("#######: %s\n", fdagnodeg.name().asChar());

			MSelectionList slist;
			MGlobal::getActiveSelectionList(slist);
			MItSelectionList iter(slist);

			for( ; !iter.isDone(); iter.next() ) 
			{
				MDagPath exLoc;
				status = iter.getDagPath( exLoc );

				//if the node is a root
				if(path==exLoc)	continue;

				//only locators 
				if(exLoc.hasFn(MFn::kLocator) && exLoc.hasFn(MFn::kTransform))
				{
					ExportLocator(exLoc);
				}
			}
		}
	}
	depth--;
}

void  SaveRTXtexture(long id)
{
  //printf(" SaveRTXtexture(%d) \n",id);
}

void AtlaExport::ExportAll(const char *fname, bool scale, bool bsp, bool product)
{

  HINSTANCE dll_inst = LoadLibrary("export.dll");
  if(dll_inst==NULL)
	{
		Error(std::string("can't load export.dll"));
		return;
	}
	CREATE_EXPORT_FUNC CreateExporter = (CREATE_EXPORT_FUNC) GetProcAddress(dll_inst, "CreateExporter");
	if(CreateExporter==NULL)
	{
		Error(std::string("export.dll: CreateExporter func not found"));
		return;
	}

	FILE *errfl = fopen("error.txt", "w+");
	fclose(errfl);

	curobject = totobjects = 0;

	char partname[256];
	sprintf(partname, "%s", fname);
	for(long s=0; s<strlen(partname)-2; s++)
		if(partname[s]=='.' && partname[s+1]=='g' && partname[s+2]=='m')
		{
			partname[s] = 0;
			break;
		}


  MSelectionList slist;
  MGlobal::getActiveSelectionList(slist);
	MItSelectionList iter(slist);

	MItDag dagIterator( MItDag::kDepthFirst, MFn::kInvalid, &status);

	MGlobal::executeCommand("window -tlc 300 300 -w 300 -h 200 -title \"RDF Export\" StatusWin",true);
	MGlobal::executeCommand("columnLayout -adjustableColumn true",true);
	MGlobal::executeCommand("text -label \"Default\" MyLable",true);
	MGlobal::executeCommand("showWindow StatusWin",true);


	MGlobal::executeCommand("text -edit -label \"Search root\" MyLable",true);
	MGlobal::executeCommand("showWindow StatusWin",true);

  MTime startFrame( MAnimControl::minTime().value(), MTime::uiUnit() );
  float startTime = (float)startFrame.value();
	MTime Time(startTime, MTime::uiUnit());
	MAnimControl::setCurrentTime(Time);

	//search for many roots
	vector<string> rootname;
	for( ; !iter.isDone(); iter.next() ) 
	{
		MDagPath objectPath;
		status = iter.getDagPath( objectPath );

		//only locators 
		if(objectPath.hasFn(MFn::kLocator) && objectPath.hasFn(MFn::kTransform))
		{
			const char *oname = objectPath.fullPathName().asChar();
			long len0 = strlen(oname);

			//search for parent or child and remove children
			for(long r=0; r<rootname.size(); r++)
			{
				long len1 = strlen(rootname[r].c_str());

				//if parent or child
				if(memcmp(oname, rootname[r].c_str(), min(len0, len1))==0)
				{
					//if parent
					if(len0>=len1)	break;

					//remove child
					rootname.erase(rootname.begin() + r);
				}

			}

			//add new entry
			if(r==rootname.size())
				rootname.push_back(oname);

			//only one root for animated model
			//if(GlobAnimation)	break;

		}
	}

	//-------------------------------------------------------------------
	//calculate number of objects
	locators.clear();
	iter.reset();
	for( ; !iter.isDone(); iter.next() ) 
	{
		MDagPath objectPath;
		status = iter.getDagPath( objectPath );

		retobj = 0;
		//only locators 
		if(objectPath.hasFn(MFn::kLocator) && objectPath.hasFn(MFn::kTransform))
		{
			PreTraverse(objectPath);
			//this locator has geometry data and must be marked under "geometry" group
			if(retobj>0)
			{
				MFnDagNode fdagnode(objectPath);
				locators.push_back(string(fdagnode.name().asChar()));
				totobjects += retobj;
			}
		}
	}

	long root_loc = 9999;
	for(long n=0; n<rootname.size(); n++)
		root_loc = min(root_loc, strlen(rootname[n].c_str()));
	//-------------------------------------------------------------------
	//export all selected locators
	iter.reset();
	for( ; !iter.isDone(); iter.next() ) 
	{
		MDagPath objectPath;
		status = iter.getDagPath( objectPath );

		//only locators 
		if(objectPath.hasFn(MFn::kLocator) && objectPath.hasFn(MFn::kTransform))
		{
			MFnDagNode fdagnode(objectPath);
			const char *pname = fdagnode.name(&status).asChar();

			//skip locators without any objects attached
			for(long l=0; l<locators.size(); l++)
				if(locators[l]==string(pname))	break;
			if(l==locators.size())	continue;

			char newname[256];
			sprintf(newname, "%s_%s.atg", partname, pname);

			//find root
			if(GlobAnimation || rootname.size()==1 && root_loc == strlen(objectPath.fullPathName().asChar()))
				sprintf(newname, "%s.atg", partname);

#ifdef LOG
			LogFile = fopen(newname, "w");
#endif
			nummat=0;

			//------------------------------------------------------------------------------
			char gmname[256];
			strcpy(gmname, newname);
			gmname[strlen(gmname)-3] = 'g';
			gmname[strlen(gmname)-2] = 'm';
			gmname[strlen(gmname)-1] = 0;
			long flags = 0;
			if(bsp)	flags |= BUILD_BSP;
			//if(product)	flags |= PRODUCTION;
			if(GlobAnimation)	flags |= ANIMATION;
			rexp = CreateExporter(gmname, SaveRTXtexture, Result, Error, 1e10f, 1.0f, flags);
			//------------------------------------------------------------------------------
			//unsigned long time = GetTickCount();

			//create locator matrix
			MMatrix localRotation;
			MMatrix locator_m = GetLocator(objectPath, localRotation);

			locator_m.inverse();

			locator_m[0][0] = locator_m[1][1] = locator_m[2][2] = locator_m[3][3] = 1.0f;
			locator_m[0][1] = locator_m[0][2] = locator_m[0][3] = 0.0f;
			locator_m[1][0] = locator_m[1][2] = locator_m[1][3] = 0.0f;
			locator_m[2][0] = locator_m[2][1] = locator_m[2][3] = 0.0f;
			locator_m[3][0] *= -1.0f;	locator_m[3][1] *= -1.0f;	locator_m[3][2] *= -1.0f;

			locator_m = localRotation.transpose()*locator_m;
			//rotation component will be stored
			locator_m.get(locator_mtx);

			//set only position

			Log( "LOCATOR\n");
			for(long ry=0; ry<4; ry++)
			{
				for(long rx=0; rx<4; rx++)
					Log( "%f, ", locator_mtx[ry][rx]);
				Log( "\n");
			}
			Log( "\n");


			for(; nummat>0; nummat--)
				material[nummat-1].MO = 0;

			mJointsList.clear();
			Traverse(objectPath);

			//------------------------------------------------------------------------------
			rexp->WriteFile();

			STARTUPINFO si;
			memset(&si, 0, sizeof si);
			si.cb = sizeof si;

			PROCESS_INFORMATION pi;
			memset(&pi, 0, sizeof pi);


			BOOL created = false;
			DWORD err;
			if(ExportCharacterPatch)
			{
				char cd[_MAX_PATH];
				MakePathForTraceMaker(cd);

				char arg[512];
				strcpy(arg, cd);
				strcat(arg, " ");
				strcat(arg, gmname);
				created = CreateProcess(cd, arg, 0, 0, FALSE, /*CREATE_NEW_CONSOLE*/DETACHED_PROCESS|NORMAL_PRIORITY_CLASS, 0, 0, &si, &pi);
				err = GetLastError();
			}
			delete rexp;
			//------------------------------------------------------------------------------

#ifdef LOG
		fclose(LogFile);
#endif
		}
	}

  FreeLibrary(GetModuleHandle("export.dll"));
	//fclose(trace);
  MGlobal::executeCommand("deleteUI -window StatusWin;");
}

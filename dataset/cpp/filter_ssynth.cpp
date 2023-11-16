#include <Qt>
#include "filter_ssynth.h"
#include <meshlabplugins/io_x3d/import_x3d.h>
#include <common/ml_document/mesh_model.h>
#include <StructureSynth/Model/RandomStreams.h>
#include <StructureSynth/Parser/Preprocessor.h>
#undef __GLEW_H__ //terrible workaround to avoid problem with #warning in visual studio
#include "mytrenderer.h"
#include <StructureSynth/Parser/Tokenizer.h>
#include <StructureSynth/Parser/EisenParser.h>
#include <StructureSynth/Model/Builder.h>

using namespace std;
using namespace vcg;
using namespace vcg::tri::io;
using namespace StructureSynth::Parser;
using namespace StructureSynth::Model;
using namespace StructureSynth::Model::Rendering;
using namespace SyntopiaCore::Exceptions;


FilterSSynth::FilterSSynth()
{
	typeList = {CR_SSYNTH};
	this->renderTemplate= "";
	for(ActionIDType tt : types())
		actionList.push_back(new QAction(filterName(tt), this));
}

QString FilterSSynth::pluginName() const
{
    return "FilterSSynth";
}

QString FilterSSynth::filterName(ActionIDType filter) const
{
	switch (filter) {
	case CR_SSYNTH: return QString("Structure Synth Mesh Creation"); break;
	default: assert(0); return QString();
	}
}

QString FilterSSynth::pythonFilterName(ActionIDType f) const
{
	switch (f) {
	case CR_SSYNTH: return QString("create_mesh_by_grammar"); break;
	default: assert(0); return QString();
	}
}

QString FilterSSynth::filterInfo(ActionIDType filterId) const
{
    switch(filterId)
    {
        case CR_SSYNTH:
            return QString("Structure Synth mesh creation based on Eisen Script.\n For further instruction visit http://structuresynth.sourceforge.net/reference.php");
            break;
        default:
            assert(0); return QString("error");
    }
}

RichParameterList FilterSSynth::initParameterList(const QAction* /*filter*/,const MeshDocument &/*md*/)
{
    RichParameterList par;
    par.addParam(RichString("grammar","set maxdepth 40 R1 R2 rule R1 { { x 1 rz 6 ry 6 s 0.99 } R1 { s 2 } sphere } rule R2 {{ x -1 rz 6 ry 6 s 0.99 } R2 { s 2 } sphere} ","Eisen Script grammar","Write a grammar according to Eisen Script specification and using the primitives box, sphere, mesh, dot and triangle "));
    par.addParam(RichInt("seed",1,"seed for random construction","Seed needed to build the mesh"));
    par.addParam(RichInt("sphereres",1,"set maximum resolution of sphere primitives, it must be included between 1 and 4","increasing the resolution of the spheres will improve the quality of the mesh "));
    return par;
}

void FilterSSynth::openX3D(const QString &fileName, MeshModel &m, int& mask, vcg::CallBackPos *cb, QWidget* /*parent*/)
{
    vcg::tri::io::AdditionalInfoX3D* info = NULL;
    /*int result = */vcg::tri::io::ImporterX3D<CMeshO>::LoadMask(fileName.toStdString().c_str(), info);
    m.enable(info->mask);
    /*result = */vcg::tri::io::ImporterX3D<CMeshO>::Open(m.cm, fileName.toStdString().c_str(), info, cb);
    /*vcg::tri::UpdateBounding<CMeshO>::Box(m.cm);
    vcg::tri::UpdateNormal<CMeshO>::PerVertexNormalizedPerFaceNormalized(m.cm);*/
    m.updateBoxAndNormals();
    mask=info->mask;
    delete(info);
}

std::map<std::string, QVariant> FilterSSynth::applyFilter(
		const QAction* filter,
		const RichParameterList & par,
		MeshDocument &md,
		unsigned int& /*postConditionMask*/,
		vcg::CallBackPos *cb)
{
	if (ID(filter) == CR_SSYNTH) {
		md.addNewMesh("",this->filterName(ID(filter)));
		QString grammar = par.getString("grammar");
		int seed = par.getInt("seed");
		int sphereres=par.getInt("sphereres");
		this->renderTemplate = GetTemplate(sphereres);
		if(this->renderTemplate != ""){
			QString path=ssynth(grammar,-50,seed,cb);
			if (! QFile::exists(path)){
				QString message=QString("An error occurred during the mesh generation: " ).append(path);
				throw MLException(message);
			}
			else {
				QFile file(path);
				int mask;
				QString name(file.fileName());
				openX3D(name,*(md.mm()),mask,cb);
				file.remove();
			}
		}
		else{
			throw MLException("Error: Sphere resolution must be between 1 and 4");
		}
	}
	else {
		wrongActionCalled(filter);
	}
	return std::map<std::string, QVariant>();
}

int FilterSSynth::getRequirements(const QAction *)
{
    return MeshModel::MM_NONE;
}

QString FilterSSynth::ssynth(QString grammar,int maxdepth,int seed,CallBackPos *cb){
    QString path("");
    if (cb != NULL)		(*cb)(0, "Loading...");
    Template templ(this->renderTemplate);
    MyTrenderer renderer(templ);
    renderer.begin();
    Preprocessor pp;
    QString out = pp.Process(grammar);
    Tokenizer token(out);
    EisenParser parser(&token);
    try
    {
        RuleSet* rs=parser.parseRuleset();
        rs->resolveNames();
        rs->dumpInfo();
        if(maxdepth>0)rs->setRulesMaxDepth(maxdepth);
        RandomStreams::SetSeed(seed);
        Builder b(&renderer,rs,false);
        b.build();
        renderer.end();
        QString output=renderer.getOutput();
        (*cb)(0, "Temp");
        QFile file(QDir::tempPath() + "/output.x3d");
        if(!file.open(QFile::WriteOnly | QFile::Text)){(*cb)(0, "File has not been opened"); return QString("");}
        QTextStream outp(&file);
        outp << output;
        file.close();
        path=file.fileName();
        if (cb != NULL){	(*cb)(99, "Done");}
    }
    catch(Exception& ex){
        return ex.getMessage();
    }
    return path;
}

int FilterSSynth::postCondition(const QAction* /*filter*/) const
{
    return MeshModel::MM_NONE;
}

FilterPlugin::FilterClass FilterSSynth::getClass(const QAction */*filter*/) const
{
    return FilterPlugin::MeshCreation;
}

std::list<FileFormat> FilterSSynth::importFormats() const
{
	return {FileFormat("Eisen Script File", tr("ES"))};
}

std::list<FileFormat> FilterSSynth::exportFormats() const
{
	std::list<FileFormat> formats;
	return formats ;
}

void FilterSSynth::open(
		const QString& formatName,
		const QString& fileName,
		MeshModel &m,
		int& mask,
		const RichParameterList & par,
		CallBackPos *cb)
{
	if (formatName.toUpper() == tr("ES")){
		this->seed=par.getInt("seed");
		int maxrec=par.getInt("maxrec");
		int sphereres=par.getInt("sphereres");
		int maxobj=par.getInt("maxobj");
		this->renderTemplate=GetTemplate(sphereres);
		if(this->renderTemplate!= ""){
			QFile grammar(fileName);
			grammar.open(QFile::ReadOnly|QFile::Text);
			QString gcontent(grammar.readAll());
			grammar.close();
			if(maxrec>0)ParseGram(&gcontent,maxrec,tr("set maxdepth"));
			if(maxobj>0)ParseGram(&gcontent,maxobj,tr("set maxobjects"));
			QString x3dfile(FilterSSynth::ssynth(gcontent,maxrec,this->seed,cb));
			if(!QFile::exists(x3dfile)){
				throw MLException("An error occurred during the mesh generation: " + x3dfile);
			}
			else{
				openX3D(x3dfile,m,mask,cb);
				QFile x3df(x3dfile);
				x3df.remove();
			}
		}
		else{
			throw MLException("Error: Sphere resolution must be between 1 and 4");
		}
	}
	else {
		wrongOpenFormat(formatName);
	}
}

void FilterSSynth::save(const QString &formatName, const QString &/*fileName*/, MeshModel &/*m*/, const int /*mask*/, const RichParameterList &, vcg::CallBackPos */*cb*/)
{
	wrongSaveFormat(formatName);
}

void FilterSSynth::exportMaskCapability(const QString &/*format*/, int &/*capability*/, int &/*defaultBits*/) const {}

RichParameterList FilterSSynth::initPreOpenParameter(const QString &/*formatName*/) const
{
    RichParameterList parlst;
    parlst.addParam(RichInt(tr("seed"),1,tr("Seed for random mesh generation"),tr("write a seed for the random generation of the mesh")));
    parlst.addParam(RichInt("maxrec",0,"set the maximum recursion","the mesh is built recursively according to the productions of the grammar, so a limit is needed. If set to 0 meshlab will generate the mesh according to the maximum recursion set in the file"));
    parlst.addParam(RichInt("sphereres",1,"set maximum resolution of sphere primitives, it must be included between 1 and 4","increasing the resolution of the spheres will improve the quality of the mesh "));
    parlst.addParam(RichInt("maxobj",0,"set the maximum number of object to be rendered","you can set a limit to the maximum number of primitives rendered. If set to 0 meshlab will generate the mesh according to the input file"));
    return parlst;
}

QString FilterSSynth::GetTemplate(int sphereres){
    QString filen;
    switch(sphereres){
        case 1:
            filen=":/x3d.rendertemplate";
            break;
        case 2:
            filen=":/x3d2.rendertemplate";
            break;
        case 3:
            filen=":/x3d3.rendertemplate";
            break;
        case 4:
            filen=":/x3d4.rendertemplate";
            break;
        default:
            return QString::Null();
            break;
    }
    QFile tr(filen);
    tr.open(QFile::ReadOnly|QFile::Text);
    QString templateR(tr.readAll());
    return templateR;
}

void FilterSSynth::ParseGram(QString* grammar, int max,QString pattern){
    int idx=grammar->indexOf(pattern);
    if(idx>-1){
        int end=pattern.length()+idx;
        while(!grammar->operator [](end).isNumber())
            end++;
        QString grec;
        while(grammar->operator [](end).isNumber()){
            grec.append(grammar->operator [](end));
            end++;
        }
        QString tosub=QString(pattern).append(" ").append(QString::number(max)).append(" ");
        QString maxrestr=grammar->mid(idx,end-idx);
        /* if(pattern=="set maxobjects")grammar->replace(maxrestr,tosub);
                 else if(grec.toInt()<max)grammar->replace(maxrestr,tosub);*/
        grammar->replace(maxrestr,tosub);
    }

    else if(pattern=="set maxobjects"){
        QString tosub=QString(pattern).append(" ").append(QString::number(max)).append(" \n");
        grammar->insert(0,tosub);
    }
}

MESHLAB_PLUGIN_NAME_EXPORTER(FilterSSynth)

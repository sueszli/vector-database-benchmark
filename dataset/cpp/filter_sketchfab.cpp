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

#include "filter_sketchfab.h"
#include <QHttpPart>
#include <QHttpMultiPart>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QSettings>
#include <QApplication>
#include <QDir>
#include <wrap/io_trimesh/export_ply.h>
#include "miniz.h"

FilterSketchFabPlugin::FilterSketchFabPlugin()
{ 
	typeList = {FP_SKETCHFAB};

	for(ActionIDType tt : types())
		actionList.push_back(new QAction(filterName(tt), this));
}

QString FilterSketchFabPlugin::pluginName() const
{
	return "FilterSketchFab";
}

QString FilterSketchFabPlugin::filterName(ActionIDType filterId) const
{
	switch (filterId) {
	case FP_SKETCHFAB: return "Export to Sketchfab";
	default: assert(0); return QString();
	}
}

QString FilterSketchFabPlugin::pythonFilterName(ActionIDType f) const
{
	switch (f) {
	case FP_SKETCHFAB: return "export_mesh_to_sketchfab";
	default: assert(0); return QString();
	}
}

QString FilterSketchFabPlugin::filterInfo(ActionIDType filterId) const
{
	switch(filterId) {
	case FP_SKETCHFAB :
		return "Upload the current layer on Sketchfab. It requires that you have an account and that you set your private API token in the preferences of MeshLab.";
	default :
		assert(0);
		return "Unknown Filter";
	}
}

FilterSketchFabPlugin::FilterClass FilterSketchFabPlugin::getClass(const QAction *a) const
{
	switch(ID(a)) {
	case FP_SKETCHFAB :
		return FilterPlugin::Smoothing;
	default :
		assert(0);
		return FilterPlugin::Generic;
	}
}

FilterPlugin::FilterArity FilterSketchFabPlugin::filterArity(const QAction* a) const
{
	switch(ID(a)) {
	case FP_SKETCHFAB :
		return SINGLE_MESH;
	default:
		return NONE;
	}
}

int FilterSketchFabPlugin::getPreConditions(const QAction*) const
{
	return MeshModel::MM_NONE;
}

int FilterSketchFabPlugin::postCondition(const QAction*) const
{
	return MeshModel::MM_NONE;
}

RichParameterList FilterSketchFabPlugin::initParameterList(const QAction* action, const MeshModel&)
{
	RichParameterList parlst;
	QSettings settings;
	QVariant v = settings.value("SketchFab Code");
	QString sketchFabAPIValue;
	if (v == QVariant()) {
		sketchFabAPIValue = DEFAULT_API;
	}
	else {
		sketchFabAPIValue = v.toString();
	}
	switch(ID(action)) {
	case FP_SKETCHFAB :
		parlst.addParam(RichString("sketchFabKeyCode", sketchFabAPIValue, "SketchFab Account API token", "The API token of the account on which to upload the model.<br>You can find it by going on Settings -> Password and API -> API token."));
		parlst.addParam(RichString("title", "MeshLabModel", "Title", "Mandatory."));
		parlst.addParam(RichString("description", "A model generated with meshlab", "Description", "Mandatory. A short description of the model that is uploaded."));
		parlst.addParam(RichString("tags", "meshlab", "Tags", "Mandatory. Tags must be separated by a space. Typical tags usually used by MeshLab users: scan, photogrammetry."));
		parlst.addParam(RichBool("isPrivate", false, "Private", "This parameter can be true only for PRO account."));
		parlst.addParam(RichBool("isPublished", false, "Publish", "If true the model will be published immediately."));
		parlst.addParam(RichBool("autoRotate", true, "Auto Rotate", "If true the model rotated by 90 degree on the X axis to maintain similar default orientation."));
		parlst.addParam(RichBool("saveApiSetting", sketchFabAPIValue != DEFAULT_API, "Save SketchFab Code", "Saves the API SketchFab code into the MeshLab settings, allowing to load it as default value every time you run this filter."));
		break;
	default :
		assert(0);
	}
	return parlst;
}

std::map<std::string, QVariant> FilterSketchFabPlugin::applyFilter(
		const QAction * action,
		const RichParameterList& par,
		MeshDocument& md,
		unsigned int& /*postConditionMask*/,
		vcg::CallBackPos* cb)
{
	std::map<std::string, QVariant> outValues;
	std::string url;
	switch (ID(action)) {
	case FP_SKETCHFAB:
		url = sketchfab(md, cb,
						 par.getString("sketchFabKeyCode"), par.getString("title"),
						 par.getString("description"), par.getString("tags"),
						 par.getBool("isPrivate"), par.getBool("isPublished"),
						 par.getBool("autoRotate"), par.getBool("saveApiSetting"));
		outValues["url_model"] = QVariant::fromValue(QString::fromStdString(url));
		break;
	default:
		wrongActionCalled(action);
	}
	return outValues;
}

std::string FilterSketchFabPlugin::sketchfab(
		MeshDocument& md,
		vcg::CallBackPos* cb,
		const QString& apiToken,
		const QString& name,
		const QString& description,
		const QString& tags,
		bool isPrivate,
		bool isPublished,
		bool /*autoRotate*/,
		bool saveApiSetting)
{
	qDebug("Export to SketchFab start ");
	this->fcb=cb;
	this->fcb(1,"Compressing Mesh");
	qDebug("APIToken = '%s' ",qUtf8Printable(apiToken));
	Matrix44m rot; rot.SetRotateDeg(-90,Point3m(1,0,0));
	Matrix44m rotI; rot.SetRotateDeg(90,Point3m(1,0,0));

	if(apiToken.isEmpty() || apiToken == DEFAULT_API) {
		throw MLException("Please set in the MeshLab preferences your private API Token string that you can find on the<a href=\"https://sketchfab.com/settings/password\">Sketchfab Password Settings.");
	}

	QSettings settings;
	if (saveApiSetting) {
		settings.setValue("SketchFab Code", apiToken);
	}
	else {
		settings.remove("SketchFab Code");
	}

	QString tmpObjFileName = QDir::tempPath() + "/xxxx.ply";
	QString tmpZipFileName = QDir::tempPath() + "/xxxx.zip";

	int mask=0;
	if(md.mm()->hasDataMask(MeshModel::MM_VERTCOLOR)) mask+=vcg::tri::io::Mask::IOM_VERTCOLOR;
	vcg::tri::UpdatePosition<CMeshO>::Matrix(md.mm()->cm,rot);
	vcg::tri::io::ExporterPLY<CMeshO>::Save(md.mm()->cm,qUtf8Printable(tmpObjFileName),mask,true);
	vcg::tri::UpdatePosition<CMeshO>::Matrix(md.mm()->cm,rotI);
	qDebug("Saved %20s",qUtf8Printable(tmpObjFileName));
	qDebug("Compressed %20s",qUtf8Printable(tmpZipFileName));
	saveMeshZip(qUtf8Printable(tmpObjFileName),"xxxx.ply",qUtf8Printable(tmpZipFileName));

	qDebug("Model Title %s %s %s\n",qUtf8Printable(name),qUtf8Printable(description),qUtf8Printable(tags));
	qDebug("Starting Upload");
	this->fcb(10,"Starting Upload");
	std::string urlModel;
	bool ret = upload(tmpZipFileName, apiToken, name, description, tags, QString::number(isPrivate), QString::number(isPublished), urlModel);
	if(!ret){
		throw MLException("Upload FAILED");
	}

	this->log("Upload Completed; you can access the uploaded model at the following URL:\n");
	this->log("<a href=\"%s\">%s</a>\n",qUtf8Printable(QString::fromStdString(urlModel)),qUtf8Printable(QString::fromStdString(urlModel)));
	return urlModel;
}

bool FilterSketchFabPlugin::upload(
		const QString& zipFileName,
		const QString& apiToken,
		const QString& name,
		const QString& description,
		const QString& tags,
		const QString& isPrivate,
		const QString& isPublished,
		std::string& urlModel)
{
	QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

	// upload parameters data
	multiPart->append(part_parameter("token", apiToken));
	multiPart->append(part_parameter("name", name));
	multiPart->append(part_parameter("description", description));
	multiPart->append(part_parameter("tags", tags));
	multiPart->append(part_parameter("private", isPrivate));
	multiPart->append(part_parameter("isPublished", isPublished));
	multiPart->append(part_parameter("source", "meshlab-exporter"));

	// upload file data
	QHttpPart modelPart;
	modelPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"modelFile\"; filename=\""+zipFileName+"\""));
	modelPart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("application/octet-stream"));
	QFile *zipfileToUpload = new QFile(zipFileName);

	zipfileToUpload->open(QIODevice::ReadOnly);
	modelPart.setBodyDevice(zipfileToUpload);
	zipfileToUpload->setParent(multiPart);
	multiPart->append(modelPart);

	QUrl url("https://api.sketchfab.com/v2/models");
	QNetworkRequest request(url);

	QNetworkAccessManager manager;
	QNetworkReply *reply = manager.post(request, multiPart);
	multiPart->setParent(reply);
	qDebug() << "Transmitting" << zipfileToUpload->size() << "bytes file.";

	QObject::connect(reply, SIGNAL(finished()), this, SLOT(finished()));
	QObject::connect(reply, SIGNAL(uploadProgress(qint64,qint64)),this,SLOT(uploadProgress(qint64,qint64)));
	uploadCompleteFlag=false;
	while(!uploadCompleteFlag) {
		QApplication::processEvents();
	}
	qDebug("got it!");
	// test for network error
	QNetworkReply::NetworkError err = reply->error();
	if (err != QNetworkReply::NoError) {
		qDebug() << reply->errorString();
		return false;
	}

	// get the api answer
	QByteArray result = reply->readAll();
	QStringList sp = QString(result).split("\"");
	std::cerr << "result:\n";
	for (int i = 0; i < sp.size(); ++i)
		std::cerr << std::to_string(i) + ": " << sp[i].toStdString() << "\n";
	
	
	//QJSValue sc;
	//QJSEngine engine;
	//qDebug() << "Result:" << result;
	//sc = engine.evaluate("(" + QString(result) + ")");
	QString uid = sp[3];
	if(uid.isEmpty())
		return false;
	qDebug() << "Model uploaded with id" << uid;
	urlModel = "https://sketchfab.com/models/"+uid.toStdString();
	return true;
}

QHttpPart FilterSketchFabPlugin::part_parameter(QString key, QString value) {
	QHttpPart part;
	part.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"" + key + "\""));
	part.setBody(value.toLatin1());
	return part;
}

int FilterSketchFabPlugin::saveMeshZip(
		const std::string& fileName,
		const std::string& internalName,
		const std::string& zipName)
{
	qDebug("Trying to add %s to %s", fileName.c_str(), zipName.c_str());
	mz_zip_archive zip_archive;
	memset(&zip_archive, 0, sizeof (zip_archive)); //Saving memory for the zip archive
	if (!mz_zip_writer_init_file(&zip_archive, zipName.c_str(), 65537)) {
		qDebug("Failed creating zip archive");
		mz_zip_writer_end(&zip_archive);
		return 0;
	}

	const char *pTestComment = "test comment";

	//MZ_BEST_COMPRESSION = 9
	if (!mz_zip_writer_add_file(&zip_archive, internalName.c_str(), fileName.c_str(), pTestComment, strlen(pTestComment), MZ_UBER_COMPRESSION)) {
		qDebug("failed adding %s to %s", fileName.c_str(), zipName.c_str());
		mz_zip_writer_end(&zip_archive);
		return 0;
	}

	mz_zip_writer_finalize_archive(&zip_archive);
	qDebug("Compressed %llu",zip_archive.m_archive_size);
	return 1;
}

void FilterSketchFabPlugin::uploadProgress(qint64 bytesSent, qint64 bytesTotal)
{
	qDebug("Upload progress %i on %i",int(bytesSent),int(bytesTotal));
	char buf[1024]; sprintf(buf,"Upload progress %i on %i",int(bytesSent),int(bytesTotal));
	if(bytesTotal) this->fcb(100*int(bytesSent)/int(bytesTotal),buf);
}

void FilterSketchFabPlugin::finished()
{
	qDebug("FilterSketchFabPlugin::finished()");

	uploadCompleteFlag = true;
}

MESHLAB_PLUGIN_NAME_EXPORTER(FilterSketchFabPlugin)

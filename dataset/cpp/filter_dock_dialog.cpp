/****************************************************************************
 * VCGLib                                                            o o     *
 * Visual and Computer Graphics Library                            o     o   *
 *                                                                _   O  _   *
 * Copyright(C) 2004-2021                                           \/)\/    *
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
#include "filter_dock_dialog.h"
#include "ui_filter_dock_dialog.h"

#include "../mainwindow.h"

FilterDockDialog::FilterDockDialog(
	const RichParameterList& rpl,
	FilterPlugin*            plugin,
	const QAction*           filter,
	QWidget*                 parent,
	GLArea*                  glArea) :
	QDockWidget(parent),
	ui(new Ui::FilterDockDialog),
	plugin(plugin),
	filter(filter),
	parameters(rpl),
	mask(plugin->postCondition(filter)),
	currentGLArea(glArea),
	isPreviewMeshStateValid(false),
	prevParams(rpl),
	mw(nullptr),
	md(nullptr),
	mesh(nullptr)
{
	ui->setupUi(this);

	this->setWindowTitle(plugin->filterName(filter));
	ui->filterInfoLabel->setText(plugin->filterInfo(filter));

	ui->parameterFrame->initParams(rpl, rpl, (QWidget*) glArea);

	// by default, the previewCheckBox is visible when the dialog is constructed.
	// now, we check if the filter is previewable:
	// - if it is previewable, we set all data structures necessary to make the preview available
	// - if it is not previewable, we set the previewCheckBox non visible
	if (!isFilterPreviewable(plugin, filter)) {
		ui->previewCheckBox->setVisible(false);
	}
	else {
		// parent should always be the mainwindow
		// if not, the filter won't be previewable
		mw = static_cast<MainWindow*>(parent);
		if (mw) {
			md   = mw->meshDoc();
			mesh = mw->meshDoc()->mm();
			if (md == nullptr || mesh == nullptr) {
				ui->previewCheckBox->setVisible(false);
				mw   = nullptr;
				md   = nullptr;
				mesh = nullptr;
			}
			else {
				noPreviewMeshState.create(mask, mesh);
				connect(ui->parameterFrame, SIGNAL(parameterChanged()), this, SLOT(applyDynamic()));
				connect(md, SIGNAL(currentMeshChanged(int)), this, SLOT(changeCurrentMesh(int)));
			}
		}
		else {
			ui->previewCheckBox->setVisible(false);
		}
	}
	ui->applyPushButton->setFocus();
	ui->applyPushButton->setDefault(true);

	setTabOrder(ui->defaultPushButton, ui->helpPushButton);
	setTabOrder(ui->helpPushButton, ui->closePushButton);
	setTabOrder(ui->closePushButton, ui->applyPushButton);

	//horrible hack to make the window of almost the right size...
	adjustSize();
	ui->parameterFrame->adjustSize();
	updateGeometry();
	resize(
		width(),
		height() + ui->parameterFrame->height() - 60);

	// set the position of the dock widget starting from the position of the parent.
	if (parent) {
		auto parentCenter = parent->geometry().center();
		auto sizes = geometry().size();

		setGeometry(
			parentCenter.x() - sizes.width() / 2,
			parentCenter.y() - sizes.height() / 2,
			sizes.width(),
			sizes.height());
	}
}

FilterDockDialog::~FilterDockDialog()
{
	delete ui;
}

void FilterDockDialog::on_previewCheckBox_stateChanged(int state)
{
	if (state == Qt::Checked) { // enable preview
		ui->parameterFrame->writeValuesOnParameterList(parameters);

		// if the preview mesh state is valid and parameters are not changed, we do not need to
		// apply again the filter
		if (isPreviewMeshStateValid && parameters == prevParams) {
			previewMeshState.apply(mesh); // apply the preview state to the mesh
			updateRenderingData(mw, mesh);
			if (currentGLArea != nullptr)
				currentGLArea->updateAllDecorators();
		}
		else { // we need to apply the filter
			applyDynamic();
			if (currentGLArea != nullptr)
				currentGLArea->updateAllDecorators();
		}
	}
	// not checked - disable preview
	else {
		noPreviewMeshState.apply(mesh); // re-apply old state of the mesh
		updateRenderingData(mw, mesh);
		if (currentGLArea != nullptr)
			currentGLArea->updateAllDecorators();
	}
}

void FilterDockDialog::on_applyPushButton_clicked()
{
	ui->parameterFrame->writeValuesOnParameterList(parameters);

	if (isPreviewable()) {
		if (mesh) {
			// first, restore the mesh to the no-preview state
			noPreviewMeshState.apply(mesh);
			updateRenderingData(mw, mesh);
		}
	}

	if (isPreviewable() && isPreviewMeshStateValid && parameters == prevParams) {
		previewMeshState.apply(mesh);
		updateRenderingData(mw, mesh);
	}
	else
		emit applyButtonClicked(filter, parameters, false, true);

	if (isPreviewable()) {
		// save the no-preview state, after the filter was applied
		noPreviewMeshState.create(mask, mesh);
	}

	if (currentGLArea)
		currentGLArea->update();
}

void FilterDockDialog::on_helpPushButton_clicked()
{
	ui->parameterFrame->toggleHelp();
	ui->parameterFrame->updateGeometry();
	ui->parameterFrame->adjustSize();
	updateGeometry();
}

void FilterDockDialog::on_closePushButton_clicked()
{
	close();
}

void FilterDockDialog::on_defaultPushButton_clicked()
{
	ui->parameterFrame->resetValues();
}

void FilterDockDialog::applyDynamic()
{
	if (ui->previewCheckBox->isChecked()) {
		prevParams = parameters;
		ui->parameterFrame->writeValuesOnParameterList(parameters);
		ui->parameterFrame->writeValuesOnParameterList(prevParams);

		// first, restore the mesh to the no-preview state
		noPreviewMeshState.apply(mesh);
		// then, apply dynamically with the new parameters
		mw->executeFilter(filter, parameters, true);
		// save the preview state
		previewMeshState.create(mask, mesh);
		isPreviewMeshStateValid = true;

		if (currentGLArea)
			currentGLArea->update();
	}
}

void FilterDockDialog::changeCurrentMesh(int meshId)
{
	if (isPreviewable()) {
		noPreviewMeshState.apply(mesh);
		mesh = md->getMesh(meshId);
		noPreviewMeshState.create(mask, mesh);
		applyDynamic();
	}
}

bool FilterDockDialog::isPreviewable() const
{
	// the actual check whether the filter is previewable or not is made in the constructor, calling
	// the function isFilterPreviewable().
	// when a filter is previewable, the previewCheckBox is visible.
	return ui->previewCheckBox->isVisible();
}

bool FilterDockDialog::isFilterPreviewable(FilterPlugin* plugin, const QAction* filter)
{
	unsigned int mask = plugin->postCondition(filter);
	if ((filter == nullptr) || (plugin == nullptr) ||
		(plugin->filterArity(filter) != FilterPlugin::SINGLE_MESH))
		return false;

	if ((mask == MeshModel::MM_UNKNOWN) || (mask == MeshModel::MM_NONE))
		return false;

	if ((mask & MeshModel::MM_VERTNUMBER) || (mask & MeshModel::MM_FACENUMBER))
		return false;

	return true;
}

void FilterDockDialog::updateRenderingData(MainWindow* mw, MeshModel* mesh)
{
	if (mw != nullptr && mesh != nullptr) {
		MultiViewer_Container* mvcont = mw->currentViewContainer();
		if (mvcont != nullptr) {
			MLSceneGLSharedDataContext* shar = mvcont->sharedDataContext();
			if (shar != nullptr) {
				shar->meshAttributesUpdated(mesh->id(), true, MLRenderingData::RendAtts(true));
				shar->manageBuffers(mesh->id());
			}
		}
	}
}

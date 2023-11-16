/*
 * Copyright 2013, Stephan Aßmus <superstippi@gmx.de>.
 * Copyright 2013, Rene Gollent, <rene@gollent.com>
 * Copyright 2020-2022, Andrew Lindesay <apl@lindesay.co.nz>
 *
 * All rights reserved. Distributed under the terms of the MIT License.
 */


#include "AbstractPackageProcess.h"

#include "Model.h"
#include "PackageManager.h"
#include "PackageUtils.h"


using namespace BPackageKit;

// #pragma mark - PackageAction


AbstractPackageProcess::AbstractPackageProcess(
		PackageInfoRef package, Model* model)
	:
	fPackage(package),
	fModel(model)
{
	if (package.IsSet())
		fInstallLocation = PackageUtils::DeriveInstallLocation(package.Get());
	else
		fInstallLocation = B_PACKAGE_INSTALLATION_LOCATION_SYSTEM;

	// TODO: ideally if the package is installed at multiple locations,
	// the user should be able to pick which one to remove.
	// TODO: allow configuring the installation location
	fPackageManager = new(std::nothrow) PackageManager(
		(BPackageInstallationLocation)fInstallLocation);
}


AbstractPackageProcess::~AbstractPackageProcess()
{
	delete fPackageManager;
}


PackageInfoRef
AbstractPackageProcess::FindPackageByName(const BString& name)
{
	return fModel->PackageForName(name);
}



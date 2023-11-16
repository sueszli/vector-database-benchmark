/*
 * Copyright 2011, Ingo Weinhold, ingo_weinhold@gmx.de.
 * Distributed under the terms of the MIT License.
 */


#include "Resolvable.h"

#include <string.h>

#include "Version.h"


Resolvable::Resolvable(::Package* package)
	:
	fPackage(package),
	fFamily(NULL),
	fName(),
	fVersion(NULL),
	fCompatibleVersion(NULL)
{
}


Resolvable::~Resolvable()
{
	delete fVersion;
	delete fCompatibleVersion;
}


status_t
Resolvable::Init(const char* name, ::Version* version,
	::Version* compatibleVersion)
{
	fVersion = version;
	fCompatibleVersion = compatibleVersion;

	if (!fName.SetTo(name))
		return B_NO_MEMORY;

	return B_OK;
}


void
Resolvable::AddDependency(Dependency* dependency)
{
	fDependencies.Add(dependency);
	dependency->SetResolvable(this);
}


void
Resolvable::RemoveDependency(Dependency* dependency)
{
	fDependencies.Remove(dependency);
	dependency->SetResolvable(NULL);
}


void
Resolvable::MoveDependencies(ResolvableDependencyList& dependencies)
{
	if (fDependencies.IsEmpty())
		return;

	for (ResolvableDependencyList::Iterator it = fDependencies.GetIterator();
			Dependency* dependency = it.Next();) {
		dependency->SetResolvable(NULL);
	}

	dependencies.MoveFrom(&fDependencies);
}

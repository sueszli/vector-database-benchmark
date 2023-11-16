#include <algorithm>
#include <eepp/physics/body.hpp>
#include <eepp/physics/constraints/constraint.hpp>
#include <eepp/physics/physicsmanager.hpp>
#include <eepp/physics/shape.hpp>
#include <eepp/physics/space.hpp>

namespace EE { namespace Physics {

SINGLETON_DECLARE_IMPLEMENTATION( PhysicsManager )

PhysicsManager::PhysicsManager() : mMemoryManager( false ) {}

PhysicsManager::~PhysicsManager() {
	if ( mMemoryManager ) {
		mMemoryManager = false;

		std::vector<Space*>::iterator its = mSpaces.begin();
		for ( ; its != mSpaces.end(); ++its )
			eeSAFE_DELETE( *its );

		std::vector<Body*>::iterator itb = mBodysFree.begin();
		for ( ; itb != mBodysFree.end(); ++itb )
			eeSAFE_DELETE( *itb );

		std::vector<Shape*>::iterator itp = mShapesFree.begin();
		for ( ; itp != mShapesFree.end(); ++itp )
			eeSAFE_DELETE( *itp );

		std::vector<Constraint*>::iterator itc = mConstraintFree.begin();
		for ( ; itc != mConstraintFree.end(); ++itc )
			eeSAFE_DELETE( *itc );
	}
}

PhysicsManager::DrawSpaceOptions* PhysicsManager::getDrawOptions() {
	return &mOptions;
}

void PhysicsManager::setMemoryManager( bool MemoryManager ) {
	mMemoryManager = MemoryManager;
}

const bool& PhysicsManager::isMemoryManagerEnabled() const {
	return mMemoryManager;
}

void PhysicsManager::addBodyFree( Body* body ) {
	if ( mMemoryManager ) {
		if ( std::find( mBodysFree.begin(), mBodysFree.end(), body ) == mBodysFree.end() )
			mBodysFree.push_back( body );
	}
}

void PhysicsManager::removeBodyFree( Body* body ) {
	if ( mMemoryManager ) {
		auto foundIt = std::find( mBodysFree.begin(), mBodysFree.end(), body );
		if ( foundIt != mBodysFree.end() )
			mBodysFree.erase( foundIt );
	}
}

void PhysicsManager::addShapeFree( Shape* shape ) {
	if ( mMemoryManager ) {
		if ( std::find( mShapesFree.begin(), mShapesFree.end(), shape ) == mShapesFree.end() )
			mShapesFree.push_back( shape );
	}
}

void PhysicsManager::removeShapeFree( Shape* shape ) {
	if ( mMemoryManager ) {
		auto foundIt = std::find( mShapesFree.begin(), mShapesFree.end(), shape );
		if ( foundIt != mShapesFree.end() )
			mShapesFree.erase( foundIt );
	}
}

void PhysicsManager::addConstraintFree( Constraint* constraint ) {
	if ( mMemoryManager ) {
		if ( std::find( mConstraintFree.begin(), mConstraintFree.end(), constraint ) ==
			 mConstraintFree.end() )
			mConstraintFree.push_back( constraint );
	}
}

void PhysicsManager::removeConstraintFree( Constraint* constraint ) {
	if ( mMemoryManager ) {
		auto foundIt = std::find( mConstraintFree.begin(), mConstraintFree.end(), constraint );
		if ( foundIt != mConstraintFree.end() )
			mConstraintFree.erase( foundIt );
	}
}

void PhysicsManager::addSpace( Space* space ) {
	if ( mMemoryManager ) {
		if ( std::find( mSpaces.begin(), mSpaces.end(), space ) == mSpaces.end() )
			mSpaces.push_back( space );
	}
}

void PhysicsManager::removeSpace( Space* space ) {
	if ( mMemoryManager ) {
		auto foundIt = std::find( mSpaces.begin(), mSpaces.end(), space );
		if ( foundIt != mSpaces.end() )
			mSpaces.erase( foundIt );
	}
}

}} // namespace EE::Physics

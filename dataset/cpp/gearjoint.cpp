#include <eepp/physics/constraints/gearjoint.hpp>

namespace EE { namespace Physics {

GearJoint::GearJoint( Body* a, Body* b, cpFloat phase, cpFloat ratio ) {
	mConstraint = cpGearJointNew( a->getBody(), b->getBody(), phase, ratio );
	setData();
}

cpFloat GearJoint::getPhase() {
	return cpGearJointGetPhase( mConstraint );
}

void GearJoint::setPhase( const cpFloat& phase ) {
	cpGearJointSetPhase( mConstraint, phase );
}

cpFloat GearJoint::getRatio() {
	return cpGearJointGetRatio( mConstraint );
}

void GearJoint::setRatio( const cpFloat& ratio ) {
	cpGearJointSetRatio( mConstraint, ratio );
}

void GearJoint::draw() {
	// Not implemented
}

}} // namespace EE::Physics

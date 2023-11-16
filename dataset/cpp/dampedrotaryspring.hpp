#ifndef EE_PHYSICS_CDAMPEDROTARYSPRING_HPP
#define EE_PHYSICS_CDAMPEDROTARYSPRING_HPP

#include <eepp/physics/constraints/constraint.hpp>

namespace EE { namespace Physics {

class EE_PHYSICS_API DampedRotarySpring : public Constraint {
  public:
	DampedRotarySpring( Body* a, Body* b, cpFloat restAngle, cpFloat stiffness, cpFloat damping );

	cpFloat getRestAngle();

	void setRestAngle( const cpFloat& restangle );

	cpFloat getStiffness();

	void setStiffness( const cpFloat& stiffness );

	cpFloat getDamping();

	void setDamping( const cpFloat& damping );

	virtual void draw();
};

}} // namespace EE::Physics

#endif

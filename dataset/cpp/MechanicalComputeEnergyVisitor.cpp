/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/mechanicalvisitor/MechanicalComputeEnergyVisitor.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/BaseMass.h>

namespace sofa::simulation::mechanicalvisitor
{


MechanicalComputeEnergyVisitor::MechanicalComputeEnergyVisitor(const sofa::core::MechanicalParams* mparams)
    : sofa::simulation::MechanicalVisitor(mparams)
    , m_kineticEnergy(0.)
    , m_potentialEnergy(0.)
{
}


MechanicalComputeEnergyVisitor::~MechanicalComputeEnergyVisitor()
{
}

SReal MechanicalComputeEnergyVisitor::getKineticEnergy()
{
    return m_kineticEnergy;
}

SReal MechanicalComputeEnergyVisitor::getPotentialEnergy()
{
    return m_potentialEnergy;
}


/// Process the BaseMass
Visitor::Result MechanicalComputeEnergyVisitor::fwdMass(simulation::Node* /*node*/, sofa::core::behavior::BaseMass* mass)
{
    m_kineticEnergy += (SReal)mass->getKineticEnergy();
    return RESULT_CONTINUE;
}
/// Process the BaseForceField
Visitor::Result MechanicalComputeEnergyVisitor::fwdForceField(simulation::Node* /*node*/, sofa::core::behavior::BaseForceField* f)
{
    m_potentialEnergy += (SReal)f->getPotentialEnergy();
    return RESULT_CONTINUE;
}

void MechanicalComputeEnergyVisitor::execute(sofa::core::objectmodel::BaseContext *c, bool precomputedTraversalOrder)
{
    m_kineticEnergy = m_potentialEnergy = 0;
    sofa::simulation::MechanicalVisitor::execute( c, precomputedTraversalOrder );
}

}

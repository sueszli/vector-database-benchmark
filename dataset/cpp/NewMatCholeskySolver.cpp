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

#include <sofa/component/linearsolver/direct/CholeskySolver.inl>
#include <SofaNewmat/NewMatMatrix.h>

#include <sofa/core/ObjectFactory.h>


namespace sofa::component::linearsolver
{


int NewMatCholeskySolverClass = core::RegisterObject("NewMat direct linear solver based on Cholesky factorization, for dense matrices")
        .add<direct::CholeskySolver< NewMatSymmetricMatrix, NewMatVector > >()
        ;

} // namespace sofa::component::linearsolver

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
#include <sofa/component/collision/response/contact/RayContact.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/RayModel.h>
#include <sofa/component/collision/geometry/TetrahedronModel.h>
#include <sofa/helper/Factory.inl>

namespace sofa::component::collision::response::contact
{

using namespace sofa::defaulttype;

Creator<core::collision::Contact::Factory, RayContact<collision::geometry::TetrahedronCollisionModel> > RayTetrahedronContactClass("RayContact",true);

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API RayContact<collision::geometry::TetrahedronCollisionModel>;

}  // namespace sofa::component::collision::response::contact

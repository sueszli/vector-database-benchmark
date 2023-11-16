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
#include <sofa/gl/component/shader/OglAttribute.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gl::component::shader
{

using namespace sofa::defaulttype;


int OglFloatAttributeClass = core::RegisterObject ( "OglFloatAttribute" ).add< OglFloatAttribute >();
int OglFloat2AttributeClass = core::RegisterObject ( "OglFloat2Attribute" ).add< OglFloat2Attribute >();
int OglFloat3AttributeClass = core::RegisterObject ( "OglFloat3Attribute" ).add< OglFloat3Attribute >();
int OglFloat4AttributeClass = core::RegisterObject ( "OglFloat4Attribute" ).add< OglFloat4Attribute >();

int OglIntAttributeClass = core::RegisterObject ( "OglIntAttribute" ).add< OglIntAttribute >();
int OglInt2AttributeClass = core::RegisterObject ( "OglInt2Attribute" ).add< OglInt2Attribute >();
int OglInt3AttributeClass = core::RegisterObject ( "OglInt3Attribute" ).add< OglInt3Attribute >();
int OglInt4AttributeClass = core::RegisterObject ( "OglInt4Attribute" ).add< OglInt4Attribute >();

int OglUIntAttributeClass = core::RegisterObject ( "OglUIntAttribute" ).add< OglUIntAttribute >();
int OglUInt2AttributeClass = core::RegisterObject ( "OglUInt2Attribute" ).add< OglUInt2Attribute >();
int OglUInt3AttributeClass = core::RegisterObject ( "OglUInt3Attribute" ).add< OglUInt3Attribute >();
int OglUInt4AttributeClass = core::RegisterObject ( "OglUInt4Attribute" ).add< OglUInt4Attribute >();

} // namespace sofa::gl::component::shader

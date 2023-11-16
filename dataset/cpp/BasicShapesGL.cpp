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
#define SOFA_HELPER_GL_BASICSHAPESGL_CPP

#include <sofa/gl/BasicShapesGL.inl>

namespace sofa::gl
{

template class SOFA_GL_API BasicShapesGL_Sphere<type::fixed_array< float, 3 > >;
template class SOFA_GL_API BasicShapesGL_Sphere<type::fixed_array< double, 3 > >;
template class SOFA_GL_API BasicShapesGL_FakeSphere<type::fixed_array< float, 3 > >;
template class SOFA_GL_API BasicShapesGL_FakeSphere<type::fixed_array< double, 3 > >;

} // namespace sofa::gl

/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "TableDataWidget.h"
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/Factory.inl>
#include <iostream>

namespace sofa::gui::qt
{

using sofa::helper::Creator;
using sofa::type::fixed_array; 
using namespace sofa::type;
using namespace sofa::defaulttype;

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<int>, TABLE_HORIZONTAL > > DWClass_vectori("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<unsigned int>, TABLE_HORIZONTAL > > DWClass_vectoru("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<float>, TABLE_HORIZONTAL > > DWClass_vectorf("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<double>, TABLE_HORIZONTAL > > DWClass_vectord("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<std::string> > > DWClass_vectorstring("default",true);

#ifdef TODOTOPO
Creator<DataWidgetFactory, TableDataWidget< sofa::component::topology::PointSubset, TABLE_HORIZONTAL > > DWClass_PointSubset("default",true);
#endif

Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqEdges      > > DWClass_SeqEdges     ("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqTriangles  > > DWClass_SeqTriangles ("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqQuads      > > DWClass_SeqQuads     ("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqTetrahedra > > DWClass_SeqTetrahedra("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::core::topology::BaseMeshTopology::SeqHexahedra  > > DWClass_SeqHexahedra ("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<1,int> > > > DWClass_vectorVec1i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<1,unsigned int> > > > DWClass_vectorVec1u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<1,float> > > > DWClass_vectorVec1f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<1,double> > > > DWClass_vectorVec1d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<2,int> > > > DWClass_vectorVec2i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<2,unsigned int> > > > DWClass_vectorVec2u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<2,float> > > > DWClass_vectorVec2f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<2,double> > > > DWClass_vectorVec2d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<3,int> > > > DWClass_vectorVec3i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<3,unsigned int> > > > DWClass_vectorVec3u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<3,float> > > > DWClass_vectorVec3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<3,double> > > > DWClass_vectorVec3d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<4,int> > > > DWClass_vectorVec4i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<4,unsigned int> > > > DWClass_vectorVec4u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<4,float> > > > DWClass_vectorVec4f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<4,double> > > > DWClass_vectorVec4d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<6,int> > > > DWClass_vectorVec6i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<6,unsigned int> > > > DWClass_vectorVec6u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<6,float> > > > DWClass_vectorVec6f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<6,double> > > > DWClass_vectorVec6d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<8,int> > > > DWClass_vectorVec8i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<Vec<8,unsigned int> > > > DWClass_vectorVec8u("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<int,1> > > > DWClass_vectorA1i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<unsigned int,1> > > > DWClass_vectorA1u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<int,2> > > > DWClass_vectorA2i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<unsigned int,2> > > > DWClass_vectorA2u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<int,3> > > > DWClass_vectorA3i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<unsigned int,3> > > > DWClass_vectorA3u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<float,3> > > > DWClass_vectorA3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<double,3> > > > DWClass_vectorA3d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<int,4> > > > DWClass_vectorA4i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<unsigned int,4> > > > DWClass_vectorA4u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<int,6> > > > DWClass_vectorA6i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<unsigned int,6> > > > DWClass_vectorA6u("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<int,8> > > > DWClass_vectorA8i("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<fixed_array<unsigned int,8> > > > DWClass_vectorA8u("default",true);

#if !defined(_MSC_VER) && !defined(__clang__)
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,1> > > > DWClass_stdvectorA1i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,1> > > > DWClass_stdvectorA1u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,2> > > > DWClass_stdvectorA2i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,2> > > > DWClass_stdvectorA2u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,3> > > > DWClass_stdvectorA3i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,3> > > > DWClass_stdvectorA3u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,4> > > > DWClass_stdvectorA4i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,4> > > > DWClass_stdvectorA4u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,6> > > > DWClass_stdvectorA6i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,6> > > > DWClass_stdvectorA6u("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<int,8> > > > DWClass_stdvectorA8i("default",true);
Creator<DataWidgetFactory, TableDataWidget< std::vector<fixed_array<unsigned int,8> > > > DWClass_stdvectorA8u("default",true);
#endif

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::type::Quat<float> > > > DWClass_vectorQuatf("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::type::Quat<double> > > > DWClass_vectorQuatd("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::helper::Polynomial_LD<double,5> > > > DWClass_vectorPolynomialLD5d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::helper::Polynomial_LD<double,4> > > > DWClass_vectorPolynomialLD4d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::helper::Polynomial_LD<double,3> > > > DWClass_vectorPolynomialLD3d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::helper::Polynomial_LD<double,2> > > > DWClass_vectorPolynomialLD2d("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::helper::Polynomial_LD<float,5> > > > DWClass_vectorPolynomialLD5f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::helper::Polynomial_LD<float,4> > > > DWClass_vectorPolynomialLD4f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::helper::Polynomial_LD<float,3> > > > DWClass_vectorPolynomialLD3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::helper::Polynomial_LD<float,2> > > > DWClass_vectorPolynomialLD2f("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::defaulttype::RigidCoord<2,float> > > > DWClass_vectorRigidCoord2f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::defaulttype::RigidCoord<2,double> > > > DWClass_vectorRigidCoord2d("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::defaulttype::RigidCoord<3,float> > > > DWClass_vectorRigidCoord3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::defaulttype::RigidCoord<3,double> > > > DWClass_vectorRigidCoord3d("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::component::solidmechanics::spring::LinearSpring<float> > > > DWClass_vectorLinearSpringf("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::component::solidmechanics::spring::LinearSpring<double> > > > DWClass_vectorLinearSpringd("default",true);

Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::component::solidmechanics::spring::JointSpring<sofa::defaulttype::Rigid3Types> > > > DWClass_vectorJointSpring3f("default",true);
Creator<DataWidgetFactory, TableDataWidget< sofa::type::vector<sofa::component::solidmechanics::spring::GearSpring<sofa::defaulttype::Rigid3Types> > > > DWClass_vectorGearSpring3f("default",true);

} //namespace sofa::gui::qt

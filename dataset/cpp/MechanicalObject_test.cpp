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
#include <sofa/component/statecontainer/MechanicalObject.h>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

namespace sofa
{

namespace
{

// Test types

template<typename T>
struct StubMechanicalObject : public component::statecontainer::MechanicalObject<T>
{};

template<typename T>
struct MechanicalObject_test :  public BaseTest
{
    typedef typename StubMechanicalObject<T>::DataTypes::Coord  Coord;
    typedef typename StubMechanicalObject<T>::DataTypes::Real   Real;

    StubMechanicalObject<T> mechanicalObject;
};


using namespace sofa::defaulttype;
typedef ::testing::Types<Vec1Types, Vec2Types, Vec3Types> DataTypesList;
TYPED_TEST_SUITE(MechanicalObject_test, DataTypesList);

namespace TestHelpers
{

template<typename T, Size INDEX>
struct CheckPositionImpl;

template<Size N, typename REAL>
struct CheckPositionImpl<type::Vec<N, REAL>, 1>
{
    void operator () (const type::Vec<N, REAL>& vec)
    {
        EXPECT_NEAR(REAL(), vec.x(), std::numeric_limits<REAL>::epsilon());
    }
};

template<Size N, typename REAL>
struct CheckPositionImpl<type::Vec<N, REAL>, 2>
{
    void operator () (const type::Vec<N, REAL>& vec)
    {
        CheckPositionImpl<type::Vec<N, REAL>, 1>()(vec);
        EXPECT_NEAR(REAL(), vec.y(), std::numeric_limits<REAL>::epsilon());
    }
};

template<Size N, typename REAL>
struct CheckPositionImpl<type::Vec<N, REAL>, 3>
{
    void operator () (const type::Vec<N, REAL>& vec)
    {
        CheckPositionImpl<type::Vec<N, REAL>, 2>()(vec);
        EXPECT_NEAR(REAL(), vec.z(), std::numeric_limits<REAL>::epsilon());
    }
};

template<typename DataType>
void CheckPosition(StubMechanicalObject<DataType>& mechanicalObject)
{
    CheckPositionImpl<typename DataType::Coord, DataType::coord_total_size>()(mechanicalObject.readPositions()[0]); // Vec<N, real>, RigidCoord<N, real>
}

} // namespace TestHelpers

// Tests

TYPED_TEST(MechanicalObject_test, checkThatDefaultSizeIsOne)
{
    ASSERT_EQ((size_t)1, this->mechanicalObject.getSize());
}

TYPED_TEST(MechanicalObject_test, checkThatTheSizeOfTheDefaultPositionIsEqualToTheSizeOfTheDataTypeCoord)
{
    ASSERT_EQ((size_t)TypeParam::coord_total_size, this->mechanicalObject.readPositions()[0].size());
}

TYPED_TEST(MechanicalObject_test, checkThatPositionDefaultValueIsAVectorOfValueInitializedReals)
{
    TestHelpers::CheckPosition(this->mechanicalObject);
}

} // namespace

} // namespace sofa

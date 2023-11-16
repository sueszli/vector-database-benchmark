/*
 * EllipseIteratorTest.cpp
 *
 *  Created on: Dec 2, 2015
 *      Author: Péter Fankhauser
 *	 Institute: ETH Zurich, ANYbotics
 */

#include "grid_map_core/iterators/EllipseIterator.hpp"
#include "grid_map_core/GridMap.hpp"

// gtest
#include <gtest/gtest.h>

// Vector
#include <vector>

using grid_map::GridMap;
using grid_map::Length;
using grid_map::Position;
using grid_map::EllipseIterator;

TEST(EllipseIterator, OneCellWideEllipse)
{
  GridMap map( { "types" });
  map.setGeometry(Length(8.0, 5.0), 1.0, Position(0.0, 0.0));

  EllipseIterator iterator(map, Position(0.0, 0.0), Length(8.0, 1.0));

  EXPECT_FALSE(iterator.isPastEnd());
  EXPECT_EQ(0, (*iterator)(0));
  EXPECT_EQ(2, (*iterator)(1));

  ++iterator;
  EXPECT_FALSE(iterator.isPastEnd());
  EXPECT_EQ(1, (*iterator)(0));
  EXPECT_EQ(2, (*iterator)(1));

  ++iterator;
  EXPECT_FALSE(iterator.isPastEnd());
  EXPECT_EQ(2, (*iterator)(0));
  EXPECT_EQ(2, (*iterator)(1));

  ++iterator;
  ++iterator;
  ++iterator;
  ++iterator;
  ++iterator;
  EXPECT_FALSE(iterator.isPastEnd());
  EXPECT_EQ(7, (*iterator)(0));
  EXPECT_EQ(2, (*iterator)(1));

  ++iterator;
  EXPECT_TRUE(iterator.isPastEnd());
}

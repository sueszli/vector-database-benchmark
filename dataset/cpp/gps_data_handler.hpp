// Copyright (c) 2020 Fetullah Atas, Norwegian University of Life Sciences
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef VOX_NAV_OPENVSLAM__GPS_DATA_HANDLER_HPP_
#define VOX_NAV_OPENVSLAM__GPS_DATA_HANDLER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vox_nav_utilities/gps_waypoint_collector.hpp>
#include <yaml-cpp/emitter.h>

#include <iostream>
#include <chrono>
#include <numeric>
#include <memory>
#include <vector>
#include <string>
#include <mutex>

namespace vox_nav_openvslam
{

/**
 * @brief
 *
 * @param file_path
 * @param map_db_path
 * @param sensor_type
 * @param gps_waypoint_collector
 * @return true
 * @return false
 */
  bool writeMapInfotoYAML(
    const std::string & file_path,
    const std::string & map_db_path,
    const std::string & sensor_type,
    const std::shared_ptr<vox_nav_utilities::GPSWaypointCollector> & gps_waypoint_collector);
}  // namespace vox_nav_openvslam

#endif    // vox_nav_OPENVSLAM__GPS_DATA_HANDLER_HPP_

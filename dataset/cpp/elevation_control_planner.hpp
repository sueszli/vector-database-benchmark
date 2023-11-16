// Copyright (c) 2022 Fetullah Atas, Norwegian University of Life Sciences
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

#ifndef VOX_NAV_PLANNING__PLUGINS__ELEVATION_CONTROL_PLANNER_HPP_
#define VOX_NAV_PLANNING__PLUGINS__ELEVATION_CONTROL_PLANNER_HPP_

// Our native planners
#include "vox_nav_planning/native_planners/RRTStarF.hpp"
#include "vox_nav_planning/native_planners/LQRPlanner.hpp"
#include "vox_nav_planning/native_planners/LQRRRTStar.hpp"
#include "vox_nav_planning/native_planners/InformedSGCP.hpp"

#include "vox_nav_planning/planner_core.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "vox_nav_utilities/elevation_state_space.hpp"

#include "ompl/control/SimpleDirectedControlSampler.h"
#include "ompl/control/spaces/RealVectorControlSpace.h"
#include "ompl/control/SimpleSetup.h"
#include "ompl/control/planners/sst/SST.h"
#include "ompl/control/planners/rrt/RRT.h"
#include "ompl/control/planners/est/EST.h"
#include "ompl/control/planners/kpiece/KPIECE1.h"
#include "ompl/control/planners/pdst/PDST.h"
#include "ompl/control/planners/syclop/SyclopRRT.h"
#include "ompl/control/planners/PlannerIncludes.h"

#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <vector>
#include <string>
#include <memory>

namespace vox_nav_planning
{

  class ElevationControlPlanner : public vox_nav_planning::PlannerCore
  {

  public:
/**
 * @brief Construct a new ElevationControlPlanner object
 *
 */
    ElevationControlPlanner();

/**
 * @brief Destroy the ElevationControlPlanner object
 *
 */
    ~ElevationControlPlanner();

    /**
     * @brief
     *
     */
    void initialize(
      rclcpp::Node * parent,
      const std::string & plugin_name) override;

    /**
     * @brief Method create the plan from a starting and ending goal.
     *
     * @param start The starting pose of the robot
     * @param goal  The goal pose of the robot
     * @return std::vector<geometry_msgs::msg::PoseStamped>   The sequence of poses to get from start to goal, if any
     */
    std::vector<geometry_msgs::msg::PoseStamped> createPlan(
      const geometry_msgs::msg::PoseStamped & start,
      const geometry_msgs::msg::PoseStamped & goal) override;

    /**
     * @brief propogates the states to next phase, given the control input
     *
     * @param start
     * @param control
     * @param duration
     * @param result
     */
    void propagate(
      const ompl::control::SpaceInformation * si,
      const ompl::base::State * start,
      const ompl::control::Control * control,
      const double duration,
      ompl::base::State * result);

    /**
    * @brief
    *
    * @param state
    * @return true
    * @return false
    */
    bool isStateValid(const ompl::base::State * state) override;

    /**
     * @brief
     *
     * @param msg
     */
    void nodePosesCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);

    /**
     * @brief Get the Opt Objective object
     *
     * @return ompl::base::OptimizationObjectivePtr
     */
    ompl::base::OptimizationObjectivePtr getOptimizationObjective();

    /**
     * @brief
     *
     * @param si
     * @return ompl::base::ValidStateSamplerPtr
     */
    ompl::base::ValidStateSamplerPtr allocValidStateSampler(
      const ompl::base::SpaceInformation * si);

    /**
   * @brief Get the Overlayed Start and Goal poses, only x and y are provided for goal ,
   * but internally planner finds closest valid node on octomap and reassigns goal to this pose
   *
   * @return std::vector<geometry_msgs::msg::PoseStamped>
   */
    std::vector<geometry_msgs::msg::PoseStamped> getOverlayedStartandGoal() override;

    /**
     * @brief
     *
     */
    void setupMap() override;

    void initializeSelectedControlPlanner(
      ompl::base::PlannerPtr & planner,
      const std::string & selected_planner_name,
      const ompl::control::SpaceInformationPtr & si,
      const rclcpp::Logger & logger)
    {
      if (selected_planner_name == std::string("RRT")) {
        planner = ompl::base::PlannerPtr(new ompl::control::RRT(si));
      } else if (selected_planner_name == std::string("RRTStarF")) {
        planner = ompl::base::PlannerPtr(new ompl::control::RRTStarF(si));
      } else if (selected_planner_name == std::string("LQRPlanner")) {
        planner = ompl::base::PlannerPtr(new ompl::control::LQRPlanner(si));
      } else if (selected_planner_name == std::string("LQRRRTStar")) {
        planner = ompl::base::PlannerPtr(new ompl::control::LQRRRTStar(si));
      } else if (selected_planner_name == std::string("SST")) {
        planner = ompl::base::PlannerPtr(new ompl::control::SST(si));
      } else if (selected_planner_name == std::string("EST")) {
        planner = ompl::base::PlannerPtr(new ompl::control::EST(si));
      } else if (selected_planner_name == std::string("KPIECE1")) {
        planner = ompl::base::PlannerPtr(new ompl::control::KPIECE1(si));
      } else if (selected_planner_name == std::string("InformedSGCP")) {
        planner = ompl::base::PlannerPtr(new ompl::control::InformedSGCP(si));
        planner->as<ompl::control::InformedSGCP>()->setUseValidSampler(false);
        planner->as<ompl::control::InformedSGCP>()->setMaxDistBetweenVertices(0.0);
        planner->as<ompl::control::InformedSGCP>()->setUseKNearest(true);
        planner->as<ompl::control::InformedSGCP>()->setSolveControlGraph(false);
        planner->as<ompl::control::InformedSGCP>()->setBatchSize(1000);
      } else {
        RCLCPP_WARN(
          logger,
          "Selected planner is not Found in available planners, using the default planner: RRTStarF");
        planner = ompl::base::PlannerPtr(new ompl::control::RRTStarF(si));
      }
    }

  protected:
    rclcpp::Logger logger_{rclcpp::get_logger("elevation_control_planner")};
    rclcpp::Client<vox_nav_msgs::srv::GetTraversabilityMap>::SharedPtr get_traversability_map_client_;

    // Surfels centers are elevated by node_elevation_distance_, and are stored in this
    // octomap, this maps is used by planner to sample states that are
    // strictly laying on ground but not touching. So it constrains the path to be on ground
    // while it can elevate thorogh ramps or slopes
    std::shared_ptr<octomap::OcTree> elevated_surfel_octomap_octree_;
    // it is also required to have orientation information of surfels, they are kept in
    // elevated_surfel_poses_msg_
    geometry_msgs::msg::PoseArray::SharedPtr elevated_surfel_poses_msg_;
    pcl::PointCloud<pcl::PointSurfel>::Ptr elevated_surfel_cloud_;
    geometry_msgs::msg::PoseStamped nearest_elevated_surfel_to_start_;
    geometry_msgs::msg::PoseStamped nearest_elevated_surfel_to_goal_;
    std::shared_ptr<fcl::CollisionObjectf> elevated_surfels_collision_object_;
    ompl::base::OptimizationObjectivePtr octocost_optimization_;
    ompl::base::StateSpacePtr state_space_;
    ompl::control::ControlSpacePtr control_state_space_;
    ompl::control::SimpleSetupPtr control_simple_setup_;

    std::shared_ptr<ompl::base::RealVectorBounds> z_bounds_;
    std::shared_ptr<ompl::base::RealVectorBounds> se2_bounds_;

    std::string selected_se2_space_name_;
    ompl::base::ElevationStateSpace::SE2StateType se2_space_type_;
    // curve radius for reeds and dubins only
    double rho_;

    // octomap acquired from original PCD map
    std::shared_ptr<octomap::OcTree> original_octomap_octree_;
    std::shared_ptr<fcl::CollisionObjectf> original_octomap_collision_object_;
    std::shared_ptr<fcl::CollisionObjectf> robot_collision_object_;
    // Better t keep this parameter consistent with map_server, 0.2 is a OK default fo this
    double octomap_voxel_size_;
    // global mutex to guard octomap
    std::mutex octomap_mutex_;
  };
}  // namespace vox_nav_planning

#endif  // VOX_NAV_PLANNING__PLUGINS__ELEVATION_CONTROL_PLANNER_HPP_

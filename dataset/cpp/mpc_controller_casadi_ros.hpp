// Copyright (c) 2021 Fetullah Atas, Norwegian University of Life Sciences
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

#include <rclcpp/rclcpp.hpp>
#include <vox_nav_control/controller_core.hpp>
#include <vox_nav_control/mpc_controller_casadi/mpc_controller_casadi_core.hpp>
#include <vox_nav_utilities/tf_helpers.hpp>
#include <vox_nav_msgs/msg/object_array.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <string>
#include <memory>
#include <vector>

#ifndef VOX_NAV_CONTROL__MPC_CONTROLLER__MPC_CONTROLLER_ROS_HPP_
#define VOX_NAV_CONTROL__MPC_CONTROLLER__MPC_CONTROLLER_ROS_HPP_

namespace vox_nav_control
{

  namespace mpc_controller_casadi
  {
/**
 * @brief A ROS wrapper around the Core MPC controller class.
 *        This class interfaces MPC controller with ControllerCore which is base class for plugins.
 *        Derivatives of ControllerCore class are used to expose Controller server for use
 *        The controller server is an ROS2 action named follow_path
 *
 */
    class MPCControllerCasadiROS : public vox_nav_control::ControllerCore
    {
    public:
      /**
       * @brief Construct a new MPCControllerCasadiROS object
       *
       */
      MPCControllerCasadiROS();

      /**
       * @brief Destroy the MPCControllerCasadiROS object
       *
       */
      ~MPCControllerCasadiROS();

      /**
       * @brief gets all parameters required by MPC controller. Sets up an ROS2 stuff, publisher etc.
       *
       * @param parent
       * @param plugin_name
       */
      void initialize(
        rclcpp::Node * parent,
        const std::string & plugin_name) override;

      /**
       * @brief Set the Plan from controller server
       *
       * @param path
       */
      void setPlan(const nav_msgs::msg::Path & path) override;

      /**
       * @brief Compute required velocity commands to drive the robot along the reference_trajectory_
       *
       * @param curr_robot_pose
       * @return geometry_msgs::msg::Twist
       */
      geometry_msgs::msg::Twist computeVelocityCommands(
        geometry_msgs::msg::PoseStamped curr_robot_pose)
      override;

      /**
       * @brief Compute required velocity commands recorrect heading. This function is used when robot already satisfies
       * goal_tolerance_distance
       *
       * @param curr_robot_pose
       * @return geometry_msgs::msg::Twist
       */
      geometry_msgs::msg::Twist computeHeadingCorrectionCommands(
        geometry_msgs::msg::PoseStamped curr_robot_pose)
      override;

      /**
       * @brief
       *
       * @param msg
       */
      void obstacleTracksCallback(const vox_nav_msgs::msg::ObjectArray::SharedPtr msg);

      /**
       * @brief Convert ROS msg type of ObjectsArray to vector of Ellipsoid to feed to MPC controller
       *
       * @param tracks
       * @return std::vector<Ellipsoid>
       */
      std::vector<vox_nav_control::common::Ellipsoid> trackMsg2Ellipsoids(
        const vox_nav_msgs::msg::ObjectArray & tracks,
        const geometry_msgs::msg::PoseStamped & curr_robot_pose);

    private:
      // Given refernce traj to follow, this is set got from planner
      nav_msgs::msg::Path reference_traj_;
      // computed velocities as result of MPC
      geometry_msgs::msg::Twist computed_velocity_;
      // MPC core controller object
      std::shared_ptr<MPCControllerCasadiCore> mpc_controller_;
      // parameters struct used for MPC controller
      vox_nav_control::common::Parameters mpc_parameters_;
      // Keep a copy of previously applied control inputs, this is neded
      // by MPC algorithm
      vox_nav_control::common::ControlInput previous_control_;
      // Publish local trajecory currently being fed to controller
      rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
        interpolated_local_reference_traj_publisher_;
      rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
        mpc_computed_traj_publisher_;
      // while following global plan whats the max look ahead distance ?, thats global_plan_look_ahead_distance_
      double global_plan_look_ahead_distance_;

      rclcpp::Node * parent_;
      rclcpp::Subscription<vox_nav_msgs::msg::ObjectArray>::SharedPtr obstacle_tracks_sub_;
      vox_nav_msgs::msg::ObjectArray obstacle_tracks_;
      std::mutex obstacle_tracks_mutex_;

      ompl::base::StateSpacePtr state_space_;
      std::string selected_se2_space_name_;
      // curve radius for reeds and dubins only
      double rho_;
      ompl::base::SpaceInformationPtr state_space_information_;

      bool solved_at_least_once_;
      std::mutex global_plan_mutex_;
    };

  } // namespace mpc_controller_casadi
}  // namespace vox_nav_control

#endif  // VOX_NAV_CONTROL__MPC_CONTROLLER__MPC_CONTROLLER_ROS_HPP_

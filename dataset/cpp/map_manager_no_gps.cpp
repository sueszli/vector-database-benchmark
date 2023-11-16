// Copyright (c) 2023 Fetullah Atas, Norwegian University of Life Sciences
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

#include "vox_nav_map_server/map_manager_no_gps.hpp"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

namespace vox_nav_map_server
{
MapManagerNoGPS::MapManagerNoGPS() : Node("vox_nav_map_manager_no_gps_rclcpp_node"), map_configured_(false)
{
  RCLCPP_INFO(this->get_logger(), "Creating..");
  // initialize shared pointers asap
  original_octomap_msg_ = std::make_shared<octomap_msgs::msg::Octomap>();
  collision_octomap_msg_ = std::make_shared<octomap_msgs::msg::Octomap>();
  elevated_surfel_octomap_msg_ = std::make_shared<octomap_msgs::msg::Octomap>();
  elevated_surfel_poses_msg_ = std::make_shared<geometry_msgs::msg::PoseArray>();
  octomap_pointcloud_msg_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
  traversable_pointcloud_msg_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
  non_traversable_pointcloud_msg_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
  elevated_surfels_pointcloud_msg_ = std::make_shared<sensor_msgs::msg::PointCloud2>();
  original_octomap_markers_msg_ = std::make_shared<visualization_msgs::msg::MarkerArray>();
  elevated_surfel_octomap_markers_msg_ = std::make_shared<visualization_msgs::msg::MarkerArray>();
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  static_transform_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

  // Declare this node's parameters
  declare_parameter("pcd_map_filename", "/home/ros2-foxy/f.pcd");
  declare_parameter("octomap_voxel_size", 0.2);
  declare_parameter("octomap_publish_frequency", 10);
  declare_parameter("publish_octomap_visuals", true);
  declare_parameter("non_traversable_pointcloud_publish_topic", "non_traversable_pointcloud");
  declare_parameter("traversable_pointcloud_publish_topic", "traversable_pointcloud");
  declare_parameter("octomap_point_cloud_publish_topic", "octomap_pointcloud");
  declare_parameter("octomap_markers_publish_topic", "octomap_markers");
  declare_parameter("map_frame_id", "map");
  declare_parameter("pcd_map_transform.translation.x", 0.0);
  declare_parameter("pcd_map_transform.translation.y", 0.0);
  declare_parameter("pcd_map_transform.translation.z", 0.0);
  declare_parameter("pcd_map_transform.rotation.r", 0.0);
  declare_parameter("pcd_map_transform.rotation.p", 0.0);
  declare_parameter("pcd_map_transform.rotation.y", 0.0);
  declare_parameter("apply_filters", true);
  declare_parameter("pcd_map_downsample_voxel_size", 0.1);
  declare_parameter("remove_outlier_mean_K", 10);
  declare_parameter("remove_outlier_stddev_threshold", 1.0);
  declare_parameter("remove_outlier_radius_search", 0.1);
  declare_parameter("remove_outlier_min_neighbors_in_radius", 1);
  declare_parameter("uniform_sample_radius", 0.2);
  declare_parameter("surfel_radius", 0.8);
  declare_parameter("max_allowed_tilt", 40.0);
  declare_parameter("max_allowed_point_deviation", 0.2);
  declare_parameter("max_allowed_energy_gap", 0.2);
  declare_parameter("node_elevation_distance", 0.5);
  declare_parameter("plane_fit_threshold", 0.2);
  declare_parameter("robot_mass", 0.1);
  declare_parameter("average_speed", 1.0);
  declare_parameter("cost_critic_weights", std::vector<double>({ 0.8, 0.1, 0.1 }));

  // get this node's parameters
  get_parameter("pcd_map_filename", pcd_map_filename_);
  get_parameter("octomap_voxel_size", octomap_voxel_size_);
  get_parameter("octomap_publish_frequency", octomap_publish_frequency_);
  get_parameter("publish_octomap_visuals", publish_octomap_visuals_);
  get_parameter("octomap_point_cloud_publish_topic", octomap_point_cloud_publish_topic_);
  get_parameter("octomap_markers_publish_topic", octomap_markers_publish_topic_);
  get_parameter("non_traversable_pointcloud_publish_topic", non_traversable_pointcloud_publish_topic_);
  get_parameter("traversable_pointcloud_publish_topic", traversable_pointcloud_publish_topic_);
  get_parameter("map_frame_id", map_frame_id_);
  get_parameter("pcd_map_transform.translation.x", pcd_map_transform_matrix_.translation_.x());
  get_parameter("pcd_map_transform.translation.y", pcd_map_transform_matrix_.translation_.y());
  get_parameter("pcd_map_transform.translation.z", pcd_map_transform_matrix_.translation_.z());
  get_parameter("pcd_map_transform.rotation.r", pcd_map_transform_matrix_.rpyIntrinsic_.x());
  get_parameter("pcd_map_transform.rotation.p", pcd_map_transform_matrix_.rpyIntrinsic_.y());
  get_parameter("pcd_map_transform.rotation.y", pcd_map_transform_matrix_.rpyIntrinsic_.z());
  get_parameter("uniform_sample_radius", cost_params_.uniform_sample_radius);
  get_parameter("surfel_radius", cost_params_.surfel_radius);
  get_parameter("max_allowed_tilt", cost_params_.max_allowed_tilt);
  get_parameter("max_allowed_point_deviation", cost_params_.max_allowed_point_deviation);
  get_parameter("max_allowed_energy_gap", cost_params_.max_allowed_energy_gap);
  get_parameter("node_elevation_distance", cost_params_.node_elevation_distance);
  get_parameter("plane_fit_threshold", cost_params_.plane_fit_threshold);
  get_parameter("robot_mass", cost_params_.robot_mass);
  get_parameter("average_speed", cost_params_.average_speed);
  get_parameter("cost_critic_weights", cost_params_.cost_critic_weights);
  get_parameter("apply_filters", preprocess_params_.apply_filters);
  get_parameter("pcd_map_downsample_voxel_size", preprocess_params_.pcd_map_downsample_voxel_size);
  get_parameter("remove_outlier_mean_K", preprocess_params_.remove_outlier_mean_K);
  get_parameter("remove_outlier_stddev_threshold", preprocess_params_.remove_outlier_stddev_threshold);
  get_parameter("remove_outlier_radius_search", preprocess_params_.remove_outlier_radius_search);
  get_parameter("remove_outlier_min_neighbors_in_radius", preprocess_params_.remove_outlier_min_neighbors_in_radius);

  // service hooks for get maps and surfels
  get_traversability_map_service_ = this->create_service<vox_nav_msgs::srv::GetTraversabilityMap>(
      std::string("get_traversability_map"),
      std::bind(&MapManagerNoGPS::getGetTraversabilityMapCallback, this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000 / octomap_publish_frequency_)),
                                   std::bind(&MapManagerNoGPS::timerCallback, this));

  octomap_pointloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      octomap_point_cloud_publish_topic_, rclcpp::SystemDefaultsQoS());

  elevated_surfel_pcl_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "vox_nav/map_server/elevated_surfel_pointcloud", rclcpp::SystemDefaultsQoS());

  traversable_pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      traversable_pointcloud_publish_topic_, rclcpp::SystemDefaultsQoS());

  non_traversable_pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      non_traversable_pointcloud_publish_topic_, rclcpp::SystemDefaultsQoS());

  octomap_markers_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      octomap_markers_publish_topic_, rclcpp::SystemDefaultsQoS());

  elevated_surfel_octomap_markers_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "vox_nav/map_server/elevated_surfel_markers", rclcpp::SystemDefaultsQoS());

  pcd_map_pointcloud_ = vox_nav_utilities::loadPointcloudFromPcd(pcd_map_filename_.c_str());

  RCLCPP_INFO(this->get_logger(), "Loaded a PCD map with %d points", pcd_map_pointcloud_->points.size());
}

MapManagerNoGPS::~MapManagerNoGPS()
{
  RCLCPP_INFO(this->get_logger(), "Destroying");
}

void MapManagerNoGPS::timerCallback()
{
  // Since this is static map we need to georefence this only once not each time
  std::call_once(configure_map_once_, [this]() {
    RCLCPP_INFO(get_logger(),
                "Configuring pcd map with given parameters,"
                " But the map and octomap will be published at %i frequency rate",
                octomap_publish_frequency_);

    preProcessPCDMap();
    regressCosts();
    handleOriginalOctomap();
    RCLCPP_INFO(get_logger(), "Georeferenced given map, ready to publish");

    map_configured_ = true;
  });
  publishMapVisuals();
}

void MapManagerNoGPS::preProcessPCDMap()
{
  if (preprocess_params_.pcd_map_downsample_voxel_size > 0.0)
  {
    pcd_map_pointcloud_ = vox_nav_utilities::downsampleInputCloud<pcl::PointXYZRGB>(
        pcd_map_pointcloud_, preprocess_params_.pcd_map_downsample_voxel_size);
  }

  RCLCPP_INFO(this->get_logger(),
              "PCD Map downsampled, it now has %d points"
              " adjust the parameters if the map looks off",
              pcd_map_pointcloud_->points.size());
  if (preprocess_params_.apply_filters)
  {
    pcd_map_pointcloud_ = vox_nav_utilities::removeOutliersFromInputCloud(
        pcd_map_pointcloud_, preprocess_params_.remove_outlier_mean_K,
        preprocess_params_.remove_outlier_stddev_threshold,
        vox_nav_utilities::OutlierRemovalType::StatisticalOutlierRemoval);
    pcd_map_pointcloud_ = vox_nav_utilities::removeNans<pcl::PointXYZRGB>(pcd_map_pointcloud_);

    /*pcd_map_pointcloud_ = vox_nav_utilities::removeOutliersFromInputCloud(
      pcd_map_pointcloud_,
      preprocess_params_.remove_outlier_min_neighbors_in_radius,
      preprocess_params_.remove_outlier_radius_search,
      vox_nav_utilities::OutlierRemovalType::RadiusOutlierRemoval);*/

    RCLCPP_INFO(this->get_logger(),
                "Applied a series of noise removal functions"
                " PCD Map downsampled, it now has %d points",
                pcd_map_pointcloud_->points.size());
  }
  // apply a rigid body transfrom if it was given one
  /*pcd_map_pointcloud_ = vox_nav_utilities::transformCloud(
    pcd_map_pointcloud_,
    vox_nav_utilities::getRigidBodyTransform(
      pcd_map_transform_matrix_.translation_,
      pcd_map_transform_matrix_.rpyIntrinsic_,
      get_logger()));*/

  Eigen::Affine3d bt = vox_nav_utilities::getRigidBodyTransform(pcd_map_transform_matrix_.translation_,
                                                                pcd_map_transform_matrix_.rpyIntrinsic_, get_logger());
  auto final_tr = tf2::eigenToTransform(bt);
  pcl_ros::transformPointCloud(*pcd_map_pointcloud_, *pcd_map_pointcloud_, final_tr);

  // Experimental, this assumes we have no prior infromation of
  // segmentation, so mark all points as traversable
  // by painting them green > 0
  pcd_map_pointcloud_ =
      vox_nav_utilities::set_cloud_color(pcd_map_pointcloud_, std::vector<double>({ 0.0, 255.0, 0.0 }));
}

void MapManagerNoGPS::regressCosts()
{
  // seperate traversble points from non-traversable ones
  auto pure_traversable_pcl = vox_nav_utilities::get_traversable_points(pcd_map_pointcloud_);
  auto pure_non_traversable_pcl = vox_nav_utilities::get_non_traversable_points(pcd_map_pointcloud_);

  // uniformly sample nodes on top of traversable cloud
  auto uniformly_sampled_nodes = vox_nav_utilities::uniformlySampleCloud<pcl::PointXYZRGB>(
      pure_traversable_pcl, cost_params_.uniform_sample_radius);

  // This is basically vector of cloud segments, each segments includes points representing a cell
  // The first element of pair is surfel_center_point while the second is pointcloud itself
  auto surfels = vox_nav_utilities::surfelize_traversability_cloud(pure_traversable_pcl, uniformly_sampled_nodes,
                                                                   cost_params_.surfel_radius);

  // this is acquired by merging all surfels
  pcl::PointCloud<pcl::PointXYZRGB> cost_regressed_cloud;
  // this is acquired by merging only elevated surfel cenroids
  pcl::PointCloud<pcl::PointSurfel> elevated_surfels_cloud;

  for (auto&& i : surfels)
  {
    auto surfel_center_point = i.first;
    auto surfel_cloud = i.second;

    // fit a plane to this surfel cloud, in order to et its orientation
    pcl::ModelCoefficients::Ptr plane_model(new pcl::ModelCoefficients);

    try
    {
      vox_nav_utilities::fit_plane_to_cloud(plane_model, surfel_cloud, cost_params_.plane_fit_threshold);
    }
    catch (...)
    {
      RCLCPP_ERROR(get_logger(),
                   "Cannot fit a plane to current surfel points, this may occur if cell size is too small");
      RCLCPP_ERROR(get_logger(), "Current surfel has %d points, Jumping to next surfel", surfel_cloud->points.size());
      continue;
    }

    // extract rpy from plane equation
    auto rpy = vox_nav_utilities::rpy_from_plane(*plane_model);

    // extract averge point deviation from surfel cloud this determines the roughness of cloud
    double average_point_deviation = vox_nav_utilities::average_point_deviation_from_plane(surfel_cloud, *plane_model);

    // extract max energy grap from surfel cloud, the higher this , the higher cost
    double max_energy_gap =
        vox_nav_utilities::max_energy_gap_in_cloud(surfel_cloud, cost_params_.robot_mass, cost_params_.average_speed);

    // regulate all costs to be less than 1.0
    double max_tilt = std::max(std::abs(rpy[0]), std::abs(rpy[1]));
    double slope_cost = std::min(max_tilt / cost_params_.max_allowed_tilt, 1.0) * cost_params_.max_color_range;
    double energy_gap_cost =
        std::min(max_energy_gap / cost_params_.max_allowed_energy_gap, 1.0) * cost_params_.max_color_range;
    double deviation_of_points_cost =
        std::min(average_point_deviation / cost_params_.max_allowed_point_deviation, 1.0) *
        cost_params_.max_color_range;

    double total_cost = cost_params_.cost_critic_weights[0] * slope_cost +
                        cost_params_.cost_critic_weights[1] * deviation_of_points_cost +
                        cost_params_.cost_critic_weights[2] * energy_gap_cost;

    // any roll or pitch thats higher than max_tilt will make that surfel NON traversable
    if (max_tilt > cost_params_.max_allowed_tilt /*||
        max_energy_gap > cost_params_.max_allowed_energy_gap ||
        average_point_deviation > cost_params_.max_allowed_point_deviation ||
        total_cost > 180*/)
    {
      surfel_cloud = vox_nav_utilities::set_cloud_color(surfel_cloud, std::vector<double>({ 255.0, 0, 0 }));
    }
    else
    {
      surfel_cloud = vox_nav_utilities::set_cloud_color(
          surfel_cloud, std::vector<double>({ 0.0, cost_params_.max_color_range - total_cost, total_cost }));

      pcl::PointSurfel elevated_surfel;
      elevated_surfel.x = surfel_center_point.x + cost_params_.node_elevation_distance * plane_model->values[0];
      elevated_surfel.y = surfel_center_point.y + cost_params_.node_elevation_distance * plane_model->values[1];
      elevated_surfel.z = surfel_center_point.z + cost_params_.node_elevation_distance * plane_model->values[2];
      elevated_surfel.r = 0.0;
      elevated_surfel.g = cost_params_.max_color_range - total_cost;
      elevated_surfel.b = total_cost;
      elevated_surfels_cloud.points.push_back(elevated_surfel);

      // inflate surfel as a cylinder by appending surfel cloud and their
      // up and down projections(by using surfel normal)
      for (auto&& sp : surfel_cloud->points)
      {
        double step_size = 0.02;
        for (double step = 0.00; step < 0.00; step += step_size)
        {
          pcl::PointSurfel sp_down, sp_up;
          sp_down.x = sp.x + (cost_params_.node_elevation_distance + step) * plane_model->values[0];
          sp_down.y = sp.y + (cost_params_.node_elevation_distance + step) * plane_model->values[1];
          sp_down.z = sp.z + (cost_params_.node_elevation_distance + step) * plane_model->values[2];
          sp_down.g = cost_params_.max_color_range - total_cost;
          sp_down.b = total_cost;
          sp_up.x = sp.x + (cost_params_.node_elevation_distance - step) * plane_model->values[0];
          sp_up.y = sp.y + (cost_params_.node_elevation_distance - step) * plane_model->values[1];
          sp_up.z = sp.z + (cost_params_.node_elevation_distance - step) * plane_model->values[2];
          sp_up.g = cost_params_.max_color_range - total_cost;
          sp_up.b = total_cost;
          elevated_surfels_cloud.points.push_back(sp_down);
          elevated_surfels_cloud.points.push_back(sp_up);
        }
      }
      geometry_msgs::msg::Pose elevated_node_pose;
      elevated_node_pose.position.x = elevated_surfel.x;
      elevated_node_pose.position.y = elevated_surfel.y;
      elevated_node_pose.position.z = elevated_surfel.z;
      elevated_node_pose.orientation = vox_nav_utilities::getMsgQuaternionfromRPY(rpy[0], rpy[1], rpy[2]);
      elevated_surfel_poses_msg_->poses.push_back(elevated_node_pose);
    }
    cost_regressed_cloud += *surfel_cloud;
  }
  elevated_surfel_pointcloud_ = pcl::make_shared<pcl::PointCloud<pcl::PointSurfel>>(elevated_surfels_cloud);

  // overlapping sufels duplicates some points , get rid of them by downsampling
  if (preprocess_params_.pcd_map_downsample_voxel_size > 0.0)
  {
    elevated_surfel_pointcloud_ = vox_nav_utilities::downsampleInputCloud<pcl::PointSurfel>(
        elevated_surfel_pointcloud_, preprocess_params_.pcd_map_downsample_voxel_size);
  }

  octomap::Pointcloud surfel_octocloud;
  for (auto&& i : elevated_surfel_pointcloud_->points)
  {
    surfel_octocloud.push_back(octomap::point3d(i.x, i.y, i.z));
  }
  auto elevated_surfels_octomap_octree = std::make_shared<octomap::OcTree>(octomap_voxel_size_);
  elevated_surfels_octomap_octree->insertPointCloud(surfel_octocloud, octomap::point3d(0, 0, 0));

  for (auto&& i : elevated_surfel_pointcloud_->points)
  {
    double cost_value = static_cast<double>(i.b / 255.0) - static_cast<double>(i.g / 255.0);
    elevated_surfels_octomap_octree->setNodeValue(i.x, i.y, i.z, std::max(0.0, cost_value));
  }

  auto header = std::make_shared<std_msgs::msg::Header>();
  header->frame_id = map_frame_id_;
  header->stamp = this->now();

  vox_nav_utilities::fillOctomapMarkers(elevated_surfel_octomap_markers_msg_, header, elevated_surfels_octomap_octree);

  try
  {
    octomap_msgs::fullMapToMsg<octomap::OcTree>(*elevated_surfels_octomap_octree, *elevated_surfel_octomap_msg_);
    elevated_surfel_octomap_msg_->binary = false;
    elevated_surfel_octomap_msg_->resolution = octomap_voxel_size_;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Exception while converting binary octomap %s:", e.what());
  }

  cost_regressed_cloud += *pure_non_traversable_pcl;
  *pcd_map_pointcloud_ = cost_regressed_cloud;

  // overlapping sufels duplicates some points , get rid of them by downsampling
  if (preprocess_params_.pcd_map_downsample_voxel_size > 0.0)
  {
    pcd_map_pointcloud_ = vox_nav_utilities::downsampleInputCloud<pcl::PointXYZRGB>(
        pcd_map_pointcloud_, preprocess_params_.pcd_map_downsample_voxel_size);
  }
  pure_traversable_pointcloud_ = vox_nav_utilities::get_traversable_points(pcd_map_pointcloud_);
  pure_non_traversable_pointcloud_ = vox_nav_utilities::get_non_traversable_points(pcd_map_pointcloud_);
}

void MapManagerNoGPS::handleOriginalOctomap()
{
  pcl::toROSMsg(*pcd_map_pointcloud_, *octomap_pointcloud_msg_);
  pcl::toROSMsg(*elevated_surfel_pointcloud_, *elevated_surfels_pointcloud_msg_);
  pcl::toROSMsg(*pure_traversable_pointcloud_, *traversable_pointcloud_msg_);
  pcl::toROSMsg(*pure_non_traversable_pointcloud_, *non_traversable_pointcloud_msg_);

  octomap::Pointcloud octocloud, collision_octocloud;
  for (auto&& i : pcd_map_pointcloud_->points)
  {
    octocloud.push_back(octomap::point3d(i.x, i.y, i.z));
  }
  for (auto&& i : pure_non_traversable_pointcloud_->points)
  {
    collision_octocloud.push_back(octomap::point3d(i.x, i.y, i.z));
  }
  auto original_octomap_octree = std::make_shared<octomap::OcTree>(octomap_voxel_size_);
  auto collision_octomap_octree = std::make_shared<octomap::OcTree>(octomap_voxel_size_);

  original_octomap_octree->insertPointCloud(octocloud, octomap::point3d(0, 0, 0));
  collision_octomap_octree->insertPointCloud(collision_octocloud, octomap::point3d(0, 0, 0));

  for (auto&& i : pcd_map_pointcloud_->points)
  {
    double value = static_cast<double>(i.b / 255.0) - static_cast<double>(i.g / 255.0);
    if (i.r == 255)
    {
      value = 2.0;
    }
    original_octomap_octree->setNodeValue(i.x, i.y, i.z, std::max(0.0, value));
  }

  for (auto&& i : pure_non_traversable_pointcloud_->points)
  {
    double value = 2.0;
    collision_octomap_octree->setNodeValue(i.x, i.y, i.z, std::max(0.0, value));
  }

  auto header = std::make_shared<std_msgs::msg::Header>();
  header->frame_id = map_frame_id_;
  header->stamp = this->now();
  vox_nav_utilities::fillOctomapMarkers(original_octomap_markers_msg_, header, original_octomap_octree);

  try
  {
    octomap_msgs::fullMapToMsg<octomap::OcTree>(*original_octomap_octree, *original_octomap_msg_);
    original_octomap_msg_->binary = false;
    original_octomap_msg_->resolution = octomap_voxel_size_;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Exception while converting original binary octomap %s:", e.what());
  }

  try
  {
    octomap_msgs::fullMapToMsg<octomap::OcTree>(*collision_octomap_octree, *collision_octomap_msg_);
    collision_octomap_msg_->binary = false;
    collision_octomap_msg_->resolution = octomap_voxel_size_;
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Exception while converting collision binary octomap %s:", e.what());
  }
}

void MapManagerNoGPS::publishMapVisuals()
{
  if (publish_octomap_visuals_)
  {
    octomap_pointcloud_msg_->header.frame_id = map_frame_id_;
    octomap_pointcloud_msg_->header.stamp = this->now();
    elevated_surfels_pointcloud_msg_->header.frame_id = map_frame_id_;
    elevated_surfels_pointcloud_msg_->header.stamp = this->now();
    traversable_pointcloud_msg_->header.frame_id = map_frame_id_;
    traversable_pointcloud_msg_->header.stamp = this->now();
    non_traversable_pointcloud_msg_->header.frame_id = map_frame_id_;
    non_traversable_pointcloud_msg_->header.stamp = this->now();

    octomap_pointloud_publisher_->publish(*octomap_pointcloud_msg_);
    octomap_markers_publisher_->publish(*original_octomap_markers_msg_);
    elevated_surfel_octomap_markers_publisher_->publish(*elevated_surfel_octomap_markers_msg_);
    elevated_surfel_pcl_publisher_->publish(*elevated_surfels_pointcloud_msg_);
    traversable_pointcloud_publisher_->publish(*traversable_pointcloud_msg_);
    non_traversable_pointcloud_publisher_->publish(*non_traversable_pointcloud_msg_);
  }
}

void MapManagerNoGPS::getGetTraversabilityMapCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<vox_nav_msgs::srv::GetTraversabilityMap::Request> request,
    std::shared_ptr<vox_nav_msgs::srv::GetTraversabilityMap::Response> response)
{
  if (!map_configured_)
  {
    RCLCPP_INFO(get_logger(), "Map has not been configured yet,  cannot handle GetTraversabilityMap request");
    response->is_valid = false;
    return;
  }
  RCLCPP_INFO(get_logger(), "Map is Cofigured Handling an incoming request");
  response->original_octomap = *original_octomap_msg_;
  response->collision_octomap = *collision_octomap_msg_;
  response->elevated_surfel_octomap = *elevated_surfel_octomap_msg_;
  response->elevated_surfel_poses = *elevated_surfel_poses_msg_;
  response->traversable_elevated_cloud = *elevated_surfels_pointcloud_msg_;
  response->traversable_cloud = *traversable_pointcloud_msg_;
  response->is_valid = true;
}
}  // namespace vox_nav_map_server

/**
 * @brief
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char const* argv[])
{
  rclcpp::init(argc, argv);
  auto map_manager_node = std::make_shared<vox_nav_map_server::MapManagerNoGPS>();
  rclcpp::spin(map_manager_node);
  rclcpp::shutdown();
  return 0;
}

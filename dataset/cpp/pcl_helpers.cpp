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

/*
 * Parts of code has been taken from
 *      Edo Jelavic
 *      Institute: ETH Zurich, Robotic Systems Lab
 */

#include <memory>
#include <string>
#include <vector>
#include "vox_nav_utilities/pcl_helpers.hpp"

namespace vox_nav_utilities
{

  Eigen::Vector3d calculateMeanOfPointPositions(
      pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr inputCloud)
  {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto &point : inputCloud->points)
    {
      mean += Eigen::Vector3d(point.x, point.y, point.z);
    }
    mean /= inputCloud->points.size();

    return mean;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformCloud(
      pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr inputCloud,
      const Eigen::Affine3f &transformMatrix)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*inputCloud, *transformedCloud, transformMatrix);
    return transformedCloud;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPointcloudFromPcd(const std::string &filename)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PCLPointCloud2 cloudBlob;
    pcl::io::loadPCDFile(filename, cloudBlob);
    pcl::fromPCLPointCloud2(cloudBlob, *cloud);
    return cloud;
  }

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr loadPointcloudFromPcd(
      const std::string &filename,
      bool label)
  {
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBL>());
    pcl::PCLPointCloud2 cloudBlob;
    pcl::io::loadPCDFile(filename, cloudBlob);
    pcl::fromPCLPointCloud2(cloudBlob, *cloud);
    return cloud;
  }

  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> extractClusterCloudsFromPointcloud(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud,
      double tolerance,
      int min_cluster_size,
      int max_cluster_size)
  {
    // Create a kd tree to cluster the input point cloud
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(inputCloud);
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> euclideanClusterExtraction;
    euclideanClusterExtraction.setClusterTolerance(tolerance);
    euclideanClusterExtraction.setMinClusterSize(min_cluster_size);
    euclideanClusterExtraction.setMaxClusterSize(max_cluster_size);
    euclideanClusterExtraction.setSearchMethod(tree);
    euclideanClusterExtraction.setInputCloud(inputCloud);
    euclideanClusterExtraction.extract(clusterIndices);

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusterClouds;
    clusterClouds.reserve(clusterIndices.size());

    for (const auto &indicesSet : clusterIndices)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      clusterCloud->points.reserve(indicesSet.indices.size());
      for (auto index : indicesSet.indices)
      {
        clusterCloud->points.push_back(inputCloud->points[index]);
      }
      clusterCloud->is_dense = true;
      clusterClouds.push_back(clusterCloud);
    }

    return clusterClouds;
  }

  Eigen::Matrix3d getRotationMatrix(
      double angle, XYZ axis,
      const rclcpp::Logger &node_logger)
  {
    Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Identity();
    switch (axis)
    {
    case XYZ::X:
    {
      rotationMatrix = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX());
      break;
    }
    case XYZ::Y:
    {
      rotationMatrix = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY());
      break;
    }
    case XYZ::Z:
    {
      rotationMatrix = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ());
      break;
    }
    default:
      RCLCPP_ERROR(node_logger, "Unknown axis while trying to rotate the pointcloud");
    }
    return rotationMatrix;
  }

  Eigen::Affine3d getRigidBodyTransform(
      const Eigen::Vector3d &translation,
      const Eigen::Vector3d &intrinsicRpy,
      const rclcpp::Logger &node_logger)
  {
    Eigen::Affine3d rigidBodyTransform;
    rigidBodyTransform.setIdentity();
    rigidBodyTransform.translation() << translation.x(), translation.y(), translation.z();
    Eigen::Matrix3d rotation(Eigen::Matrix3d::Identity());
    rotation *= getRotationMatrix(intrinsicRpy.x(), XYZ::X, node_logger);
    rotation *= getRotationMatrix(intrinsicRpy.y(), XYZ::Y, node_logger);
    rotation *= getRotationMatrix(intrinsicRpy.z(), XYZ::Z, node_logger);
    rigidBodyTransform.rotate(rotation);

    return rigidBodyTransform;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr removeOutliersFromInputCloud(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud, int int_param, double double_param,
      OutlierRemovalType outlier_removal_type)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    if (outlier_removal_type == OutlierRemovalType::StatisticalOutlierRemoval)
    {
      pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
      sor.setInputCloud(inputCloud);
      sor.setMeanK(int_param);
      sor.setStddevMulThresh(double_param);
      sor.filter(*filteredCloud);
    }
    else
    {
      pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
      outrem.setInputCloud(inputCloud);
      outrem.setMinNeighborsInRadius(int_param);
      outrem.setRadiusSearch(double_param);
      outrem.setKeepOrganized(true);
      outrem.filter(*filteredCloud);
    }
    return filteredCloud;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr removeOutliersFromInputCloud(
      pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, int int_param, double double_param,
      OutlierRemovalType outlier_removal_type)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>());

    if (outlier_removal_type == OutlierRemovalType::StatisticalOutlierRemoval)
    {
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud(inputCloud);
      sor.setMeanK(int_param);
      sor.setStddevMulThresh(double_param);
      sor.filter(*filteredCloud);
    }
    else
    {
      pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
      outrem.setInputCloud(inputCloud);
      outrem.setMinNeighborsInRadius(int_param);
      outrem.setRadiusSearch(double_param);
      outrem.setKeepOrganized(true);
      outrem.filter(*filteredCloud);
    }
    return filteredCloud;
  }

  void fitBoxtoPointCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr input,
      vox_nav_msgs::msg::Object &output)
  {
    // Compute principal directions
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*input, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*input, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*input, *cloudPointsProjected, projectionTransform);
    // Get the minimum and maximum points of the transformed cloud.
    pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());

    // Final transform
    const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);
    const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

    // return the bounding box with ros message
    output.pose.position.x = bboxTransform.x();
    output.pose.position.y = bboxTransform.y();
    output.pose.position.z = bboxTransform.z();
    output.pose.orientation.x = bboxQuaternion.x();
    output.pose.orientation.y = bboxQuaternion.y();
    output.pose.orientation.z = bboxQuaternion.z();
    output.pose.orientation.w = bboxQuaternion.w();
    output.shape.dimensions.push_back(maxPoint.x - minPoint.x);
    output.shape.dimensions.push_back(maxPoint.y - minPoint.y);
    output.shape.dimensions.push_back(maxPoint.z - minPoint.z);
    output.shape.type = shape_msgs::msg::SolidPrimitive::BOX;
  }

  void voxnavObjects2VisionObjects(
      const vox_nav_msgs::msg::ObjectArray &input,
      vision_msgs::msg::Detection3DArray &output)
  {
    // RVIZ visualization of dynamic objects
    output.header = input.header;

    for (const auto &obj : input.objects)
    {
      vision_msgs::msg::Detection3D detection;
      detection.header = input.header;
      detection.bbox.center.position.x = obj.pose.position.x;
      detection.bbox.center.position.y = obj.pose.position.y;
      detection.bbox.center.position.z = obj.pose.position.z;
      detection.bbox.center.orientation.x = obj.pose.orientation.x;
      detection.bbox.center.orientation.y = obj.pose.orientation.y;
      detection.bbox.center.orientation.z = obj.pose.orientation.z;
      detection.bbox.center.orientation.w = obj.pose.orientation.w;
      detection.bbox.size.x = obj.shape.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X];
      detection.bbox.size.y = obj.shape.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y];
      detection.bbox.size.z = obj.shape.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z];
      detection.id = obj.id;

      vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
      hypothesis.hypothesis.class_id = obj.classification_label;
      // make sure the score is not NAN or INF
      if (std::isnan(obj.classification_probability) || std::isinf(obj.classification_probability))
      {
        hypothesis.hypothesis.score = -1.0;
      }
      else
      {
        hypothesis.hypothesis.score = obj.classification_probability;
      }
      hypothesis.pose.pose = obj.pose;
      detection.results.push_back(hypothesis);

      // If any of the dimensions is NAN or INF, skip this detection
      if (std::isnan(detection.bbox.size.x) || std::isinf(detection.bbox.size.x) ||
          std::isnan(detection.bbox.size.y) || std::isinf(detection.bbox.size.y) ||
          std::isnan(detection.bbox.size.z) || std::isinf(detection.bbox.size.z))
      {
        continue;
      }

      output.detections.push_back(detection);
    }
  }

  Eigen::Vector3f getColorByIndexEig(int index)
  {
    Eigen::Vector3f result;
    switch (index)
    {
    case -2: // BLACK:
      result[0] = 0.0;
      result[1] = 0.0;
      result[2] = 0.0;
      break;
    case -1: // BLACK:
      result[0] = 0.0;
      result[1] = 0.0;
      result[2] = 0.0;
      break;
    case 0: // RED:
      result[0] = 0.8;
      result[1] = 0.1;
      result[2] = 0.1;
      break;
    case 1: // GREEN:
      result[0] = 0.1;
      result[1] = 0.8;
      result[2] = 0.1;
      break;
    case 2: // GREY:
      result[0] = 0.9;
      result[1] = 0.9;
      result[2] = 0.9;
      break;
    case 3: // DARK_GREY:
      result[0] = 0.6;
      result[1] = 0.6;
      result[2] = 0.6;
      break;
    case 4: // WHITE:
      result[0] = 1.0;
      result[1] = 1.0;
      result[2] = 1.0;
      break;
    case 5: // ORANGE:
      result[0] = 1.0;
      result[1] = 0.5;
      result[2] = 0.0;
      break;
    case 6: // Maroon:
      result[0] = 0.5;
      result[1] = 0.0;
      result[2] = 0.0;
      break;
    case 7: // Olive:
      result[0] = 0.5;
      result[1] = 0.5;
      result[2] = 0.0;
      break;
    case 8: // Navy:
      result[0] = 0.0;
      result[1] = 0.0;
      result[2] = 0.5;
      break;
    case 9: // BLACK:
      result[0] = 0.0;
      result[1] = 0.0;
      result[2] = 0.0;
      break;
    case 10: // YELLOW:
      result[0] = 1.0;
      result[1] = 1.0;
      result[2] = 0.0;
      break;
    case 11: // BROWN:
      result[0] = 0.597;
      result[1] = 0.296;
      result[2] = 0.0;
      break;
    case 12: // PINK:
      result[0] = 1.0;
      result[1] = 0.4;
      result[2] = 1;
      break;
    case 13: // LIME_GREEN:
      result[0] = 0.6;
      result[1] = 1.0;
      result[2] = 0.2;
      break;
    case 14: // PURPLE:
      result[0] = 0.597;
      result[1] = 0.0;
      result[2] = 0.597;
      break;
    case 15: // CYAN:
      result[0] = 0.0;
      result[1] = 1.0;
      result[2] = 1.0;
      break;
    case 16: // MAGENTA:
      result[0] = 1.0;
      result[1] = 0.0;
      result[2] = 1.0;
    }
    return result;
  }

} // namespace vox_nav_utilities

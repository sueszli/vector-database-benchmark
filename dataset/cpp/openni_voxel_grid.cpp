/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/openni_camera/openni_driver.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <mutex>

// clang-format off
#define FPS_CALC(_WHAT_)                                                               \
  do {                                                                                 \
    static unsigned count = 0;                                                         \
    static double last = pcl::getTime();                                               \
    double now = pcl::getTime();                                                       \
    ++count;                                                                           \
    if (now - last >= 1.0) {                                                           \
      std::cout << "Average framerate(" << _WHAT_ << "): "                             \
                << double(count) / double(now - last) << " Hz" << std::endl;           \
      count = 0;                                                                       \
      last = now;                                                                      \
    }                                                                                  \
  } while (false)
// clang-format on

template <typename PointType>
class OpenNIVoxelGrid {
public:
  using Cloud = pcl::PointCloud<PointType>;
  using CloudPtr = typename Cloud::Ptr;
  using CloudConstPtr = typename Cloud::ConstPtr;

  OpenNIVoxelGrid(const std::string& device_id = "",
                  const std::string& = "z",
                  float = 0,
                  float = 5.0,
                  float leaf_size_x = 0.01,
                  float leaf_size_y = 0.01,
                  float leaf_size_z = 0.01)
  : viewer("PCL OpenNI VoxelGrid Viewer"), device_id_(device_id)
  {
    grid_.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);
    // grid_.setFilterFieldName (field_name);
    // grid_.setFilterLimits (min_v, max_v);
  }

  void
  cloud_cb_(const CloudConstPtr& cloud)
  {
    set(cloud);
  }

  void
  set(const CloudConstPtr& cloud)
  {
    // lock while we set our cloud;
    std::lock_guard<std::mutex> lock(mtx_);
    cloud_ = cloud;
  }

  CloudPtr
  get()
  {
    // lock while we swap our cloud and reset it.
    std::lock_guard<std::mutex> lock(mtx_);
    CloudPtr temp_cloud(new Cloud);

    grid_.setInputCloud(cloud_);
    grid_.filter(*temp_cloud);

    return temp_cloud;
  }

  void
  run()
  {
    pcl::OpenNIGrabber interface(device_id_);

    std::function<void(const CloudConstPtr&)> f = [this](const CloudConstPtr& cloud) {
      cloud_cb_(cloud);
    };
    boost::signals2::connection c = interface.registerCallback(f);

    interface.start();

    while (!viewer.wasStopped()) {
      if (cloud_) {
        FPS_CALC("drawing");
        // the call to get() sets the cloud_ to null;
        viewer.showCloud(get());
      }
    }

    interface.stop();
  }

  pcl::VoxelGrid<PointType> grid_;
  pcl::visualization::CloudViewer viewer;
  std::string device_id_;
  std::mutex mtx_;
  CloudConstPtr cloud_;
};

void
usage(char** argv)
{
  std::cout
      << "usage: " << argv[0] << " [options]\n\n"
      << "where options are:\n"
      << "    -device_id X: specify the device id (default: \"#1\").\n"
      << "    -minmax min-max: set the ApproximateVoxelGrid min-max cutting values "
         "(default: 0-5.0)\n"
      << "    -field X: use field/dimension 'X' to filter data on (default: 'z')\n"
      << "    -leaf x, y, z: set the ApproximateVoxelGrid leaf size (default: "
         "0.01)\n\n";

  openni_wrapper::OpenNIDriver& driver = openni_wrapper::OpenNIDriver::getInstance();
  if (driver.getNumberDevices() > 0) {
    for (unsigned deviceIdx = 0; deviceIdx < driver.getNumberDevices(); ++deviceIdx) {
      // clang-format off
      std::cout << "Device: " << deviceIdx + 1 
                << ", vendor: " << driver.getVendorName (deviceIdx) 
                << ", product: " << driver.getProductName (deviceIdx)
                << ", connected: " << driver.getBus (deviceIdx) << " @ " << driver.getAddress (deviceIdx) << ", serial number: \'" << driver.getSerialNumber (deviceIdx) << "\'" 
                << std::endl;
      // clang-format on
    }

    std::cout << "\ndevice_id may be:" << std::endl
              << "    #1, #2, ... for the first second etc device in the list or"
              << std::endl

              << "    bus@address for the device connected to a specific "
                 "usb-bus/address combination (works only in Linux) or"

              << "    <serial-number> (only in Linux and for devices which provide "
                 "serial numbers)";
  }
  else
    std::cout << "No devices connected." << std::endl;
}

int
main(int argc, char** argv)
{
  /////////////////////////////////////////////////////////////////////
  if (pcl::console::find_argument(argc, argv, "-h") != -1 ||
      pcl::console::find_argument(argc, argv, "--help") != -1) {
    usage(argv);
    return 1;
  }

  std::string device_id = "";
  float min_v = 0.0f, max_v = 5.0f;
  std::string field_name = "z";
  float leaf_x = 0.01f, leaf_y = 0.01f, leaf_z = 0.01f;

  if (pcl::console::parse_argument(argc, argv, "-device_id", device_id) == -1 &&
      argc > 1 && argv[1][0] != '-')
    device_id = argv[1];

  pcl::console::parse_2x_arguments(argc, argv, "-minmax", min_v, max_v);
  pcl::console::parse_argument(argc, argv, "-field", field_name);
  PCL_INFO(
      "Filtering data on %s between %f -> %f.\n", field_name.c_str(), min_v, max_v);

  pcl::console::parse_3x_arguments(argc, argv, "-leaf", leaf_x, leaf_y, leaf_z);
  PCL_INFO("Using %f, %f, %f as a leaf size for VoxelGrid.\n", leaf_x, leaf_y, leaf_z);
  /////////////////////////////////////////////////////////////////////

  pcl::OpenNIGrabber grabber(device_id);
  if (grabber.providesCallback<pcl::OpenNIGrabber::sig_cb_openni_point_cloud_rgba>()) {
    OpenNIVoxelGrid<pcl::PointXYZRGBA> v(
        device_id, field_name, min_v, max_v, leaf_x, leaf_y, leaf_z);
    v.run();
  }
  else {
    OpenNIVoxelGrid<pcl::PointXYZ> v(
        device_id, field_name, min_v, max_v, leaf_x, leaf_y, leaf_z);
    v.run();
  }

  return 0;
}

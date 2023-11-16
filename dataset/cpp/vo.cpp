
#include "my_slam/vo/vo.h"
#include "my_slam/optimization/g2o_ba.h"
#include <numeric>

namespace my_slam
{
namespace vo
{

VisualOdometry::VisualOdometry() : map_(new (Map))
{
    vo_state_ = BLANK;
}

void VisualOdometry::getMappointsInCurrentView_(
    vector<MapPoint::Ptr> &candidate_mappoints_in_map,
    vector<cv::Point2f> &candidate_2d_pts_in_image,
    cv::Mat &corresponding_mappoints_descriptors)
{
    // vector<MapPoint::Ptr> candidate_mappoints_in_map;
    // cv::Mat corresponding_mappoints_descriptors;
    candidate_mappoints_in_map.clear();
    corresponding_mappoints_descriptors.release();
    for (auto &iter_map_point : map_->map_points_)
    {
        MapPoint::Ptr &p_world = iter_map_point.second;

        // -- Check if p in curr frame image
        bool is_p_in_curr_frame = true;
        cv::Point3f p_cam = basics::preTranslatePoint3f(p_world->pos_, curr_->T_w_c_.inv()); // T_c_w * p_w = p_c
        if (p_cam.z < 0)
            is_p_in_curr_frame = false;
        cv::Point2f pixel = geometry::cam2pixel(p_cam, curr_->camera_->K_);
        const bool is_inside_image = pixel.x > 0 && pixel.y > 0 && pixel.x < curr_->rgb_img_.cols && pixel.y < curr_->rgb_img_.rows;
        if (!is_inside_image)
            is_p_in_curr_frame = false;

        // -- If is in current frame,
        //      then add this point to candidate_mappoints_in_map
        if (is_p_in_curr_frame)
        {
            candidate_mappoints_in_map.push_back(p_world);
            candidate_2d_pts_in_image.push_back(pixel);
            corresponding_mappoints_descriptors.push_back(p_world->descriptor_);
            p_world->visible_times_++;
        }
    }
}

// --------------------------------- Initialization ---------------------------------

void VisualOdometry::estimateMotionAnd3DPoints_()
{
    // -- Rename output
    vector<cv::DMatch> &inlier_matches = curr_->inliers_matches_with_ref_;
    vector<cv::Point3f> &pts3d_in_curr = curr_->inliers_pts3d_;
    vector<cv::DMatch> &inliers_matches_for_3d = curr_->inliers_matches_for_3d_;
    cv::Mat &T = curr_->T_w_c_;

    // -- Start: call this big function to compute everything
    // (1) motion from Essential && Homography, (2) inliers indices, (3) triangulated points
    vector<cv::Mat> list_R, list_t, list_normal;
    vector<vector<cv::DMatch>> list_matches; // these are the inliers matches
    vector<vector<cv::Point3f>> sols_pts3d_in_cam1_by_triang;
    bool is_print_res = false, is_frame_cam2_to_cam1 = true;
    bool is_calc_homo = true;
    cv::Mat &K = curr_->camera_->K_;
    int best_sol = geometry::helperEstimatePossibleRelativePosesByEpipolarGeometry(
        /*Input*/
        ref_->keypoints_, curr_->keypoints_, curr_->matches_with_ref_, K,
        /*Output*/
        list_R, list_t, list_matches, list_normal, sols_pts3d_in_cam1_by_triang,
        /*settings*/
        is_print_res, is_calc_homo, is_frame_cam2_to_cam1);

    // -- Only retain the data of the best solution
    const cv::Mat &R_curr_to_prev = list_R[best_sol];
    const cv::Mat &t_curr_to_prev = list_t[best_sol];
    inlier_matches = list_matches[best_sol];
    const vector<cv::Point3f> &pts3d_in_cam1 = sols_pts3d_in_cam1_by_triang[best_sol];
    pts3d_in_curr.clear();
    for (const cv::Point3f &p1 : pts3d_in_cam1)
        pts3d_in_curr.push_back(basics::transCoord(p1, R_curr_to_prev, t_curr_to_prev));

    // -- Output

    // compute camera pose
    T = ref_->T_w_c_ * basics::convertRt2T(R_curr_to_prev, t_curr_to_prev).inv();

    // Get points that are used for triangulating new map points
    retainGoodTriangulationResult_();

    int N = curr_->inliers_pts3d_.size();
    if (N < 20)
    {
        printf("After remove bad triag, only %d pts. It's too few ...\n", N);
        return;
    }

    //Normalize Points Depth to 1, and
    double mean_depth_without_scale = basics::calcMeanDepth(curr_->inliers_pts3d_);
    static const double assumed_mean_pts_depth_during_vo_init =
        basics::Config::get<double>("assumed_mean_pts_depth_during_vo_init");
    double scale = assumed_mean_pts_depth_during_vo_init / mean_depth_without_scale;
    t_curr_to_prev *= scale;
    for (cv::Point3f &p : curr_->inliers_pts3d_)
        basics::scalePointPos(p, scale);
    T = ref_->T_w_c_ * basics::convertRt2T(R_curr_to_prev, t_curr_to_prev).inv(); // update pose
}

bool VisualOdometry::isVoGoodToInit_()
{

    // -- Rename input
    const vector<cv::KeyPoint> &init_kpts = ref_->keypoints_;
    const vector<cv::KeyPoint> &curr_kpts = curr_->keypoints_;
    // const vector<cv::DMatch> &matches = curr_->inliers_matches_with_ref_;
    const vector<cv::DMatch> &matches = curr_->inliers_matches_for_3d_;

    // Params
    static const int min_inlier_matches = basics::Config::get<int>("min_inlier_matches");
    static const double min_pixel_dist = basics::Config::get<double>("min_pixel_dist");
    static const double min_median_triangulation_angle = basics::Config::get<double>("min_median_triangulation_angle");

    // -- Check CRITERIA_0: num inliers should be large
    bool criteria_0 = true;
    if (matches.size() < min_inlier_matches)
    {
        printf("%d inlier points are too few... threshold is %d.\n",
               int(matches.size()), min_inlier_matches);
        criteria_0 = false;
    }

    // -- Check criteria_1
    bool criteria_1 = false; // init vo only when distance between matched keypoints are large
    {
        vector<double> dists_between_kpts;
        double mean_dist = geometry::computeMeanDistBetweenKeypoints(init_kpts, curr_kpts, matches);
        printf("Pixel movement of matched keypoints: %.1f. Threshold is %.1f\n", mean_dist, min_pixel_dist);

        criteria_1 = mean_dist > min_pixel_dist;
    }

    // -- Check criteria_2
    bool criteria_2 = false; // Triangulation angle of each point should be larger than threshold.
    if (curr_->triangulation_angles_of_inliers_.size() > 0)
    {
        vector<double> sort_a = curr_->triangulation_angles_of_inliers_; // a copy of angles
        int N = sort_a.size();                                           // num of 3d points triangulated from inlier points
        sort(sort_a.begin(), sort_a.end());
        double mean_angle = accumulate(sort_a.begin(), sort_a.end(), 0.0) / N;
        double median_angle = sort_a[N / 2];
        printf("Triangulation angle: mean=%f, median=%f, min=%f, max=%f.\n",
               mean_angle,   // mean
               median_angle, // median
               sort_a[0],    // min
               sort_a[N - 1] // max
        );

        // Thresholding
        printf("    median_angle is %.2f, threshold is %.2f.\n",
               median_angle, min_median_triangulation_angle);
        if (median_angle > min_median_triangulation_angle)
            criteria_2 = true;
    }

    // -- Return
    return criteria_0 && criteria_1 && criteria_2;
}

bool VisualOdometry::isInitialized()
{
    return vo_state_ == DOING_TRACKING;
}

// ------------------------------- Triangulation -------------------------------

// Compute the triangulation angle of each point, and get the statistics.
// Remove those with a too large or too small angle.
void VisualOdometry::retainGoodTriangulationResult_()
{
    static const double min_triang_angle = basics::Config::get<double>("min_triang_angle");
    static const double max_ratio_between_max_angle_and_median_angle =
        basics::Config::get<double>("max_ratio_between_max_angle_and_median_angle");

    // -- Input
    // 1. vector<cv::DMatch>  curr_ -> inliers_matches_with_ref_; // input
    // 2. vector<cv::Point3f> pts3d_in_curr:  curr_ -> inliers_pts3d_; // update this

    // -- Output
    // 1. generate this:
    vector<double> &angles = curr_->triangulation_angles_of_inliers_;
    // 2. update this:
    //vector<cv::Point3f> pts3d_in_curr:  curr_ -> inliers_pts3d_;
    // 3. generate this:
    //vector<cv::DMatch> inliers_matches: curr_ -> inliers_matches_for_3d_;

    // -- Compute angles
    int N = (int)curr_->inliers_pts3d_.size();
    if (N == 0)
        return;
    for (int i = 0; i < N; i++)
    {
        cv::Point3f &p_in_curr = curr_->inliers_pts3d_[i];
        cv::Mat p_in_world = basics::point3f_to_mat3x1(basics::preTranslatePoint3f(p_in_curr, curr_->T_w_c_));
        cv::Mat vec_p_to_cam_curr = basics::getPosFromT(curr_->T_w_c_) - p_in_world;
        cv::Mat vec_p_to_cam_prev = basics::getPosFromT(ref_->T_w_c_) - p_in_world;
        double angle = basics::calcAngleBetweenTwoVectors(vec_p_to_cam_curr, vec_p_to_cam_prev);
        angles.push_back(angle / 3.1415926 * 180.0);
    }

    // Get statistics
    vector<double> sort_a = angles;
    sort(sort_a.begin(), sort_a.end());
    double mean_angle = accumulate(sort_a.begin(), sort_a.end(), 0.0) / N;
    double median_angle = sort_a[N / 2];
    printf("Triangulation angle: mean=%f, median=%f, min=%f, max=%f\n",
           mean_angle,   // mean
           median_angle, // median
           sort_a[0],    // min
           sort_a[N - 1] // max
    );

    // Get good triangulation points

    vector<cv::Point3f> old_inlier_points = curr_->inliers_pts3d_;
    curr_->inliers_pts3d_.clear();

    vector<double> old_angles = angles;
    angles.clear();

    for (int i = 0; i < N; i++)
    {
        if (old_angles[i] < min_triang_angle ||
            old_angles[i] / median_angle > max_ratio_between_max_angle_and_median_angle)
            continue;
        cv::DMatch dmatch = curr_->inliers_matches_with_ref_[i];
        curr_->inliers_matches_for_3d_.push_back(dmatch);
        curr_->inliers_pts3d_.push_back(old_inlier_points[i]);
        angles.push_back(old_angles[i]);
    }
    return;
}

// ------------------- Tracking -------------------
bool VisualOdometry::checkLargeMoveForAddKeyFrame_(Frame::Ptr curr, Frame::Ptr ref)
{
    cv::Mat T_key_to_curr = ref->T_w_c_.inv() * curr->T_w_c_;
    cv::Mat R, t, R_vec;
    basics::getRtFromT(T_key_to_curr, R, t);
    cv::Rodrigues(R, R_vec);

    static const double min_dist_between_two_keyframes = basics::Config::get<double>("min_dist_between_two_keyframes");
    // static const double min_rotation_angle_betwen_two_keyframes = basics::Config::get<double>("min_rotation_angle_betwen_two_keyframes");

    double moved_dist = basics::calcMatNorm(t);
    double rotated_angle = basics::calcMatNorm(R_vec);

    printf("Wrt prev keyframe, relative dist = %.5f, angle = %.5f\n", moved_dist, rotated_angle);

    // Satisfy each one will be a good keyframe
    bool res = moved_dist > min_dist_between_two_keyframes;
    return res;
}

bool VisualOdometry::poseEstimationPnP_()
{
    // -- From the local map, find the keypoints that fall into the current view
    vector<MapPoint::Ptr> candidate_mappoints_in_map;
    vector<cv::Point2f> candidate_2d_pts_in_image;
    cv::Mat corresponding_mappoints_descriptors;
    getMappointsInCurrentView_(
        candidate_mappoints_in_map,
        candidate_2d_pts_in_image,
        corresponding_mappoints_descriptors);
    vector<cv::KeyPoint> candidate_2d_kpts_in_image = geometry::pts2Keypts(candidate_2d_pts_in_image);

    // -- Compare descriptors to find matches, and extract 3d 2d correspondance
    static const float max_matching_pixel_dist_in_pnp =
        basics::Config::get<float>("max_matching_pixel_dist_in_pnp");
    static const int method_index = basics::Config::get<float>("feature_match_method_index_pnp");
    geometry::matchFeatures(
        corresponding_mappoints_descriptors, curr_->descriptors_,
        curr_->matches_with_map_,
        method_index,
        false,
        candidate_2d_kpts_in_image, curr_->keypoints_,
        max_matching_pixel_dist_in_pnp);

    const int num_matches = curr_->matches_with_map_.size();
    cout << "Number of 3d-2d pairs: " << num_matches << endl;
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d; // a point's 2d pos in image2 pixel curr_
    for (int i = 0; i < num_matches; i++)
    {
        cv::DMatch &match = curr_->matches_with_map_[i];
        MapPoint::Ptr mappoint = candidate_mappoints_in_map[match.queryIdx];
        pts_3d.push_back(mappoint->pos_);
        pts_2d.push_back(curr_->keypoints_[match.trainIdx].pt);
    }

    // -- Solve PnP, get T_world_to_camera
    constexpr int kMinPtsForPnP = 5;
    static const double max_possible_dist_to_prev_keyframe =
        basics::Config::get<double>("max_possible_dist_to_prev_keyframe");

    cv::Mat pnp_inliers_mask; // type = 32SC1, size = 999x1
    cv::Mat R_vec, t;

    bool is_pnp_good = num_matches >= kMinPtsForPnP;
    if (is_pnp_good)
    {
        bool useExtrinsicGuess = false;
        int iterationsCount = 100;
        float reprojectionError = 2.0;
        double confidence = 0.999;
        cv::solvePnPRansac(pts_3d, pts_2d, curr_->camera_->K_, cv::Mat(), R_vec, t,
                           useExtrinsicGuess,
                           iterationsCount, reprojectionError, confidence, pnp_inliers_mask);
        // Output two variables:
        //      1. curr_->matches_with_map_
        //      2. curr_->T_w_c_

        cv::Mat R;
        cv::Rodrigues(R_vec, R); // angle-axis rotation to 3x3 rotation matrix

        // -- Get inlier matches used in PnP
        vector<cv::Point2f> tmp_pts_2d;
        vector<cv::Point3f *> inlier_candidates_pos;
        vector<MapPoint::Ptr> inlier_candidates;
        vector<cv::DMatch> tmp_matches_with_map_;
        int num_inliers = pnp_inliers_mask.rows;
        for (int i = 0; i < num_inliers; i++)
        {
            int good_idx = pnp_inliers_mask.at<int>(i, 0);

            // good match
            cv::DMatch &match = curr_->matches_with_map_[good_idx];
            tmp_matches_with_map_.push_back(match);

            // good pts 2d
            tmp_pts_2d.push_back(pts_2d[good_idx]);

            // good pts 3d
            MapPoint::Ptr inlier_mappoint = candidate_mappoints_in_map[match.queryIdx];
            inlier_candidates_pos.push_back(&(inlier_mappoint->pos_));
            inlier_mappoint->matched_times_++;

            // Update graph info
            curr_->inliers_to_mappt_connections_[match.trainIdx] = PtConn{-1, inlier_mappoint->id_};
        }
        pts_2d.swap(tmp_pts_2d);
        curr_->matches_with_map_.swap(tmp_matches_with_map_);

        // -- Update current camera pos
        curr_->T_w_c_ = basics::convertRt2T(R, t).inv();

        // -- Check relative motion with previous frame
        cv::Mat R_prev, t_prev, R_curr, t_curr;
        basics::getRtFromT(curr_->T_w_c_, R_prev, t_prev);
        basics::getRtFromT(prev_->T_w_c_, R_curr, t_curr);
        double dist_to_prev_keyframe = basics::calcMatNorm(t_prev - t_curr);
        if (dist_to_prev_keyframe >= max_possible_dist_to_prev_keyframe)
        {
            printf("PnP: distance with prev keyframe is %.3f. Threshold is %.3f.\n",
                   dist_to_prev_keyframe, max_possible_dist_to_prev_keyframe);
            is_pnp_good = false;
        }
    }
    else
    {
        printf("PnP num inlier matches: %d.\n", num_matches);
    }

    if (!is_pnp_good) // Set this frame's pose the same as previous frame
    {
        curr_->T_w_c_ = prev_->T_w_c_.clone();
    }
    return is_pnp_good;
}

// bundle adjustment
void VisualOdometry::callBundleAdjustment_()
{
    // Read settings from config.yaml
    static const bool is_enable_ba = basics::Config::getBool("is_enable_ba");
    static const int num_prev_frames_to_opti_by_ba = basics::Config::get<int>("num_prev_frames_to_opti_by_ba");
    static const vector<double> im = basics::str2vecdouble(
        basics::Config::get<string>("information_matrix"));
    static const bool is_ba_fix_map_points = basics::Config::getBool("is_ba_fix_map_points");
    static const bool is_ba_update_map_points = !is_ba_fix_map_points;

    // Set params
    const int kTotalFrames = frames_buff_.size();
    const int kNumFramesForBA = std::min(num_prev_frames_to_opti_by_ba, kTotalFrames - 1);
    const static cv::Mat information_matrix = (cv::Mat_<double>(2, 2) << im[0], im[1], im[2], im[3]);

    if (is_enable_ba != true)
    {
        printf("\nNot using bundle adjustment ... \n");
        return;
    }
    printf("\nCalling bundle adjustment on %d frames ... \n", kNumFramesForBA);

    // Measurement (which is fixed; truth)
    vector<vector<cv::Point2f *>> v_pts_2d;
    vector<vector<int>> v_pts_2d_to_3d_idx;

    // Things to to optimize
    std::unordered_map<int, cv::Point3f *> um_pts_3d_in_prev_frames;
    vector<cv::Point3f *> v_pts_3d_only_in_curr;
    vector<cv::Mat *> v_camera_poses;

    // Set up input vars
    int ith_frame = 0;
    for (int ith_frame_in_buff = kTotalFrames - 1;
         ith_frame_in_buff >= kTotalFrames - kNumFramesForBA;
         ith_frame_in_buff--, ith_frame++)
    {
        Frame::Ptr frame = frames_buff_[ith_frame_in_buff];
        int num_mappt_in_frame = frame->inliers_to_mappt_connections_.size();
        if (num_mappt_in_frame < 3)
        {
            continue; // Too few mappoints. Not optimizing this frame
        }
        printf("Frame id: %d, num map points = %d\n", frame->id_, num_mappt_in_frame);
        v_pts_2d.push_back(vector<cv::Point2f *>());
        v_pts_2d_to_3d_idx.push_back(vector<int>());

        // Get camera poses
        v_camera_poses.push_back(&frame->T_w_c_);

        // Iterate through this camera's mappoints
        for (std::unordered_map<int, PtConn>::iterator ite = frame->inliers_to_mappt_connections_.begin();
             ite != frame->inliers_to_mappt_connections_.end(); ite++)
        {
            int kpt_idx = ite->first;
            int mappt_idx = ite->second.pt_map_idx;
            if (map_->map_points_.find(mappt_idx) == map_->map_points_.end())
                continue; // point has been deleted

            // Get 2d pos
            v_pts_2d.back().push_back(&(frame->keypoints_[kpt_idx].pt));
            v_pts_2d_to_3d_idx.back().push_back(mappt_idx);

            // Get 3d pos
            cv::Point3f *p = &(map_->map_points_[mappt_idx]->pos_);
            um_pts_3d_in_prev_frames[mappt_idx] = p;
            if (ith_frame == 0)
                v_pts_3d_only_in_curr.push_back(p);
        }
    }
    // Bundle Adjustment
    cv::Mat pose_src = basics::getPosFromT(curr_->T_w_c_);
    if (1)
    {
        optimization::bundleAdjustment(
            v_pts_2d, v_pts_2d_to_3d_idx, curr_->camera_->K_,
            um_pts_3d_in_prev_frames, v_camera_poses,
            information_matrix,
            is_ba_fix_map_points, is_ba_update_map_points);
    }
    else // This is a deprecated function. I will remove it later.
    {
        optimization::optimizeSingleFrame(
            v_pts_2d[0], curr_->camera_->K_,
            v_pts_3d_only_in_curr, curr_->T_w_c_,
            is_ba_fix_map_points, is_ba_update_map_points); // Update pts_3d and curr_->T_w_c_
    }

    // Print result
    cv::Mat pose_new = basics::getPosFromT(curr_->T_w_c_);
    printf("Cam pos: Before:{%.5f,%.5f,%.5f}, After:{%.5f,%.5f,%.5f}\n",
           pose_src.at<double>(0, 0), pose_src.at<double>(1, 0), pose_src.at<double>(2, 0),
           pose_new.at<double>(0, 0), pose_new.at<double>(1, 0), pose_new.at<double>(2, 0));
    printf("Bundle adjustment finishes... \n\n");
}

// ------------------- Mapping -------------------

void VisualOdometry::addKeyFrame_(Frame::Ptr frame)
{
    map_->insertKeyFrame(frame);
    ref_ = frame;
}

void VisualOdometry::optimizeMap_()
{
    static const double default_erase = 0.1;
    static double map_point_erase_ratio = default_erase;

    // remove the hardly seen and no visible points
    for (auto iter = map_->map_points_.begin(); iter != map_->map_points_.end();)
    {
        if (!curr_->isInFrame(iter->second->pos_))
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }

        float match_ratio = float(iter->second->matched_times_) / iter->second->visible_times_;
        if (match_ratio < map_point_erase_ratio)
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }

        double angle = getViewAngle_(curr_, iter->second);
        if (angle > M_PI / 4.)
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        iter++;
    }

    if (map_->map_points_.size() > 1000)
    {
        // TODO map is too large, remove some one
        map_point_erase_ratio += 0.05;
    }
    else
        map_point_erase_ratio = default_erase;
    cout << "map points: " << map_->map_points_.size() << endl;
}

void VisualOdometry::pushCurrPointsToMap_()
{
    // -- Input
    const vector<cv::Point3f> &inliers_pts3d_in_curr = curr_->inliers_pts3d_;
    const cv::Mat &T_w_curr = curr_->T_w_c_;
    const cv::Mat &descriptors = curr_->descriptors_;
    const vector<vector<unsigned char>> &kpts_colors = curr_->kpts_colors_;
    const vector<cv::DMatch> &inliers_matches_for_3d = curr_->inliers_matches_for_3d_;

    // -- Output
    std::unordered_map<int, PtConn> &inliers_to_mappt_connections = curr_->inliers_to_mappt_connections_;

    // -- Start
    for (int i = 0; i < inliers_matches_for_3d.size(); i++)
    {
        const cv::DMatch &dm = inliers_matches_for_3d[i];
        int pt_idx = dm.trainIdx;
        int map_point_id;

        // Points already triangulated in previous frames.
        //      Just find the mappoint, no need to create new.
        if (1 && ref_->isMappoint(dm.queryIdx))
        {
            map_point_id = ref_->inliers_to_mappt_connections_[dm.queryIdx].pt_map_idx;
        }
        else // Not triangulated before. Create and push to map.
        {

            // Change coordinate of 3d points to world frame
            cv::Point3f world_pos = basics::preTranslatePoint3f(inliers_pts3d_in_curr[i], T_w_curr);

            // Create map point
            MapPoint::Ptr map_point(new MapPoint( // createMapPoint
                world_pos,
                descriptors.row(pt_idx).clone(),                                                        // descriptor
                basics::getNormalizedMat(basics::point3f_to_mat3x1(world_pos) - curr_->getCamCenter()), // view direction of the point
                kpts_colors[pt_idx][0], kpts_colors[pt_idx][1], kpts_colors[pt_idx][2]                  // rgb color
                ));
            map_point_id = map_point->id_;
            // cout<<map_point->id_ <<", "<< map_point->factory_id_<<endl;

            // Push to map
            map_->insertMapPoint(map_point);
        }
        // Update graph connection of current frame
        inliers_to_mappt_connections.insert({pt_idx, PtConn{dm.queryIdx, map_point_id}});
    }
    return;
}

double VisualOdometry::getViewAngle_(Frame::Ptr frame, MapPoint::Ptr point)
{
    cv::Mat n = basics::point3f_to_mat3x1(point->pos_) - frame->getCamCenter();
    n = basics::getNormalizedMat(n);
    cv::Mat vector_dot_product = n.t() * point->norm_;
    return acos(vector_dot_product.at<double>(0, 0));
}

} // namespace vo
} // namespace my_slam

#include "my_slam/geometry/motion_estimation.h"

#define DEBUG_PRINT_RESULT false

namespace my_slam
{
namespace geometry
{

int helperEstimatePossibleRelativePosesByEpipolarGeometry(
    const vector<cv::KeyPoint> &keypoints_1,
    const vector<cv::KeyPoint> &keypoints_2,
    const vector<cv::DMatch> &matches,
    const cv::Mat &K,
    vector<cv::Mat> &list_R, vector<cv::Mat> &list_t,
    vector<vector<cv::DMatch>> &list_matches,
    vector<cv::Mat> &list_normal,
    vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
    bool is_print_res,
    bool is_calc_homo,
    const bool is_motion_cam2_to_cam1)
{
    list_R.clear();
    list_t.clear();
    list_matches.clear();
    list_normal.clear();
    sols_pts3d_in_cam1.clear();

    // Get matched points: pts_img1 & pts_img2
    // All computations after this step are operated on these matched points
    vector<cv::Point2f> pts_img1_all = convertkeypointsToPoint2f(keypoints_1);
    vector<cv::Point2f> pts_img2_all = convertkeypointsToPoint2f(keypoints_2);
    vector<cv::Point2f> pts_img1, pts_img2; // matched points
    extractPtsFromMatches(pts_img1_all, pts_img2_all, matches, pts_img1, pts_img2);
    vector<cv::Point2f> pts_on_np1, pts_on_np2; // matched points on camera normalized plane
    int num_matched = (int)matches.size();
    for (int i = 0; i < num_matched; i++)
    {
        pts_on_np1.push_back(pixel2CamNormPlane(pts_img1[i], K));
        pts_on_np2.push_back(pixel2CamNormPlane(pts_img2[i], K));
    }

    // Estiamte motion by Essential Matrix
    cv::Mat R_e, t_e, essential_matrix;
    vector<int> inliers_index_e; // index of the inliers
    estiMotionByEssential(pts_img1, pts_img2, K,
                          essential_matrix,
                          R_e, t_e, inliers_index_e);

    if (is_print_res && DEBUG_PRINT_RESULT)
    {
        printResult_estiMotionByEssential(essential_matrix, // debug
                                          inliers_index_e, R_e, t_e);
    }

    // Estiamte motion by Homography Matrix
    vector<cv::Mat> R_h_list, t_h_list, normal_list;
    vector<int> inliers_index_h; // index of the inliers
    cv::Mat homography_matrix;
    if (is_calc_homo)
    {
        estiMotionByHomography(pts_img1, pts_img2, K,
                               /*output*/
                               homography_matrix,
                               R_h_list, t_h_list, normal_list,
                               inliers_index_h);
        removeWrongRtOfHomography(pts_on_np1, pts_on_np2, inliers_index_h, R_h_list, t_h_list, normal_list);
    }
    int num_h_solutions = R_h_list.size();
    if (is_print_res && DEBUG_PRINT_RESULT && is_calc_homo)
    {
        printResult_estiMotionByHomography(homography_matrix, // debug
                                           inliers_index_h, R_h_list, t_h_list, normal_list);
    }

    // Combine the motions from Essential/Homography
    // Return: vector<cv::Mat> list_R, list_t, list_normal;
    vector<vector<int>> list_inliers;
    list_R.push_back(R_e);
    list_t.push_back(t_e);
    list_normal.push_back(cv::Mat());
    list_inliers.push_back(inliers_index_e);
    for (int i = 0; i < num_h_solutions; i++)
    {
        list_R.push_back(R_h_list[i]);
        list_t.push_back(t_h_list[i]);
        list_normal.push_back(normal_list[i]);
        list_inliers.push_back(inliers_index_h);
    }
    int num_solutions = list_R.size();

    // Convert [inliers of matches] to the [cv::DMatch of all kpts]
    for (int i = 0; i < num_solutions; i++)
    {
        list_matches.push_back(vector<cv::DMatch>());
        const vector<int> &inliers = list_inliers[i];
        for (const int &idx : inliers)
        {
            list_matches[i].push_back(
                cv::DMatch(matches[idx].queryIdx, matches[idx].trainIdx, matches[idx].distance));
        }
    }

    // Triangulation for all 3 solutions
    // return: vector<vector<cv::Point3f>> sols_pts3d_in_cam1;
    for (int i = 0; i < num_solutions; i++)
    {
        vector<cv::Point3f> pts3d_in_cam1;
        doTriangulation(pts_on_np1, pts_on_np2, list_R[i], list_t[i], list_inliers[i], pts3d_in_cam1);
        sols_pts3d_in_cam1.push_back(pts3d_in_cam1);
    }

    // Change frame
    // Caution: This should be done after all other algorithms
    if (is_motion_cam2_to_cam1 == false)
        for (int i = 0; i < num_solutions; i++)
            basics::invRt(list_R[i], list_t[i]);

    // Debug EpipolarError and TriangulationResult
    if (is_print_res && !is_calc_homo)
    {
        print_EpipolarError_and_TriangulationResult_By_Solution(
            pts_img1, pts_img2, pts_on_np1, pts_on_np2,
            sols_pts3d_in_cam1, list_inliers, list_R, list_t, K);
    }
    else if (is_print_res && is_calc_homo)
    {
        print_EpipolarError_and_TriangulationResult_By_Common_Inlier(
            pts_img1, pts_img2, pts_on_np1, pts_on_np2,
            sols_pts3d_in_cam1, list_inliers, list_R, list_t, K);
    }

    // -- Choose a solution
    double score_E = checkEssentialScore(essential_matrix, K, pts_img1, pts_img2, inliers_index_e);
    double score_H = checkHomographyScore(homography_matrix, pts_img1, pts_img2, inliers_index_h);

    double ratio = score_H / (score_E + score_H);
    printf("Evaluate E/H score: E = %.1f, H = %.1f, H/(E+H)=%.3f\n", score_E, score_H, ratio);
    int best_sol = 0;
    if (ratio > 0.5)
    {
        best_sol = 1;
        double largest_norm_z = fabs(list_normal[1].at<double>(2, 0));
        for (int i = 2; i < num_solutions; i++)
        {
            double norm_z = fabs(list_normal[i].at<double>(2, 0));
            if (norm_z > largest_norm_z)
            {
                largest_norm_z = norm_z;
                best_sol = i;
            }
        }
    }
    printf("Best index = %d, which is [%s].\n\n", best_sol, best_sol == 0 ? "E" : "H");
    return best_sol;
}

void helperEstiMotionByEssential(
    const vector<cv::KeyPoint> &keypoints_1,
    const vector<cv::KeyPoint> &keypoints_2,
    const vector<cv::DMatch> &matches,
    const cv::Mat &K,
    cv::Mat &R, cv::Mat &t,
    vector<cv::DMatch> &inlier_matches,
    bool is_print_res)
{
    vector<cv::Point2f> pts_in_img1, pts_in_img2;
    extractPtsFromMatches(keypoints_1, keypoints_2, matches, pts_in_img1, pts_in_img2);
    cv::Mat essential_matrix;
    vector<int> inliers_index;
    estiMotionByEssential(pts_in_img1, pts_in_img2, K, essential_matrix, R, t, inliers_index);
    inlier_matches.clear();
    for (int idx : inliers_index)
    {
        const cv::DMatch &m = matches[idx];
        inlier_matches.push_back(
            cv::DMatch(m.queryIdx, m.trainIdx, m.distance));
    }
}

vector<cv::DMatch> helperFindInlierMatchesByEpipolarCons(
    const vector<cv::KeyPoint> &keypoints_1,
    const vector<cv::KeyPoint> &keypoints_2,
    const vector<cv::DMatch> &matches,
    const cv::Mat &K)
{
    // Output
    vector<cv::DMatch> inlier_matches;

    // Estimate Essential to get inlier matches
    cv::Mat dummy_R, dummy_t;
    helperEstiMotionByEssential(
        keypoints_1, keypoints_2,
        matches, K,
        dummy_R, dummy_t, inlier_matches);
    return inlier_matches;
}


// Triangulate points
vector<cv::Point3f> helperTriangulatePoints(
    const vector<cv::KeyPoint> &prev_kpts, const vector<cv::KeyPoint> &curr_kpts,
    const vector<cv::DMatch> &curr_inlier_matches,
    const cv::Mat &T_curr_to_prev,
    const cv::Mat &K)
{
    cv::Mat R_curr_to_prev, t_curr_to_prev;
    basics::getRtFromT(T_curr_to_prev, R_curr_to_prev, t_curr_to_prev);
    return helperTriangulatePoints(prev_kpts, curr_kpts, curr_inlier_matches,
                                   R_curr_to_prev, t_curr_to_prev, K);
}

vector<cv::Point3f> helperTriangulatePoints(
    const vector<cv::KeyPoint> &prev_kpts, const vector<cv::KeyPoint> &curr_kpts,
    const vector<cv::DMatch> &curr_inlier_matches,
    const cv::Mat &R_curr_to_prev, const cv::Mat &t_curr_to_prev,
    const cv::Mat &K)
{
    // Extract matched keypoints, and convert to camera normalized plane
    vector<cv::Point2f> pts_img1, pts_img2;
    extractPtsFromMatches(prev_kpts, curr_kpts, curr_inlier_matches, pts_img1, pts_img2);

    vector<cv::Point2f> pts_on_np1, pts_on_np2; // matched points on camera normalized plane
    for (const cv::Point2f &pt : pts_img1)
        pts_on_np1.push_back(pixel2CamNormPlane(pt, K));
    for (const cv::Point2f &pt : pts_img2)
        pts_on_np2.push_back(pixel2CamNormPlane(pt, K));

    // Set inliers indices
    const cv::Mat &R = R_curr_to_prev, &t = t_curr_to_prev; //rename
    vector<int> inliers;
    for (int i = 0; i < pts_on_np1.size(); i++)
        inliers.push_back(i); // all are inliers

    // Do triangulation
    vector<cv::Point3f> pts_3d_in_prev; // pts 3d pos to compute
    doTriangulation(pts_on_np1, pts_on_np2, R, t, inliers, pts_3d_in_prev);

    // Change pos to current frame
    vector<cv::Point3f> pts_3d_in_curr;
    for (const cv::Point3f &pt3d : pts_3d_in_prev)
        pts_3d_in_curr.push_back(basics::transCoord(pt3d, R, t));

    // Return
    return pts_3d_in_curr;
}

double computeScoreForEH(double d2, double TM)
{
    double TAO = 5.99; // Same as TH
    if (d2 < TM)
        return TAO - d2;
    else
        return 0;
}

// (Deprecated) Choose EH by triangulation error. This helps nothing.
void helperEvalEppiAndTriangErrors(
    const vector<cv::KeyPoint> &keypoints_1,
    const vector<cv::KeyPoint> &keypoints_2,
    const vector<vector<cv::DMatch>> &list_matches,
    const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1_by_triang,
    const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const vector<cv::Mat> &list_normal,
    const cv::Mat &K,
    bool is_print_res)
{
    vector<double> list_error_epipolar;
    vector<double> list_error_triangulation;
    int num_solutions = list_R.size();

    const double TF = 3.84, TH = 5.99; // Param for computing mean_score. Here F(fundmental)==E(essential)

    for (int i = 0; i < num_solutions; i++)
    {
        const cv::Mat &R = list_R[i], &t = list_t[i];
        const vector<cv::DMatch> &matches = list_matches[i];
        const vector<cv::Point3f> &pts3d = sols_pts3d_in_cam1_by_triang[i];
        vector<cv::Point2f> inlpts1, inlpts2;
        extractPtsFromMatches(keypoints_1, keypoints_2, matches, inlpts1, inlpts2);

        // epipolar error
        double err_epipolar = computeEpipolarConsError(inlpts1, inlpts2, R, t, K);

        // In image frame,  the error between triangulation and real
        double err_triangulation = 0; // more correctly called: symmetric transfer error
        double mean_score = 0;        // f_rc(d2)+f_cr(d2) from ORB-SLAM
        int num_inlier_pts = inlpts1.size();
        for (int idx_inlier = 0; idx_inlier < num_inlier_pts; idx_inlier++)
        {
            const cv::Point2f &p1 = inlpts1[idx_inlier], &p2 = inlpts2[idx_inlier];
            // print triangulation result
            cv::Mat pts3dc1 = basics::point3f_to_mat3x1(pts3d[idx_inlier]); // 3d pos in camera 1
            cv::Mat pts3dc2 = R * pts3dc1 + t;
            cv::Point2f pts2dc1 = cam2pixel(pts3dc1, K);
            cv::Point2f pts2dc2 = cam2pixel(pts3dc2, K);
            double dist1 = calcErrorSquare(p1, pts2dc1), dist2 = calcErrorSquare(p2, pts2dc2);
            err_triangulation += dist1 + dist2;
            // printf("%dth inlier, err_triangulation = %f\n", idx_inlier, err_triangulation);
        }
        if (num_inlier_pts == 0)
        {
            err_triangulation = 9999999999;
            mean_score = 0;
        }
        else
        {
            err_triangulation = sqrt(err_triangulation / 2.0 / num_inlier_pts);
            mean_score /= num_inlier_pts;
        }
        // Store the error
        list_error_epipolar.push_back(err_epipolar);
        list_error_triangulation.push_back(err_triangulation);
    }

    // -- Print out result
    if (is_print_res)
    {
        printf("\n------------------------------------\n");
        printf("Print the mean error of each E/H method by using the inlier points.\n");
        for (int i = 0; i < num_solutions; i++)
        {

            printf("\n---------------\n");
            printf("Solution %d, num inliers = %d \n", i, (int)list_matches[i].size());
            basics::print_R_t(list_R[i], list_t[i]);
            if (!list_normal[i].empty())
                cout << "norm is:" << (list_normal[i]).t() << endl;
            printf("-- Epipolar cons error = %f \n", list_error_epipolar[i]);
            printf("-- Triangulation error = %f \n", list_error_triangulation[i]);
        }
    }
}

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// ----------- debug functions --------------------------------------------------
// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------

void printResult_estiMotionByEssential(
    const cv::Mat &essential_matrix,
    const vector<int> &inliers_index,
    const cv::Mat &R,
    const cv::Mat &t)
{
    cout << endl
         << "=====" << endl;
    cout << "* Essential_matrix (by findEssentialMat) is: " << endl
         << essential_matrix << endl;
    cout << "* Number of inliers (after triangulation): " << inliers_index.size() << endl;
    cout << "* Recovering R and t from essential matrix:" << endl;
    basics::print_R_t(R, t);
    cout << endl;
}

void printResult_estiMotionByHomography(
    const cv::Mat &homography_matrix,
    const vector<int> &inliers_index,
    const vector<cv::Mat> &Rs, const vector<cv::Mat> &ts,
    vector<cv::Mat> &normals)
{
    cout << endl
         << "=====" << endl;
    cout << "* Homography_matrix (by findHomography) is " << endl
         << homography_matrix << endl;
    cout << "* Number of inliers: " << inliers_index.size() << endl;

    // cout << "The inliers_mask is " << inliers_mask << endl;
    // !!! add it later

    int num_solutions = Rs.size();
    for (int i = 0; i < num_solutions; i++)
    {
        cout << endl;
        cout << "* Solution " << i + 1 << ":" << endl; // Start index from 1. The index 0 is for essential matrix.
        basics::print_R_t(Rs[i], ts[i]);
        cout << "plane normal: " << normals[i].t() << endl;
    }
}

// Check [Epipoloar error] and [Triangulation result] for each common inlier in both E and H
void print_EpipolarError_and_TriangulationResult_By_Common_Inlier(
    const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
    const vector<cv::Point2f> &pts_on_np1, const vector<cv::Point2f> &pts_on_np2,
    const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
    const vector<vector<int>> &list_inliers,
    const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const cv::Mat &K)
{
    constexpr int kMaxNumPtsToCheckAndPrint = 1000;
    int num_solutions = list_R.size();
    const vector<int> &inliers_index_e = list_inliers[0];
    const vector<int> &inliers_index_h = list_inliers[1];

    cout << "\n---------------------------------------" << endl;
    cout << "Check [Epipoloar error] and [Triangulation result]" << endl;
    cout << "for the first " << kMaxNumPtsToCheckAndPrint << " points:";

    // Iterate through points.
    int cnt = 0;
    int num_points = pts_img1.size();
    for (int i = 0; i < num_points && cnt < kMaxNumPtsToCheckAndPrint; i++)
    {
        auto pe = find(inliers_index_e.begin(), inliers_index_e.end(), i);
        auto ph = find(inliers_index_h.begin(), inliers_index_h.end(), i);
        if (pe == inliers_index_e.end() || ph == inliers_index_h.end())
            continue;
        int ith_in_e_inliers = pe - inliers_index_e.begin();
        int ith_in_h_inliers = ph - inliers_index_h.begin();
        cout << "\n--------------" << endl;
        printf("Printing the %dth (in common) and %dth (in matched) point's real position in image:\n", cnt++, i);

        // Print point pos in image frame.
        cv::Point2f p1 = pts_img1[i], p2 = pts_img2[i];
        cout << "cam1, pixel pos (u,v): " << p1 << endl;
        cout << "cam2, pixel pos (u,v): " << p2 << endl;

        // Print result of each method.
        cv::Point2f p_cam1 = pts_on_np1[i]; // point pos on the normalized plane
        cv::Point2f p_cam2 = pts_on_np2[i];
        for (int j = 0; j < num_solutions; j++)
        {
            const cv::Mat &R = list_R[j], &t = list_t[j];

            // print epipolar error
            double err_epipolar = computeEpipolarConsError(p1, p2, R, t, K);
            cout << "===solu " << j << ": epipolar_error*1e6 is " << err_epipolar * 1e6 << endl;

            // print triangulation result
            int ith_in_curr_sol;
            if (j == 0)
                ith_in_curr_sol = ith_in_e_inliers;
            else
                ith_in_curr_sol = ith_in_h_inliers;

            cv::Mat pts3dc1 = basics::point3f_to_mat3x1(sols_pts3d_in_cam1[j][ith_in_curr_sol]); // 3d pos in camera 1
            cv::Mat pts3dc2 = R * pts3dc1 + t;
            cv::Point2f pts2dc1 = cam2pixel(pts3dc1, K);
            cv::Point2f pts2dc2 = cam2pixel(pts3dc2, K);

            cout << "-- In img1, pos: " << pts2dc1 << endl;
            cout << "-- In img2, pos: " << pts2dc2 << endl;
            cout << "-- On cam1, pos: " << pts3dc1.t() << endl;
            cout << "-- On cam2, pos: " << pts3dc2.t() << endl;
        }

        cout << endl;
    }
}

// Print each solution's result in order
void print_EpipolarError_and_TriangulationResult_By_Solution(
    const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
    const vector<cv::Point2f> &pts_on_np1, const vector<cv::Point2f> &pts_on_np2,
    const vector<vector<cv::Point3f>> &sols_pts3d_in_cam1,
    const vector<vector<int>> &list_inliers,
    const vector<cv::Mat> &list_R, const vector<cv::Mat> &list_t, const cv::Mat &K)
{
    cout << "\n---------------------------------------" << endl;
    cout << "Check [Epipoloar error] and [Triangulation result]" << endl;
    cout << "for each solution. Printing from back to front." << endl;
    int num_solutions = list_R.size();
    for (int j = 0; j < num_solutions; j++)
    {
        const cv::Mat &R = list_R[j], &t = list_t[j];
        const vector<int> &inliers = list_inliers[j];
        int num_inliers = inliers.size();
        constexpr int kMaxNumInliersToPrint = 5;
        for (int _idx_inlier = 0; _idx_inlier < std::min(kMaxNumInliersToPrint, num_inliers); _idx_inlier++)
        {
            int idx_inlier = num_inliers - 1 - _idx_inlier;
            int idx_pts = inliers[idx_inlier];
            cout << "\n--------------" << endl;
            printf("Printing the %dth last inlier point in solution %d\n", _idx_inlier, j);
            printf("which is %dth cv::KeyPoint\n", idx_pts);

            // Print point pos in image frame.
            cv::Point2f p1 = pts_img1[idx_pts], p2 = pts_img2[idx_pts];
            cout << "cam1, pixel pos (u,v): " << p1 << endl;
            cout << "cam2, pixel pos (u,v): " << p2 << endl;

            // print epipolar error
            double err_epipolar = computeEpipolarConsError(p1, p2, R, t, K);
            cout << "===solu " << j << ": epipolar_error*1e6 is " << err_epipolar * 1e6 << endl;

            // print triangulation result
            cv::Mat pts3dc1 = basics::point3f_to_mat3x1(sols_pts3d_in_cam1[j][idx_inlier]); // 3d pos in camera 1
            cv::Mat pts3dc2 = R * pts3dc1 + t;
            cv::Point2f pts2dc1 = cam2pixel(pts3dc1, K);
            // cv::Point2f pts2dc1 = cam2pixel(sols_pts3d_in_cam1[j][idx_inlier], K);
            cv::Point2f pts2dc2 = cam2pixel(pts3dc2, K);

            cout << "-- In img1, pos: " << pts2dc1 << endl;
            cout << "-- In img2, pos: " << pts2dc2 << endl;
            cout << "-- On cam1, pos: " << pts3dc1.t() << endl;
            cout << "-- On cam2, pos: " << pts3dc2.t() << endl;
        }
    }
}

double checkEssentialScore(const cv::Mat &E21, const cv::Mat &K, const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                           vector<int> &inliers_index, double sigma)
{
    vector<int> inliers_index_new;

    // Essential to Fundmental
    cv::Mat Kinv = K.inv(), KinvT;
    cv::transpose(Kinv, KinvT);
    cv::Mat F21 = KinvT * E21 * Kinv;

    const double f11 = F21.at<double>(0, 0);
    const double f12 = F21.at<double>(0, 1);
    const double f13 = F21.at<double>(0, 2);
    const double f21 = F21.at<double>(1, 0);
    const double f22 = F21.at<double>(1, 1);
    const double f23 = F21.at<double>(1, 2);
    const double f31 = F21.at<double>(2, 0);
    const double f32 = F21.at<double>(2, 1);
    const double f33 = F21.at<double>(2, 2);

    double score = 0;

    const double th = 3.841;
    const double thScore = 5.991;

    const double invSigmaSquare = 1.0 / (sigma * sigma);

    int N = inliers_index.size();
    for (int i = 0; i < N; i++)
    {
        bool good_point = true;

        const cv::Point2f &p1 = pts_img1[inliers_index[i]];
        const cv::Point2f &p2 = pts_img2[inliers_index[i]];

        const double u1 = p1.x, v1 = p1.y;
        const double u2 = p2.x, v2 = p2.y;

        // Reprojection error in second image == Epipolar constraint error
        // l2=F21x1=(a2,b2,c2)

        const double a2 = f11 * u1 + f12 * v1 + f13;
        const double b2 = f21 * u1 + f22 * v1 + f23;
        const double c2 = f31 * u1 + f32 * v1 + f33;

        const double num2 = a2 * u2 + b2 * v2 + c2;
        const double squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

        const double chiSquare1 = squareDist1 * invSigmaSquare;
        if (chiSquare1 > th)
        {
            score += 0;
            good_point = false;
        }
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const double a1 = f11 * u2 + f21 * v2 + f31;
        const double b1 = f12 * u2 + f22 * v2 + f32;
        const double c1 = f13 * u2 + f23 * v2 + f33;

        const double num1 = a1 * u1 + b1 * v1 + c1;
        const double squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);
        const double chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
        {
            score += 0;
            good_point = false;
        }
        else
            score += thScore - chiSquare2;

        if (good_point)
            inliers_index_new.push_back(inliers_index[i]);
    }
    printf("E score: sum = %.1f, mean = %.2f\n", score, score / inliers_index.size());
    inliers_index_new.swap(inliers_index);
    return score;
}

double checkHomographyScore(const cv::Mat &H21, const vector<cv::Point2f> &pts_img1, const vector<cv::Point2f> &pts_img2,
                            vector<int> &inliers_index, double sigma)
{
    double score;                  // output
    vector<int> inliers_index_new; // output
    cv::Mat H12 = H21.inv();

    const double h11 = H21.at<double>(0, 0);
    const double h12 = H21.at<double>(0, 1);
    const double h13 = H21.at<double>(0, 2);
    const double h21 = H21.at<double>(1, 0);
    const double h22 = H21.at<double>(1, 1);
    const double h23 = H21.at<double>(1, 2);
    const double h31 = H21.at<double>(2, 0);
    const double h32 = H21.at<double>(2, 1);
    const double h33 = H21.at<double>(2, 2);

    const double h11inv = H12.at<double>(0, 0);
    const double h12inv = H12.at<double>(0, 1);
    const double h13inv = H12.at<double>(0, 2);
    const double h21inv = H12.at<double>(1, 0);
    const double h22inv = H12.at<double>(1, 1);
    const double h23inv = H12.at<double>(1, 2);
    const double h31inv = H12.at<double>(2, 0);
    const double h32inv = H12.at<double>(2, 1);
    const double h33inv = H12.at<double>(2, 2);

    const double th = 5.991;
    const double invSigmaSquare = 1.0 / (sigma * sigma);

    const int N = inliers_index.size();
    for (int i = 0; i < N; i++)
    {
        bool good_point = true;

        const cv::Point2f &p1 = pts_img1[inliers_index[i]];
        const cv::Point2f &p2 = pts_img2[inliers_index[i]];

        const double u1 = p1.x, v1 = p1.y;
        const double u2 = p2.x, v2 = p2.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        const double w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
        const double u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        const double v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

        const double squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

        const double chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            good_point = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const double w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
        const double u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
        const double v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

        const double squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

        const double chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            good_point = false;
        else
            score += th - chiSquare2;

        if (good_point)
            inliers_index_new.push_back(inliers_index[i]);
    }
    printf("H score: sum = %.1f, mean = %.2f\n", score, score / inliers_index.size());
    inliers_index_new.swap(inliers_index);
    return score;
}

} // namespace geometry
} // namespace my_slam

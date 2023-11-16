#include "zeno/zeno.h"
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/types/ListObject.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <igl/point_mesh_squared_distance.h>
#include <igl/gaussian_curvature.h>
#include <igl/principal_curvature.h>

namespace zeno {

namespace {

// 计算图像的梯度
void computeGradient(std::shared_ptr<PrimitiveObject> & image, std::vector<std::vector<float>>& gradientX, std::vector<std::vector<float>>& gradientY) {
    auto &ud = image->userData();
    int height  = ud.get2<int>("h");
    int width = ud.get2<int>("w");

    gradientX.resize(height, std::vector<float>(width));
    gradientY.resize(height, std::vector<float>(width));

#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x > 0 && x < width - 1) {
                gradientX[y][x] = (image->verts[y * width + x + 1][0] - image->verts[y * width + x  - 1])[0] / 2.0f;
            } else {
                gradientX[y][x] = 0.0f;
            }
            if (y > 0 && y < height - 1) {
                gradientY[y][x] = (image->verts[(y+1) * width + x][0] - image->verts[(y - 1) * width + x])[0] / 2.0f;
            } else {
                gradientY[y][x] = 0.0f;
            }
        }
    }
}
// 计算图像的曲率
void computeCurvature(std::shared_ptr<PrimitiveObject> & image, const std::vector<std::vector<float>>& gradientX,
                      const std::vector<std::vector<float>>& gradientY) {
    int height = gradientX.size();
    int width = gradientX[0].size();
    if(!image->verts.has_attr("curvature")){
        image->verts.add_attr<float>("curvature");
    }
    auto &cur = image->verts.attr<float>("curvature");
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = gradientX[y][x];
            float dy = gradientY[y][x];
            float dxx = 0.0f;
            float dyy = 0.0f;
            float dxy = 0.0f;

            if (x > 0 && x < width - 1) {
                dxx = gradientX[y][x + 1] - 2.0f * dx + gradientX[y][x - 1];
            }

            if (y > 0 && y < height - 1) {
                dyy = gradientY[y + 1][x] - 2.0f * dy + gradientY[y - 1][x];
            }

            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                dxy = (gradientX[y + 1][x + 1] - gradientX[y + 1][x - 1] - gradientX[y - 1][x + 1] + gradientX[y - 1][x - 1]) / 4.0f;
            }
            cur[y * width + x] = (dxx * dyy - dxy * dxy) / ((dxx + dyy) * (dxx + dyy) + 1e-6f);
        }
    }
}
// 计算几何体顶点的平均曲率
static void computeVertexCurvature(std::shared_ptr<PrimitiveObject> & prim) {
    auto &cur = prim->verts.add_attr<float>("curvature");
#pragma omp parallel for
    for (size_t i = 0; i < prim->verts.size(); ++i) {
        // 构建顶点的邻域面
        std::vector<size_t> neighborFaces;
        for (int m = 0;m < prim->tris.size();m++) {
            if (prim->tris[m][0] == i || prim->tris[m][1] == i || prim->tris[m][2] == i) {
                neighborFaces.push_back(m);
            }
        }
        // 构建邻域面法线矩阵
        Eigen::MatrixXd normals(3, neighborFaces.size());
        for (size_t j = 0; j < neighborFaces.size(); ++j) {
            auto & face = prim->tris[neighborFaces[j]];
            auto & vert = prim->verts;
            auto & v1 = face[0];
            auto & v2 = face[1];
            auto & v3 = face[2];

            Eigen::Vector3d v12(vert[v2][0] - vert[v1][0], vert[v2][1] - vert[v1][1], vert[v2][2] - vert[v1][2]);
            Eigen::Vector3d v13(vert[v3][0] - vert[v1][0], vert[v3][1] - vert[v1][1], vert[v3][2] - vert[v1][2]);
            Eigen::Vector3d normal = v12.cross(v13).normalized();

            normals(0, j) = normal.x();
            normals(1, j) = normal.y();
            normals(2, j) = normal.z();
        }
        // 计算邻域面法线的协方差矩阵
        Eigen::MatrixXd covariance = (normals * normals.transpose()) / neighborFaces.size();
        // 计算特征值和特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        // 计算曲率
        double curvature = eigenvalues.minCoeff() / eigenvalues.sum();
        cur[i] = curvature;
    }
}

struct PrimCurvature2: INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto type = get_input2<std::string>("type");
        if(type == "object"){
            computeVertexCurvature(prim);
        }
        else if(type == "image"){
            auto &ud = prim->userData();
            int w = ud.get2<int>("w");
            int h = ud.get2<int>("h");
            std::vector<std::vector<float>> gx(h, std::vector<float>(w, 0));
            std::vector<std::vector<float>> gy(h, std::vector<float>(w, 0));
            computeGradient(prim,gx, gy);
            computeCurvature(prim,gx,gy);
        }
        set_output("prim", prim);
    }
};
ZENDEFNODE(PrimCurvature2, {
    {
        {"PrimitiveObject", "prim"},
        {"enum object image", "type", "object"},
    },
    {
        {"PrimitiveObject", "prim"},
    },
    {},
    {"deprecated"},
});

//use igl
struct PrimCurvature: INode {
    void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto gaussianCurvature = get_input2<bool>("gaussianCurvature");
        auto curvature = get_input2<bool>("curvature");
        int n = prim->verts.size();
        int dim = 3;
        Eigen::MatrixXd V(n, dim);
        for (int i = 0; i < n; ++i) {
            V.row(i) << prim->verts[i][0], prim->verts[i][1], prim->verts[i][2];
        }
        int m = prim->tris.size();
        int vertices_per_face = 3;
        Eigen::MatrixXi F(m, vertices_per_face);
        for (int i = 0; i < m; ++i) {
            F.row(i) << prim->tris[i][0], prim->tris[i][1], prim->tris[i][2];
        }
        if(gaussianCurvature){
            Eigen::VectorXd K;
            igl::gaussian_curvature(V, F, K);
            prim->verts.add_attr<float>("gaussianCurvature");
            for(int i = 0;i < prim->verts.size();i++){
                prim->verts.attr<float>("gaussianCurvature")[i] = K(i);
            }
        }
        if(curvature){
            Eigen::MatrixXd PD1, PD2;
            Eigen::VectorXd PV1, PV2;
            igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);
            prim->verts.add_attr<float>("curvature");
            for(int i = 0;i < prim->verts.size();i++){
                prim->verts.attr<float>("curvature")[i] = (PV1(i) + PV2(i)) / 2.0;
            }
        }
        set_output("prim", prim);
    }
};
ZENDEFNODE(PrimCurvature, {
    {
        {"PrimitiveObject", "prim"},
        {"bool", "curvature", "0"},
        {"bool", "gaussianCurvature", "0"},
    },
    {
        {"PrimitiveObject", "prim"},
    },
    {},
    {"primitive"},
});

//struct CompCurvature: INode {
//    void apply() override {
//        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
//        auto threshold = get_input2<float>("threshold");
//        auto channel = get_input2<std::string>("channel");
//        auto &ud = image->userData();
//        int w = ud.get2<int>("w");
//        int h = ud.get2<int>("h");
//        cv::Mat imagecvgray(h, w, CV_32F);
//        cv::Mat imagecvcurvature(h, w, CV_32F);
//        if(channel == "R"){
//            for (int i = 0; i < h; i++) {
//                for (int j = 0; j < w; j++) {
//                    vec3f rgb = image->verts[i * w + j];
//                    imagecvgray.at<float>(i, j) = rgb[0];
//                }
//            }
//        }
//        else if(channel == "G"){
//            for (int i = 0; i < h; i++) {
//                for (int j = 0; j < w; j++) {
//                    vec3f rgb = image->verts[i * w + j];
//                    imagecvgray.at<float>(i, j) = rgb[1];
//                }
//            }
//        }
//        else if(channel == "B"){
//            for (int i = 0; i < h; i++) {
//                for (int j = 0; j < w; j++) {
//                    vec3f rgb = image->verts[i * w + j];
//                    imagecvgray.at<float>(i, j) = rgb[2];
//                }
//            }
//        }
//        // 计算图像的梯度
//        cv::Mat dx, dy;
//        cv::Sobel(imagecvgray, dx, CV_32F, 1, 0);
//        cv::Sobel(imagecvgray, dy, CV_32F, 0, 1);
//        // 计算梯度的二阶导数
//        cv::Mat dxx, dyy, dxy;
//        cv::Sobel(dx, dxx, CV_32F, 1, 0);
//        cv::Sobel(dy, dyy, CV_32F, 0, 1);
//        cv::Sobel(dx, dxy, CV_32F, 0, 1);
//        // 计算曲率
//        imagecvcurvature = (dxx.mul(dyy) - dxy.mul(dxy)) / ((dxx + dyy).mul(dxx + dyy));
//        for (int i = 0; i < h; i++) {
//            for (int j = 0; j < w; j++) {
//                float cur = imagecvcurvature.at<float>(i, j);
//                if(cur > threshold){
//                    image->verts[i * w + j] = {1,1,1};
//                }
//                else{
//                    image->verts[i * w + j] = {0,0,0};
//                }
//            }
//        }
//        set_output("image", image);
//    }
//};
//ZENDEFNODE(CompCurvature, {
//    {
//        {"image"},
//        {"float","threshold","0"},
//        {"enum R G B","channel","R"}
//    },
//    {
//        {"image"},
//    },
//    {},
//    {"Comp"},
//});
}
}
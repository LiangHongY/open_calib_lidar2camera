/*
    Copyright (C) 2024 by Yang LiangHong Limited. All rights reserved.
    Yang LiangHong <2252512364@qq.com>
*/
#include <iostream>
#include <vector>
#include <map>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/pca.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <json/json.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class ExtractPcdCornerPoint{
    protected:
        std::map<std::string, PointT> m_chessboard;
        static PointT tmp_p;
        std::vector<std::string> pcd_paths;
        int chessboard_width, chessboard_height;
        double gap;
        double chessboard_size_w, chessboard_size_h;
        double radius;
        std::vector<Eigen::Vector3d> chessboard_corner;
        std::map<std::string, std::vector<Eigen::Vector3d>> map_chessboards_corner;

    public:
        ExtractPcdCornerPoint(std::string file_path);
        void loadParam(std::string file_path);
        static void pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, void* args);
        void selectChessboard(std::string pcd_path);
        void gainChessboardCenter(std::string folder_path);
        void gainPointcloudFromCenter(std::string pcd_path, PointT center);
        void gainBox(const pcl::PointCloud<PointT>::Ptr& cloud, std::string pcd_path);
        void gainPoint(Eigen::Vector3d through_point, Eigen::Vector3d vertical_direction, double distance_to_move, Eigen::Vector3d& translated_point);
        void projectPointcloudToPlane(const PointCloudT::Ptr cloud,
                              const Eigen::Vector3f& plane_normal,
                              const pcl::PointXYZ& point_on_plane,
                              PointCloudT::Ptr projected_cloud);
        void gainCornerPoint(Eigen::Vector3d line_vector_1,double lenght_1, Eigen::Vector3d line_vector_2,
        double lenght_2, Eigen::Vector3d point_1, Eigen::Vector3d& point_2);

        void process(std::map<std::string, std::vector<Eigen::Vector3d>>& map_chessboards_cerner_points);
        void boundaryEstimation(PointCloudT::Ptr& in_cloud, PointCloudT::Ptr& out_cloud);
};
PointT ExtractPcdCornerPoint::tmp_p = PointT(0, 0, 0);

ExtractPcdCornerPoint::ExtractPcdCornerPoint(std::string file_path)
{
    loadParam(file_path);
}

void ExtractPcdCornerPoint::loadParam(std::string file_path){
    Json::Reader reader;
    Json::Value root;
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Error Opening " << file_path << std::endl;
        return;
    }
    if (reader.parse(in, root, false)) {
        auto name = root.getMemberNames();
        for(auto id = name.begin(); id != name.end(); ++id)
        {
            Json::Value data = root[*id];
            if(*id == "chessboard_width"){
                chessboard_width = int(data[0].asDouble());
            }
            else if(*id == "chessboard_height"){
                chessboard_height = int(data[0].asDouble());
            }
            else if(*id == "chessboard_grap"){
                gap = data[0].asDouble();
            }
            else if(*id == "chessboard_size"){
                chessboard_size_w = data[0].asDouble();
                chessboard_size_h = data[1].asDouble();
            }
            else if(*id == "extract_radius")
            {
                radius = data[0].asDouble();
            }
        }
    }
    in.close();
    return;
}

void ExtractPcdCornerPoint::projectPointcloudToPlane(const PointCloudT::Ptr cloud,
                              const Eigen::Vector3f& plane_normal,
                              const pcl::PointXYZ& point_on_plane,
                              PointCloudT::Ptr projected_cloud)
{
    Eigen::Vector3f basis_vector = Eigen::Vector3f::UnitX();
    if (std::abs(plane_normal.dot(basis_vector)) > 0.9) {
        basis_vector = Eigen::Vector3f::UnitY();
    }
    Eigen::Vector3f u = plane_normal.cross(basis_vector).normalized();
    Eigen::Vector3f v = plane_normal.cross(u).normalized();
    Eigen::Vector3f p = point_on_plane.getVector3fMap();
    for (const auto& point : cloud->points) {
        Eigen::Vector3f q = point.getVector3fMap();
        Eigen::Vector3f pq = q - p;
        float dist = pq.dot(plane_normal);
        Eigen::Vector3f projected_point = q - dist * plane_normal;
        projected_cloud->push_back(PointT(projected_point.x(), projected_point.y(), projected_point.z()));
    }
}

void ExtractPcdCornerPoint::pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, void* args) {
    if (event.getPointIndex() == -1) {
        return;
    }
    PointT point;
    event.getPoint(point.x, point.y, point.z);
    tmp_p = point;
    std::cout << "Point picked: " << point.x << " " << point.y << " " << point.z << std::endl;
}

void ExtractPcdCornerPoint::selectChessboard(std::string pcd_path)
{
    PointCloudT::Ptr cloud(new PointCloudT());
    pcl::io::loadPCDFile<PointT>(pcd_path, *cloud);
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler(cloud, 255, 255, 255); // 点颜色
    viewer->addPointCloud(cloud, color_handler, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");  // 设置可视化点大小
    viewer->registerPointPickingCallback(pointPickingEventOccurred, nullptr);   // 通过 Shfit+鼠标左键选择点
    // Start viewer
    viewer->spin();
}

void ExtractPcdCornerPoint::gainChessboardCenter(std::string folder_path)
{
    if (!boost::filesystem::exists(folder_path) || !boost::filesystem::is_directory(folder_path)) {
        std::cerr << "Invalid pcd folder path!" << std::endl;
    }
    for (const auto& entry : boost::filesystem::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".pcd") {
            pcd_paths.push_back(entry.path().string());
        }
    }
    sort(pcd_paths.begin(), pcd_paths.end());
    for (const auto& pcd_path : pcd_paths) {
        selectChessboard(pcd_path);
        m_chessboard[pcd_path] = tmp_p;
    }
}

void ExtractPcdCornerPoint::gainPointcloudFromCenter(std::string pcd_path, PointT center)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud);
    pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>);
    pcl::ExtractIndices<PointT> extract;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    for (size_t i = 0; i < cloud->size(); ++i) {
        PointT point = cloud->points[i];
        double distance = sqrt(pow(point.x - center.x, 2) + pow(point.y - center.y, 2) + pow(point.z - center.z, 2));
        if (distance <= radius) {
            inliers->indices.push_back(i);
        }
    }
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*filtered_cloud);
    pcl::PointCloud<PointT>::Ptr smooth_filtered_cloud(new pcl::PointCloud<PointT>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(filtered_cloud);
    sor.setMeanK(50);  // Number of nearest neighbors to compute mean distance
    sor.setStddevMulThresh(1.0);  // Standard deviation threshold
    sor.filter(*smooth_filtered_cloud);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_(new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    seg.setInputCloud(smooth_filtered_cloud);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.segment(*inliers_, *coefficients);
    PointT point_c;
    Eigen::Vector4f center_;
    pcl::compute3DCentroid(*cloud, *inliers, center_);
    point_c.x = center_[0];
    point_c.y = center_[1];
    point_c.z = center_[2];
    Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    PointCloudT::Ptr projected_cloud(new PointCloudT);
    projectPointcloudToPlane(smooth_filtered_cloud, normal, point_c, projected_cloud);
    gainBox(projected_cloud, pcd_path);
}

void ExtractPcdCornerPoint::gainPoint(Eigen::Vector3d through_point, Eigen::Vector3d vertical_direction, 
double distance_to_move, Eigen::Vector3d& translated_point)
{
    Eigen::Vector3d unit_vertical_direction = vertical_direction.normalized();
    Eigen::Vector3d translation_v = distance_to_move * unit_vertical_direction;
    translated_point = through_point + translation_v;
}

void ExtractPcdCornerPoint::gainCornerPoint(Eigen::Vector3d line_v_1,double lenght_1, Eigen::Vector3d line_v_2,
double lenght_2, Eigen::Vector3d point_1, Eigen::Vector3d& point_2)
{
    Eigen::Vector3d unit_line_v_1 = line_v_1.normalized();
    Eigen::Vector3d t_v_1 = lenght_1*unit_line_v_1;
    Eigen::Vector3d unit_line_v_2 = line_v_2.normalized();
    Eigen::Vector3d t_v_2 = lenght_2*unit_line_v_2;
    Eigen::Vector3d t_v_3 = t_v_1 + t_v_2;
    point_2 = point_1 + t_v_3;
}

void ExtractPcdCornerPoint::gainBox(const pcl::PointCloud<PointT>::Ptr& cloud, std::string pcd_path)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    seg.setInputCloud(cloud);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.segment(*inliers, *coefficients);
    PointT center;
    Eigen::Vector4f center_;
    pcl::compute3DCentroid(*cloud, *inliers, center_);
    center.x = center_[0];
    center.y = center_[1];
    center.z = center_[2];
    Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    pcl::PCA<PointT> pca;
    pca.setInputCloud(cloud);
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
    PointT min_pt, max_pt, geometric_center;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    geometric_center.x = 0.5 * (min_pt.x + max_pt.x);
    geometric_center.y = 0.5 * (min_pt.y + max_pt.y);
    geometric_center.z = 0.5 * (min_pt.z + max_pt.z);
    Eigen::Vector3d line_direction(eigen_vectors(0, 0), eigen_vectors(1, 0), eigen_vectors(2, 0));
    Eigen::Vector3d through_point(center_[0], center_[1], center_[2]);
    Eigen::Vector3d vertical_direction(eigen_vectors(0, 1), eigen_vectors(1, 1), eigen_vectors(2, 1));
    double lenght_1 = (chessboard_width - 1)*gap / 2;
    double lenght_2 = (chessboard_height -1)*gap / 2;
    Eigen::Vector3d new_point;
    chessboard_corner.clear();
    Eigen::Vector3d new_point_1, new_point_2, new_point_3, new_point_4;
    gainCornerPoint(line_direction, lenght_1, vertical_direction, lenght_2, through_point, new_point_1);
    chessboard_corner.push_back(new_point_1);
    gainCornerPoint(line_direction, -lenght_1, vertical_direction, lenght_2, through_point, new_point_2);
    chessboard_corner.push_back(new_point_2);
    gainCornerPoint(line_direction, -lenght_1, vertical_direction, -lenght_2, through_point, new_point_3);
    chessboard_corner.push_back(new_point_3);
    gainCornerPoint(line_direction, lenght_1, vertical_direction, -lenght_2, through_point, new_point_4);
    chessboard_corner.push_back(new_point_4);
    std::sort(chessboard_corner.begin(),chessboard_corner.end(),[](Eigen::Vector3d& a, Eigen::Vector3d& b){
        return a(2) > b(2);
    });
    map_chessboards_corner[pcd_path] = chessboard_corner;

    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addPointCloud<PointT>(cloud, "cloud");
    PointT corner5(center.x, center.y, center.z);
    PointT corner6(center.x + normal[0], center.y + normal[1], center.z + normal[2]);
    PointT corner7(center.x + eigen_vectors(0,0), center.y + eigen_vectors(1,0), center.z + eigen_vectors(2,0));//长轴
    PointT corner8(center.x + eigen_vectors(0,1), center.y + eigen_vectors(1,1), center.z + eigen_vectors(2,1));// 短轴
    viewer.addLine(corner5, corner6, 1.0, 0.0, 0.0, "line5");
    viewer.addLine(corner5, corner7, 255.0, 0.0, 0.0, "line6");
    viewer.addLine(corner5, corner8, 0.0, 255.0, 0.0, "line7");

    PointT corner11(new_point_1[0], new_point_1[1], new_point_1[2]);
    PointT corner12(new_point_2[0], new_point_2[1], new_point_2[2]);
    PointT corner13(new_point_3[0], new_point_3[1], new_point_3[2]);
    PointT corner14(new_point_4[0], new_point_4[1], new_point_4[2]);
    viewer.addLine(corner11, corner12, 1.0, 0.0, 0.0, "line9");
    viewer.addLine(corner12, corner13, 1.0, 0.0, 0.0, "line10");
    viewer.addLine(corner13, corner14, 1.0, 0.0, 0.0, "line11");
    viewer.addLine(corner14, corner11, 1.0, 0.0, 0.0, "line12");
    for(int i = 0; i < 4; ++i){
        viewer.addText3D(std::to_string(i+1), pcl::PointXYZ(chessboard_corner[i][0],chessboard_corner[i][1],chessboard_corner[i][2])
        , 0.1, 1.0, 1.0, 1.0, "label"+std::to_string(i+1));
    }
    lenght_1 = chessboard_size_w / 2;
    lenght_2 = chessboard_size_h / 2;
    Eigen::Vector3d new_point_5, new_point_6, new_point_7, new_point_8;
    gainCornerPoint(line_direction, lenght_1, vertical_direction, lenght_2, through_point, new_point_5);
    gainCornerPoint(line_direction, -lenght_1, vertical_direction, lenght_2, through_point, new_point_6);
    gainCornerPoint(line_direction, -lenght_1, vertical_direction, -lenght_2, through_point, new_point_7);
    gainCornerPoint(line_direction, lenght_1, vertical_direction, -lenght_2, through_point, new_point_8);
    PointT corner15(new_point_5[0], new_point_5[1], new_point_5[2]);
    PointT corner16(new_point_6[0], new_point_6[1], new_point_6[2]);
    PointT corner17(new_point_7[0], new_point_7[1], new_point_7[2]);
    PointT corner18(new_point_8[0], new_point_8[1], new_point_8[2]);
    viewer.addLine(corner15, corner16, 1.0, 0.0, 0.0, "line13");
    viewer.addLine(corner16, corner17, 1.0, 0.0, 0.0, "line14");
    viewer.addLine(corner17, corner18, 1.0, 0.0, 0.0, "line15");
    viewer.addLine(corner18, corner15, 1.0, 0.0, 0.0, "line16");
    
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

void ExtractPcdCornerPoint::process(std::map<std::string, std::vector<Eigen::Vector3d>>& map_chessboards_cerner_points)
{
    for(auto i : m_chessboard)
    {
        gainPointcloudFromCenter(i.first, i.second);
    }
    map_chessboards_cerner_points = map_chessboards_corner;
}

void ExtractPcdCornerPoint::boundaryEstimation(PointCloudT::Ptr& in_cloud, PointCloudT::Ptr& out_cloud){
    // updata....
}
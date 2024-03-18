/*
    Copyright (C) 2024 by Yang LiangHong Limited. All rights reserved.
    Yang LiangHong <2252512364@qq.com>
*/
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <json/json.h>

void sortFolds(std::string folder_path, std::string form, std::vector<std::string>& files_path_v)
{
    if (!boost::filesystem::exists(folder_path) || !boost::filesystem::is_directory(folder_path)) {
        std::cerr << "Invalid pcd folder path!" << std::endl;
    }
    for (const auto& entry : boost::filesystem::directory_iterator(folder_path)) {
        if (entry.path().extension() == form) {
            files_path_v.push_back(entry.path().string());
        }
    }
    sort(files_path_v.begin(), files_path_v.end());
}

void axisAngleToEuler(Eigen::Vector3d& axis_angle, Eigen::Vector3d& euler_angles, int order1, int order2, int order3)
{
    Eigen::AngleAxisd rotation_vector(axis_angle.norm(), axis_angle.normalized());
    Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();
    euler_angles = rotation_matrix.eulerAngles(order1, order2, order3); // 使用Z-Y-X的顺序210
}

void lidar2uv(std::vector<double> camera_params, std::vector<double> laser_pose, cv::Point3d world_point, cv::Point2d& point_uv)
{
        double fx = camera_params[0];
        double fy = camera_params[1];
        double cx = camera_params[2];
        double cy = camera_params[3];
        double k1 = camera_params[4];
        double k2 = camera_params[5];
        double p1 = camera_params[6];
        double p2 = camera_params[7];
        double wx = double(world_point.x);
        double wy = double(world_point.y);
        double wz = double(world_point.z);
        double camera_x = double(laser_pose[0]) * wx + double(laser_pose[1]) * wy + double(laser_pose[2]) * wz + double(laser_pose[3]);
        double camera_y = double(laser_pose[4]) * wx + double(laser_pose[5]) * wy + double(laser_pose[6]) * wz + double(laser_pose[7]);
        double camera_z = double(laser_pose[8]) * wx + double(laser_pose[9]) * wy + double(laser_pose[10]) * wz + double(laser_pose[11]);
        double u = camera_x / camera_z;
        double v = camera_y / camera_z;
        double r2 = u * u + v * v;
        double distortion = double(1.0) + r2 * (k1 + r2 * (k2 + r2 * double(1.0)));
        double u_distorted = u * distortion + double(2.0) * p1 * u * v + p2 * (r2 + double(2.0) * u * u);
        double v_distorted = v * distortion + p1 * (r2 + double(2.0) * v * v) + double(2.0) * p2 * u * v;
        double predicted_x = fx * u_distorted + cx;
        double predicted_y = fy * v_distorted + cy;
        point_uv.x = predicted_x;
        point_uv.y = predicted_y;
}

void calculateV(std::vector<std::vector<cv::Point3d>> lidar_bp_v, std::vector<cv::Point3d>& lidar_v)
{
    for(int i = 0; i < lidar_bp_v.size(); ++i)
    {
        Eigen::Vector3d l_0 = Eigen::Vector3d(lidar_bp_v[i][0].x, lidar_bp_v[i][0].y, lidar_bp_v[i][0].z);
        Eigen::Vector3d l_1 = Eigen::Vector3d(lidar_bp_v[i][1].x, lidar_bp_v[i][1].y, lidar_bp_v[i][1].z);
        Eigen::Vector3d l_2 = Eigen::Vector3d(lidar_bp_v[i][2].x, lidar_bp_v[i][2].y, lidar_bp_v[i][2].z);
        Eigen::Vector3d lv_1 = l_1 - l_0;
        Eigen::Vector3d lv_2 = l_2 - l_0;
        Eigen::Vector3d lv_n = lv_1.cross(lv_2);
        lidar_v.push_back(cv::Point3d(lv_n[0], lv_n[1], lv_n[2]));
    }
}

void loadParam(std::string filename, std::string key, std::vector<double>& params, int num) {
    params.clear();
    Json::Reader reader;
    Json::Value root;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "Error Opening " << filename << std::endl;
        return;
    }

    if (reader.parse(in, root, false)) {
        auto name = root.getMemberNames();
        for(auto id = name.begin(); id != name.end(); ++id)
        {
            if(*id == key){
                Json::Value data = root[*id];
                for(int i = 0; i < num; ++i)
                {
                    params.push_back(data[i].asDouble());
                }
            }
        }
    }
    in.close();
    return;
}

void matToRotV(std::vector<double> mat, std::vector<double>& ret)
{
        Eigen::Matrix4d ext;
        ext << mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7],
                mat[8], mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15];
        Eigen::Matrix3d rot = ext.block<3,3>(0, 0);
        Eigen::Vector3d t = ext.block<3,1>(0, 3);
        Eigen::AngleAxisd angle_axis(rot);
        Eigen::Vector3d rot_v = angle_axis.angle() * angle_axis.axis();
        if(!ret.empty()) ret.clear();
        ret.push_back(rot_v[0]);
        ret.push_back(rot_v[1]);
        ret.push_back(rot_v[2]);
        ret.push_back(t[0]);
        ret.push_back(t[1]);
        ret.push_back(t[2]);   
}

void lidarToCameraPoint(std::vector<double> rt, cv::Point3d p1, cv::Point3d& p2)
{
    p2.x = rt[0] * p1.x + rt[1] * p1.y + rt[2] * p1.z + rt[3];
    p2.y = rt[4] * p1.x + rt[5] * p1.y + rt[6] * p1.z + rt[7];
    p2.z = rt[8] * p1.x + rt[9] * p1.y + rt[10] * p1.z + rt[11];
}

//来源opencalib
cv::Scalar fakeColor(float value)
        {
            float posSlope = 255 / 60.0;
            float negSlope = -255 / 60.0;
            value *= 255;
            cv::Vec3f color;
            if (value < 60)
            {
            color[0] = 255;
            color[1] = posSlope * value + 0;
            color[2] = 0;
            }
            else if (value < 120)
            {
            color[0] = negSlope * value + 2 * 255;
            color[1] = 255;
            color[2] = 0;
            }
            else if (value < 180)
            {
            color[0] = 0;
            color[1] = 255;
            color[2] = posSlope * value - 2 * 255;
            }
            else if (value < 240)
            {
            color[0] = 0;
            color[1] = negSlope * value + 4 * 255;
            color[2] = 255;
            }
            else if (value < 300)
            {
            color[0] = posSlope * value - 4 * 255;
            color[1] = 0;
            color[2] = 255;
            }
            else
            {
            color[0] = 255;
            color[1] = 0;
            color[2] = negSlope * value + 6 * 255;
            }
            return cv::Scalar(color[0], color[1], color[2]);
        }

void pcdToImage(std::string img_path, std::string pcd_path, std::vector<double> mat,cv::Mat K, cv::Mat D)
{
    cv::Mat raw_img = cv::imread(img_path);
    cv::Mat undistort_img;
    cv::fisheye::undistortImage(raw_img, undistort_img, K, D, K);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *cloud);
    std::vector<cv::Point3d> p_s;
    cv::Point3d pc_p1, cam_p1;
    std::vector<float> ps_intensity;
    cv::Mat dis_coeffs = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
    for(size_t i = 0; i < cloud->size(); ++i)
    {
        pcl::PointXYZI point = cloud->points[i];
        pc_p1.x = point.x;
        pc_p1.y = point.y;
        pc_p1.z = point.z;
        ps_intensity.push_back(point.intensity);
        lidarToCameraPoint(mat, pc_p1, cam_p1);
        p_s.push_back(cam_p1);
    }
    std::vector<cv::Point2d> img_p_s;
    cv::projectPoints(p_s, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, dis_coeffs, img_p_s);
    cv::Scalar color;
    for(int i = 0; i < img_p_s.size(); ++i)
    {
        color = fakeColor(ps_intensity[i] / 25);
        cv::circle(undistort_img, cv::Point(int(img_p_s[i].x), int(img_p_s[i].y)), 1.5, color, -1);
    }
    cv::imshow("project pcd to img ", undistort_img);
    cv::waitKey(0);
    cv::imwrite("project_pcd_to_undist_img.png", undistort_img);
    return;
}
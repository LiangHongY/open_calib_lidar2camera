/*
    Copyright (C) 2024 by Yang LiangHong Limited. All rights reserved.
    Yang LiangHong <2252512364@qq.com>
*/
#include <iostream>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "extract_pc.hpp"
#include "extract_img.hpp"
#include "ceres_lidar2cam.hpp"
#include "common.hpp"

void processCalibLC(std::vector<cv::Point3d> lidar_p, std::vector<cv::Point3d> cam_p, 
std::vector<cv::Point2d> image_p, cv::Mat& rvec, cv::Mat& tvec, std::string json_path)
{
    double fx, fy, cx, cy, k1, k2, p1, p2;
    std::vector<double> initial_external_params;
    std::string key = "camera_matrix";
    std::vector<double> params;
    int param_num = 4;
    loadParam(json_path, key, params, param_num);
    fx = params[0];
    fy = params[1];
    cx = params[2];
    cy = params[3];
    key = "initial_external_params";
    param_num = 16;
    loadParam(json_path, key, params, param_num);
    initial_external_params = params;
    std::vector<double> rot_v;
    matToRotV(initial_external_params, rot_v);
    std::vector<std::vector<cv::Point3d>> lidar_bp_v, cam_bp_v;
    std::vector<cv::Point3d> lidar_bp, cam_bp;
    int num = 0;
    for(int i = 0; i < lidar_p.size(); ++i)
    {
        lidar_bp.push_back(lidar_p[i]);
        cam_bp.push_back(cam_p[i]);
        ++num;
        if(num == 4){
            num = 0;
            lidar_bp_v.push_back(lidar_bp);
            cam_bp_v.push_back(cam_bp);
            lidar_bp.clear();
            cam_bp.clear();
        }
    }
    std::vector<cv::Point3d> lidar_v, cam_v;
    calculateV(lidar_bp_v, lidar_v);
    calculateV(cam_bp_v, cam_v);
    k1 = 0.000000000; k2 = 0.000000000; p1 = 0.000000000; p2 = 0.000000000;
    double camera_params[8] = {fx, fy, cx, cy, k1, k2, p1, p2};
    double laser_pose[6] = {rot_v[0], rot_v[1], rot_v[2], rot_v[3], rot_v[4], rot_v[5]};
    ceres::Problem problem;    
    for (size_t i = 0; i < lidar_p.size(); ++i) {
        ceres::CostFunction* costFunction_1 = new ceres::AutoDiffCostFunction<LaserCameraErrorCamD, 3, 6>(
            new LaserCameraErrorCamD(lidar_p[i], cam_p[i])
        );
        problem.AddResidualBlock(costFunction_1, new ceres::CauchyLoss(0.05), laser_pose);
    }
    for (size_t i = 0; i < lidar_p.size(); ++i) {
        ceres::CostFunction* costFunction_2 = new ceres::AutoDiffCostFunction<LaserCameraErrorUv, 2, 8, 6>(
            new LaserCameraErrorUv(lidar_p[i], image_p[i])
        );
        problem.AddResidualBlock(costFunction_2, new ceres::CauchyLoss(0.05), camera_params, laser_pose);
    }
    for (size_t i = 0; i < lidar_v.size(); ++i) {
        ceres::CostFunction* costFunction_3 = new ceres::AutoDiffCostFunction<LaserCameraErrorCamNv, 3, 6>(
            new LaserCameraErrorCamNv(lidar_v[i], cam_v[i])
        );
        problem.AddResidualBlock(costFunction_3, new ceres::CauchyLoss(0.05), laser_pose);
    }

    int num_iterations = 100000;
    ceres::Solver::Options options;
    options.max_num_iterations = num_iterations;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    // options.callbacks.push_back(new OptimizationCallback(1e-12)); 
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    std::cout << " --- " << std::endl;
    std::cout << summary.BriefReport() << std::endl;
    std::cout << " --- " << std::endl;
    Eigen::Matrix<double, 4, 4> lidar2camer_mat;
    double rot[9];
    ceres::AngleAxisToRotationMatrix(laser_pose, rot);
    lidar2camer_mat << rot[0], rot[1], rot[2], laser_pose[3],
                        rot[3], rot[4], rot[5], laser_pose[4],
                        rot[6], rot[7], rot[8], laser_pose[5],
                        0, 0, 0, 1;
    std::cout << lidar2camer_mat << std::endl;
    std::cout << " -- " << std::endl;
    std::cout << lidar2camer_mat.inverse() << std::endl;
    rvec.at<double>(0, 0) = laser_pose[0];
    rvec.at<double>(0, 1) = laser_pose[1];
    rvec.at<double>(0, 2) = laser_pose[2];
    tvec.at<double>(0, 0) = laser_pose[3];
    tvec.at<double>(0, 1) = laser_pose[4];
    tvec.at<double>(0, 2) = laser_pose[5];
}

int main(int argc, char** argv) {
    
    std::string pcd_path = argv[1];
    std::string img_path = argv[2];
    std::string json_path = argv[3];
    double fx, fy, cx, cy, k1, k2, p1, p2, k3;
    int camera_type;
    std::vector<double> initial_external_params;
    std::string key = "camera_matrix";
    std::vector<double> params;
    int param_num = 4;
    loadParam(json_path, key, params, param_num);
    fx = params[0];
    fy = params[1];
    cx = params[2];
    cy = params[3];
    key = "camera_type";
    param_num = 1;
    loadParam(json_path, key, params, param_num);
    camera_type = int(params[0]);
    key = "distortion_coeffs";
    param_num = 4;
    loadParam(json_path, key, params, param_num);
    k1 = params[0];
    k2 = params[1];
    p1 = params[2];
    p2 = params[3];
    if(camera_type == 1) k3 = params[4];
    ExtractImgCornerPoint ei(json_path);
    std::map<std::string, std::vector<cv::Point3f>> map_chessboards_points_cam_xyz;
    std::map<std::string, std::vector<cv::Point2f>> map_chessboards_corner_points_img;
    ei.gainImgCornerPointWorldXyz(img_path, map_chessboards_points_cam_xyz);
    ei.gainImgCornerPointUv(img_path, map_chessboards_corner_points_img);
    std::map<std::string, std::vector<Eigen::Vector3d>> map_chessboards_corner_points;
    ExtractPcdCornerPoint ep(json_path);
    ep.gainChessboardCenter(pcd_path);
    ep.process(map_chessboards_corner_points);
    std::vector<std::string> pcd_path_v;
    std::vector<std::string> img_path_v;
    std::map<std::string, std::string> map_img_pcd_path;
    sortFolds(pcd_path, ".pcd", pcd_path_v);
    sortFolds(img_path, ".png", img_path_v);
    if(pcd_path_v.size() == img_path_v.size()){
        for(int i = 0; i < img_path_v.size(); ++i)
        {
            map_img_pcd_path[img_path_v[i]] = pcd_path_v[i];
        }
    }else{
        std::cout << " chech the nums of pcd / image ! " << std::endl;
        return 0;
    }

    std::vector<cv::Point3d> world_points;
    std::vector<cv::Point3d> camera_points;
    std::vector<cv::Point2d> image_points;
    cv::Point3d world_p;
    cv::Point3d cam_p;
    cv::Point2d img_p;
    cv::Mat rvec = (cv::Mat_<double>(1, 3) << 0, 0, 0);
    cv::Mat tvec = (cv::Mat_<double>(1, 3) << 0, 0, 0);
    for(auto i : map_chessboards_points_cam_xyz)
    {
        for(int j = 0; j < 4; ++j)
        {
            cam_p.x = i.second[j].x;
            cam_p.y = i.second[j].y;
            cam_p.z = i.second[j].z;
            camera_points.push_back(cam_p);
            world_p.x = map_chessboards_corner_points[map_img_pcd_path[i.first]][j][0];
            world_p.y = map_chessboards_corner_points[map_img_pcd_path[i.first]][j][1];
            world_p.z = map_chessboards_corner_points[map_img_pcd_path[i.first]][j][2];
            world_points.push_back(world_p);
        }
    }
    
    for(auto i : map_chessboards_corner_points_img){
        for(int j = 0; j < 4; ++j)
        {
            img_p.x = i.second[j].x;
            img_p.y = i.second[j].y;
            image_points.push_back(img_p);
        }
    }
    processCalibLC(world_points, camera_points, image_points, rvec, tvec, json_path);
    double rot[9];
    double laser_pose[6];
    laser_pose[0] = rvec.at<double>(0,0);
    laser_pose[1] = rvec.at<double>(0,1);
    laser_pose[2] = rvec.at<double>(0,2);
    laser_pose[3] = tvec.at<double>(0,0);
    laser_pose[4] = tvec.at<double>(0,1);
    laser_pose[5] = tvec.at<double>(0,2);
    ceres::AngleAxisToRotationMatrix(laser_pose, rot);
    std::vector<double> cam_ini;
    std::vector<double> exec;
    cam_ini.push_back(fx);
    cam_ini.push_back(fy);
    cam_ini.push_back(cx);
    cam_ini.push_back(cy);
    cam_ini.push_back(0.0);
    cam_ini.push_back(0.0);
    cam_ini.push_back(0.0);
    cam_ini.push_back(0.0);
    int num = 0;
    for(int i = 0, j = 0; i < 9; ++i)
    {
        exec.push_back(rot[i]);
        num++;
        if(num == 3){
            exec.push_back(laser_pose[j+3]);
            ++j;
            num = 0;
        }
    }
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distortion_coeffs;
    if(camera_type == 2){
        distortion_coeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
    }else{
        distortion_coeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2, k3);
    }
    cv::Mat dis_coeffs = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
    for(auto i : img_path_v){
        cv::Mat distorted_image = cv::imread(i);
        cv::Mat undistorted_image;
        if(camera_type == 2){
            cv::fisheye::undistortImage(distorted_image, undistorted_image, camera_matrix, distortion_coeffs, camera_matrix);
        }else{
            cv::undistort(distorted_image, undistorted_image, camera_matrix, distortion_coeffs, camera_matrix);
        }
        std::vector<cv::Point3d> points;
        cv::Point3d point, tmp_p;
        for(int j = 0; j < 4; ++j)
        {
            point.x = map_chessboards_corner_points[map_img_pcd_path[i]][j][0];
            point.y = map_chessboards_corner_points[map_img_pcd_path[i]][j][1];
            point.z = map_chessboards_corner_points[map_img_pcd_path[i]][j][2];
            lidarToCameraPoint(exec, point, tmp_p);
            points.push_back(tmp_p);
        }
        std::vector<cv::Point2d> images;
        cv::projectPoints(points, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera_matrix, dis_coeffs, images);//投影结果合理
        if(images.size() != 4){
            std::cout << " project fail ! " << std::endl;
            break;
        }
        for(int j = 0; j < images.size(); ++j)
        {
            std::string label = std::to_string(j+1);
            cv::circle(undistorted_image, cv::Point(int(images[j].x), int(images[j].y)), 3, cv::Scalar(0, 255, 0), -1);
            cv::putText(undistorted_image, label, cv::Point(int(images[j].x) - 10, int(images[j].y) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("Image with Point", undistorted_image);
        cv::waitKey(100);
    }
    pcdToImage(img_path_v[0], pcd_path_v[0], exec, camera_matrix, distortion_coeffs);
    return 0;
}



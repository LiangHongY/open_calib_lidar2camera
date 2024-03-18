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
#include <json/json.h>

class ExtractImgCornerPoint{
    protected:
        double fx, fy, cx, cy, k1, k2, p1, p2, k3;
        int chessboard_width, chessboard_height;
        double gap;
        int camera_type; // pinhold : 1 , fisheye 2

    public:
        ExtractImgCornerPoint(std::string file_path);
        void loadParam(std::string file_path);
        void gainImgCornerPointUv(std::string image_path, std::map<std::string, std::vector<cv::Point2f>>& map_chessboards_corner_points_img);
        void gainImgCornerPointWorldXyz(std::string image_path, std::map<std::string, std::vector<cv::Point3f>>& map_chessboards_points_world_xyz);

};
ExtractImgCornerPoint::ExtractImgCornerPoint(std::string file_path){
    loadParam(file_path);
}

void ExtractImgCornerPoint::loadParam(std::string file_path) {
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
        // std::cout << *id <<std::endl;
        if(*id == "chessboard_width"){
            // std::cout << *id << std::endl;
            chessboard_width = int(data[0].asDouble());
            continue;
        }
        else if(*id == "chessboard_height"){
            chessboard_height = int(data[0].asDouble());
            continue;
        }
        else if(*id == "chessboard_grap"){
            gap = data[0].asDouble();
            continue;
        }
        else if(*id == "camera_matrix"){
            fx = data[0].asDouble();
            fy = data[1].asDouble();
            cx = data[2].asDouble();
            cy = data[3].asDouble();
            continue;
        }
        else if(*id == "distortion_coeffs"){
            k1 = data[0].asDouble();
            k2 = data[1].asDouble();
            p1 = data[2].asDouble();
            p2 = data[3].asDouble();
            if(data[4].asDouble() != 0)
            {
                k3 = data[4].asDouble();
            }
            continue;
        }else if(*id == "camera_type")
        {
            camera_type = int(data[0].asDouble());
            continue;
        }
    }
  }
  in.close();
  return;
}

void ExtractImgCornerPoint::gainImgCornerPointUv(std::string image_path, std::map<std::string, std::vector<cv::Point2f>>& map_chessboards_corner_points_img)
{
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distortion_coeffs;
    if(camera_type == 2){
        distortion_coeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
    }else{
        distortion_coeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2, k3);
    }
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<cv::String> images;
    cv::glob(image_path, images);
    cv::Mat frame, gray;
    for (size_t i = 0; i < images.size(); i++) {
        frame = cv::imread(images[i]);
        cv::Mat undistorted_image;
        if(camera_type == 2){
            cv::fisheye::undistortImage(frame, undistorted_image, camera_matrix, distortion_coeffs, camera_matrix);
        }else{
            cv::undistort(frame, undistorted_image, camera_matrix, distortion_coeffs, camera_matrix);
        }
        cv::cvtColor(undistorted_image, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        std::vector<cv::Point2f> four_corners;
        bool pattern_found = cv::findChessboardCorners(gray, cv::Size(chessboard_width, chessboard_height), corners);
        if (pattern_found) {
            image_points.push_back(corners);
            four_corners.push_back(corners[0]);
            four_corners.push_back(corners[chessboard_width - 1]);
            four_corners.push_back(corners[chessboard_width * (chessboard_height -1)]);
            four_corners.push_back(corners[chessboard_width * chessboard_height -1]);
            std::sort(four_corners.begin(),four_corners.end(),[](cv::Point2f& a, cv::Point2f& b){
                return a.y < b.y;
            });
            map_chessboards_corner_points_img[images[i]] = four_corners;
        }
    }
}

void ExtractImgCornerPoint::gainImgCornerPointWorldXyz(std::string image_path, std::map<std::string, std::vector<cv::Point3f>>& map_chessboards_points_world_xyz)
{
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distortion_coeffs;
    if(camera_type == 2){
        distortion_coeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
    }else{
        distortion_coeffs = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2, k3);
    }
    cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 0.0);
    std::vector<cv::Point3f> obj;   
    for (int i = 0; i < chessboard_height; i++) {
        for (int j = 0; j < chessboard_width; j++) {
            obj.push_back(cv::Point3f(j*gap, i*gap, 0));
        }
    }
    std::vector<cv::Point3f> obj_w;
    std::vector<cv::Mat> rvec_v, tvec_v;
    std::vector<cv::String> images;
    cv::glob(image_path, images);
    cv::Mat frame, gray;
    for (size_t i = 0; i < images.size(); i++) {
        frame = cv::imread(images[i]);
        cv::Mat undistorted_image;
        if(camera_type == 2){
            cv::fisheye::undistortImage(frame, undistorted_image, camera_matrix, distortion_coeffs, camera_matrix);
        }else{
            cv::undistort(frame, undistorted_image, camera_matrix, distortion_coeffs, camera_matrix);
        }
        cv::cvtColor(undistorted_image, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        std::vector<cv::Point3f> four_corners;
        bool pattern_found = cv::findChessboardCorners(gray, cv::Size(chessboard_width, chessboard_height), corners);
        std::vector<cv::Point2d> points_2d;
        if (pattern_found) {
            cv::Mat rvec, tvec;
            cv::solvePnP(obj, corners, camera_matrix, distCoeffs, rvec, tvec);
            cv::Point3d point_3d(tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));
            std::vector<cv::Point3d> points_3d;
            points_3d.push_back(point_3d);
            rvec_v.push_back(rvec);
            tvec_v.push_back(tvec);
            for (int j = 0; j < obj.size(); j++)
            {
                const cv::Point3f &p = obj[j];
                cv::Vec3d pt_cam(p.x, p.y, p.z);
                cv::Vec3d rvec_(rvec.at<double>(0, 0), rvec.at<double>(0, 1), rvec.at<double>(0, 2));
                cv::Vec3d tvec_(tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));
                pt_cam = rvec_ * pt_cam + tvec_;
                cv::Point3f tmp;
                tmp.x = pt_cam[0];
                tmp.y = pt_cam[1];
                tmp.z = pt_cam[2];
                obj_w.push_back(tmp);
            }
            four_corners.push_back(obj_w[0]);
            four_corners.push_back(obj_w[chessboard_width - 1]);
            four_corners.push_back(obj_w[chessboard_width * (chessboard_height -1)]);
            four_corners.push_back(obj_w[chessboard_width * chessboard_height -1]);
            std::sort(four_corners.begin(),four_corners.end(),[](cv::Point3f& a, cv::Point3f& b){
                return a.y < b.y;
            });
            map_chessboards_points_world_xyz[images[i]] = four_corners;
            obj_w.clear();
        }
    }
}


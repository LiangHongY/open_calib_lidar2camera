/*
    Copyright (C) 2024 by Yang LiangHong Limited. All rights reserved.
    Yang LiangHong <2252512364@qq.com>
*/
#include <iostream>
#include <vector>
#include <map>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct LaserCameraErrorUv {
    const cv::Point3d& world_point;
    const cv::Point2d& image_point;

    LaserCameraErrorUv(const cv::Point3d& world_point, const cv::Point2d& image_point) 
        : world_point(world_point), image_point(image_point) {}
    template <typename T>
    bool operator()(const T* const camera_params,
                    const T* const laser_pose,
                    T* residuals) const {
        T rot[9];
        T x[6] = { T(laser_pose[0]), T(laser_pose[1]), T(laser_pose[2]), T(laser_pose[3]), T(laser_pose[4]), T(laser_pose[5]) };
        ceres::AngleAxisToRotationMatrix(x, rot);
        T fx = camera_params[0];
        T fy = camera_params[1];
        T cx = camera_params[2];
        T cy = camera_params[3];
        T k1 = camera_params[4];
        T k2 = camera_params[5];
        T p1 = camera_params[6];
        T p2 = camera_params[7];
        T wx = T(world_point.x);
        T wy = T(world_point.y);
        T wz = T(world_point.z);
        T camera_x = T(rot[0]) * wx + T(rot[1]) * wy + T(rot[2]) * wz + T(x[3]);
        T camera_y = T(rot[3]) * wx + T(rot[4]) * wy + T(rot[5]) * wz + T(x[4]);
        T camera_z = T(rot[6]) * wx + T(rot[7]) * wy + T(rot[8]) * wz + T(x[5]);
        T u = camera_x / camera_z;
        T v = camera_y / camera_z;
        T r2 = u * u + v * v;
        T distortion = T(1.0) + r2 * (k1 + r2 * (k2 + r2 * T(1.0)));
        T u_distorted = u * distortion + T(2.0) * p1 * u * v + p2 * (r2 + T(2.0) * u * u);
        T v_distorted = v * distortion + p1 * (r2 + T(2.0) * v * v) + T(2.0) * p2 * u * v;
        T predicted_x = fx * u_distorted + cx;
        T predicted_y = fy * v_distorted + cy;
        residuals[0] = predicted_x - T(image_point.x);
        residuals[1] = predicted_y - T(image_point.y);
        return true;
    }
};

struct LaserCameraErrorCamD {
    const cv::Point3d& world_point;
    const cv::Point3d& camera_point;
    LaserCameraErrorCamD(const cv::Point3d& world_point_, const cv::Point3d& camera_point_) 
        : world_point(world_point_), camera_point(camera_point_) {}
    template <typename T>
    bool operator()(const T* const laser_pose,
                    T* residuals) const {
        T rot[9];
        T x[6] = { T(laser_pose[0]), T(laser_pose[1]), T(laser_pose[2]), T(laser_pose[3]), T(laser_pose[4]), T(laser_pose[5]) };
        ceres::AngleAxisToRotationMatrix(x, rot);
        T wx = T(world_point.x);
        T wy = T(world_point.y);
        T wz = T(world_point.z);
        T camera_x = T(rot[0]) * wx + T(rot[1]) * wy + T(rot[2]) * wz + T(x[3]);
        T camera_y = T(rot[3]) * wx + T(rot[4]) * wy + T(rot[5]) * wz + T(x[4]);
        T camera_z = T(rot[6]) * wx + T(rot[7]) * wy + T(rot[8]) * wz + T(x[5]);
        residuals[0] = camera_x - T(camera_point.x);
        residuals[1] = camera_y - T(camera_point.y);
        residuals[2] = camera_z - T(camera_point.z);

        return true;
    }
};

struct LaserCameraErrorCamNv {
    const cv::Point3d& world_point;
    const cv::Point3d& camera_point;
    LaserCameraErrorCamNv(const cv::Point3d& world_point_, const cv::Point3d& camera_point_) 
        : world_point(world_point_), camera_point(camera_point_) {}
    template <typename T>
    bool operator()(const T* const laser_pose,
                    T* residuals) const {
        T rot[9];
        T x[6] = { T(laser_pose[0]), T(laser_pose[1]), T(laser_pose[2]), T(laser_pose[3]), T(laser_pose[4]), T(laser_pose[5]) };
        ceres::AngleAxisToRotationMatrix(x, rot);
        T wx = T(world_point.x);
        T wy = T(world_point.y);
        T wz = T(world_point.z);
        T camera_x = T(rot[0]) * wx + T(rot[1]) * wy + T(rot[2]) * wz;
        T camera_y = T(rot[3]) * wx + T(rot[4]) * wy + T(rot[5]) * wz;
        T camera_z = T(rot[6]) * wx + T(rot[7]) * wy + T(rot[8]) * wz;
        T norm = sqrt(camera_x * camera_x + camera_y * camera_y + camera_z * camera_z);
        camera_x /= norm;
        camera_y /= norm;
        camera_z /= norm;
        T cx = T(camera_point.x);
        T cy = T(camera_point.y);
        T cz = T(camera_point.z);
        T norm_ = sqrt(cx * cx + cy * cy + cz * cz);
        cx /= norm_;
        cy /= norm_;
        cz /= norm_;
        residuals[0] = camera_x - cx;
        residuals[1] = camera_y - cy;
        residuals[2] = camera_z - cz;

        return true;
    }
};

// Callback class to monitor optimization process
class OptimizationCallback : public ceres::IterationCallback {
public:
    OptimizationCallback(double residual_threshold) : residual_threshold_(residual_threshold) {}

    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
        std::cout << "Iteration " << summary.iteration << ", residual: " << summary.cost << std::endl;
        if (summary.cost <= residual_threshold_) {
            std::cout << "Residual reached threshold. Stopping optimization." << std::endl;
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    double residual_threshold_;
};


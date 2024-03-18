基于棋盘格的多线激光雷达和鱼眼/针孔模型相机外参标定的程序

前言

标定数据，只需要一个棋盘格标定板。把标定板放置lidar 与camera 共视区域，拜拍几个pose进行采集。

基于简谐原则而编写，不加骚操作，方便大家借鉴学习。觉得不错，点个小星星。代码框架在2024年春节在家完成，多谢bb放的几天假。后面有时间，会继续优化精度。

一、使用注意事项

0、两个文件夹数据，注意序号对应上

1、假设lidar z轴朝上，图片的uv分别向右向下。目的对齐两者的匹配点

2、选择每帧pcd的标定板中心时，需要注意shift+鼠标左键

3、依赖&编译

  pcl、opencv、Ceres、json
  mkdir build
  cd build 
  cmake .. && make

4、运行

  ./bin/calib_lidar2camera ./data/lidar ./data/camera ./config/calib_chessboard.json

5、关于配置文件说明

  {
      "camera_type": [2],
      "chessboard_size": [0.8, 0.6],
      "chessboard_width": [11],
      "chessboard_height": [8],
      "chessboard_grap": [0.06],
      "camera_matrix": [1720.5573642902884, 1725.4310071228047, 1953.549178143742, 1061.1840108704785],
      "distortion_coeffs": [0.08092536120554537, -0.049501707005857196, 0.020142406202130876, -0.0067493332722243764],
      "initial_external_params": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,1],
      "extract_radius": [0.5]
  }

camera_type ：              1 表示针孔相机（fov < 90°）  2 表示鱼眼相机（fov > 90°） [当前版本没有适配全景相机]
chessboard_size ：          表示标定板的物理尺寸，注意不是棋盘格轮廓尺寸，单位 米。填写方式，opencv适配棋盘格时角点宽高方式一致
chessboard_width：          表示棋盘格宽方向11个角点
chessboard_height：         表示棋盘格高方向8个角点
chessboard_grap ：          每个角点之间0.06米
camera_matrix：             相机内参
distortion_coeffs：         相机畸变系数
initial_external_params：   迭代优化给的初始化外参
extract_radius：            提取标定板点云的半径

测试说明：
打了马赛克图片，迭代效果不理想；估计处理过的图片，角点提取有影响
  
数据集外参投影效果图：
![Image text](https://github.com/LiangHongY/calib_lidar2camera/blob/master/data/1.png)

二、可优化项

1、标定板点云提取精度

2、图片角点提取精度

3、图片角点pnp恢复尺度信息算法精度

4、采集数据时，标定板的稳定性与环境光线情况[标定板材质和稳定性对标定结果影响较大]

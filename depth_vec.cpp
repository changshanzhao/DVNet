#include <limits>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/opencv.hpp>

/**
 * 对齐对象模板集合到一个示例点云
 * 调用命令格式 ./template_alignment2 ./data/object_templates.txt ./data/person.pcd
 *                  程序                          多个模板的文本文件         目标点云
 * 调用命令格式 ./template_alignment2 ./data/object_templates2.txt ./data/target.pcd
 *                  程序                          多个模板的文本文件         目标点云
 *
 * 实时的拍照得到RGB和深度图
 * 合成目标点云图
 * 通过直通滤波框定范围（得到感兴趣区域）
 * 将感兴趣区域进行降采样（提高模板匹配效率）
 */
int main(int argc, char **argv) {
    /*
    if (argc < 3) {
        printf("No target PCD file given!\n");
        return (-1);
    }
     */

    std::string pcd_filename;
    /*
    while (input_stream.good()) {
        // 按行读取模板中的文件名
        std::getline(input_stream, pcd_filename);
        if (pcd_filename.empty() || pcd_filename.at(0) == '#') // Skip blank lines or comments
            continue;

        // 加载特征云
        FeatureCloud template_cloud;
        template_cloud.loadInputCloud(pcd_filename);
        object_templates.push_back(template_cloud);
    }
    input_stream.close();
    */
    // Load the target cloud PCD file
//    FeatureCloud template_cloud;
//    template_cloud.loadInputCloud("./tem/ok.pcd");
//    object_templates.push_back(template_cloud);
//    template_cloud.loadInputCloud("./tem/ng1.pcd");
//    object_templates.push_back(template_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<std::string> list_name{"1", "2", "3","4",\
   "5", "6", "7", "8","9","10",\
   "11", "12","13","14","15","16"};
    std::vector<std::string> list_name2{"105090+","105090-","303904+","303904-",\
    "303906-","303906+","303908+","303908-","303910+","303910-","303922+"\
    ,"303922-","303925+","303925-","303926+","303926-","303930+","303930-"};





    for (int i = 0;i<list_name.size();i++){
        pcl::io::loadPCDFile("./data/NG/"+list_name[i]+".pcd", *cloud);

        //平面上的点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inner(new pcl::PointCloud<pcl::PointXYZ>);
        //平面外的点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outer(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        //创建分割对象
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        //可选设置
        seg.setOptimizeCoefficients(true);
        //必须设置
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.2);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        //判断是否分割成功
        if (inliers->indices.size() == 0)
        {
            PCL_ERROR("Could not estimate a planar model for the given dataset.");
            return (-1);
        }
        std::cerr << std::endl << "Model coefficients: " << coefficients->values[0] << " "
                  << coefficients->values[1] << " "
                  << coefficients->values[2] << " "
                  << coefficients->values[3] << std::endl << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudAbovePlane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudBelowPlane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_inner);
        extract.setNegative(true);
        extract.filter(*cloud_outer);
        cloudAbovePlane->width = cloud_outer->points.size();
        cloudAbovePlane->height = 1;
        cloudAbovePlane->points.resize(cloudAbovePlane->width * cloudAbovePlane->height);
        cloudBelowPlane->width = cloud_outer->points.size();
        cloudBelowPlane->height = 1;
        cloudBelowPlane->points.resize(cloudBelowPlane->width * cloudBelowPlane->height);
        for (size_t i=0, j=0, k=0 ; i < cloud_outer->points.size(); ++i)
        {
            //遍历
            if (coefficients->values[0] * (*cloud_outer).points[i].x + coefficients->values[1] * (*cloud_outer).points[i].y +
                coefficients->values[2] * (*cloud_outer).points[i].z + coefficients->values[3] > 0)
            {
                cloudAbovePlane->points[j].x = (*cloud_outer).points[i].x;
                cloudAbovePlane->points[j].y = (*cloud_outer).points[i].y;
                cloudAbovePlane->points[j].z = (*cloud_outer).points[i].z;
                ++j;
            }
            else
            {
                cloudBelowPlane->points[k].x = (*cloud_outer).points[i].x;
                cloudBelowPlane->points[k].y = (*cloud_outer).points[i].y;
                cloudBelowPlane->points[k].z = (*cloud_outer).points[i].z;
                ++k;
            }
        }




        // ... and downsampling the point cloud 降采样点云, 减少计算量
        // 定义体素大小 5mm
        const float voxel_grid_size = 0.05f;
        pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
        vox_grid.setInputCloud(cloudBelowPlane);
        // 设置叶子节点的大小lx, ly, lz
        vox_grid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
        //vox_grid.filter (*cloud); // Please see this http://www.pcl-developers.org/Possible-problem-in-new-VoxelGrid-implementation-from-PCL-1-5-0-td5490361.html
        pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
        vox_grid.filter(*tempCloud);
        cloudBelowPlane = tempCloud;
        std::cerr <<cloudBelowPlane->points.size()<<endl;
        typedef pcl::PointXYZ PointType;
        typedef pcl::Normal NormalType;
        typedef pcl::PointCloud<PointType> PointCloud;
        typedef pcl::PointCloud<NormalType> NormalCloud;
        pcl::NormalEstimation<PointType, NormalType> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        ne.setInputCloud(cloudBelowPlane);
        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.03);  // 设置搜索半径，用于估计每个点的法向量
        ne.compute (*cloud_normals);



        // 确定深度图的尺寸
        int depth_image_width = 1280;  // 深度图像宽度
        int depth_image_height = 640;

        // 创建深度图
        cv::Mat depth_image(depth_image_height, depth_image_width, CV_32FC4, cv::Scalar(0));

        // 填充深度图
        for (size_t i = 0; i < cloudBelowPlane->points.size(); ++i)
        {
            // 获取点的深度值
            float depth = cloudBelowPlane->points[i].z+10;

            // 将深度值映射到深度图的像素灰度值上
            unsigned short depth_value = static_cast<unsigned short>(depth * 10) < 255 ? static_cast<unsigned short>(depth * 10) : 254; // 将深度值乘以1000转换为毫米单位

            // 将深度值写入深度图像素
            int u = static_cast<int>((cloudBelowPlane->points[i].x+35)*10) < 640 ? static_cast<int>((cloudBelowPlane->points[i].x+35)*10) : 639;
            int v = static_cast<int>((cloudBelowPlane->points[i].y+70)*10) < 1280 ? static_cast<int>((cloudBelowPlane->points[i].y+70)*10) : 1279;
            if (depth_image.at<cv::Vec4f>(u, v)[0] < depth_value){
                depth_image.at<cv::Vec4f>(u, v)[0] = depth_value;
                const pcl::Normal& normal = cloud_normals->at(i);
                depth_image.at<cv::Vec4f>(u, v)[1] = normal.normal_x;
                depth_image.at<cv::Vec4f>(u, v)[2] = normal.normal_y;
                depth_image.at<cv::Vec4f>(u, v)[3] = normal.normal_z;
            }
        }


        // 保存深度图为PNG格式
        cv::imwrite("process/ng"+list_name[i]+".png", depth_image);

    }
    // Assign to the target FeatureCloud 对齐到目标特征点云
//    FeatureCloud target_cloud;
//    target_cloud.setInputCloud(cloudBelowPlane);

    // Set the TemplateAlignment inputs
//    TemplateAlignment template_align;
//    for (size_t i = 0; i < object_templates.size(); i++) {
//        FeatureCloud &object_template = object_templates[i];
//        // 添加模板点云
//        template_align.addTemplateCloud(object_template);
//    }
//    // 设置目标点云
//    template_align.setTargetCloud(target_cloud);
//
//
//    std::cout << "findBestAlignment" << std::endl;
//    // Find the best template alignment
//    // 核心代码
//    TemplateAlignment::Result best_alignment;
//    int best_index = template_align.findBestAlignment(best_alignment);
//    const FeatureCloud &best_template = object_templates[best_index];
//
//    // Print the alignment fitness score (values less than 0.00002 are good)
//    printf("Best fitness score: %f\n", best_alignment.fitness_score);
//    printf("Best fitness best_index: %d\n", best_index);
//
//    // Print the rotation matrix and translation vector
//    Eigen::Matrix3f rotation = best_alignment.final_transformation.block<3, 3>(0, 0);
//    Eigen::Vector3f translation = best_alignment.final_transformation.block<3, 1>(0, 3);
//
//    Eigen::Vector3f euler_angles = rotation.eulerAngles(2, 1, 0) * 180 / M_PI;
//
//    printf("\n");
//    printf("    | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
//    printf("R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
//    printf("    | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
//    printf("\n");
//    cout << "yaw(z) pitch(y) roll(x) = " << euler_angles.transpose() << endl;
//    printf("\n");
//    printf("t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
//
//    // Save the aligned template for visualization
//    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
//    // 将模板中保存的点云图进行旋转矩阵变换，把变换结果保存到transformed_cloud
//    pcl::transformPointCloud(*best_template.getPointCloud(), transformed_cloud, best_alignment.final_transformation);
//
////    pcl::io::savePCDFileBinary("output.pcd", transformed_cloud);
//    // =============================================================================
//
//    pcl::visualization::PCLVisualizer viewer("example");
//    // 设置坐标系系统
//    viewer.addCoordinateSystem(0.5, "cloud", 0);
//    // 设置背景色
//    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
//
//    // 1. 旋转后的点云rotated --------------------------------
//    pcl::PointCloud<pcl::PointXYZ>::Ptr t_cloud(&transformed_cloud);
//    PCLHandler transformed_cloud_handler(t_cloud, 255, 255, 255);
//    viewer.addPointCloud(t_cloud, transformed_cloud_handler, "transformed_cloud");
//    // 设置渲染属性（点大小）
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
//
//    // 2. 目标点云target --------------------------------
//    PCLHandler target_cloud_handler(cloudBelowPlane, 255, 100, 100);
//    viewer.addPointCloud(cloudBelowPlane, target_cloud_handler, "target_cloud");
//    // 设置渲染属性（点大小）
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud");
//
//    // 3. 模板点云template --------------------------------
//    PCLHandler template_cloud_handler(cloudBelowPlane, 100, 255, 255);
//    viewer.addPointCloud(best_template.getPointCloud(), template_cloud_handler, "template_cloud");
//    // 设置渲染属性（点大小）
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "template_cloud");
//
//    while (!viewer.wasStopped()) { // Display the visualiser until 'q' key is pressed
//        viewer.spinOnce();
//    }

    return (0);
}

/*
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include<pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include<vector>
int main(int argc, char** argv)
{
    //原始点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("./data/NG/16.pcd", *cloud);
    //平面上的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_inner(new pcl::PointCloud<pcl::PointXYZ>);
    //平面外的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outer(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //创建分割对象
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    //可选设置
    seg.setOptimizeCoefficients(true);
    //必须设置
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    //判断是否分割成功
    if (inliers->indices.size() == 0)
    {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return (-1);
    }
    std::cerr << std::endl << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudAbovePlane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudBelowPlane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_inner);
    extract.setNegative(true);
    extract.filter(*cloud_outer);
    cloudAbovePlane->width = cloud_outer->points.size();
    cloudAbovePlane->height = 1;
    cloudAbovePlane->points.resize(cloudAbovePlane->width * cloudAbovePlane->height);
    cloudBelowPlane->width = cloud_outer->points.size();
    cloudBelowPlane->height = 1;
    cloudBelowPlane->points.resize(cloudBelowPlane->width * cloudBelowPlane->height);
    for (size_t i=0, j=0, k=0 ; i < cloud_outer->points.size(); ++i)
    {
        //遍历
        if (coefficients->values[0] * (*cloud_outer).points[i].x + coefficients->values[1] * (*cloud_outer).points[i].y +
             coefficients->values[2] * (*cloud_outer).points[i].z + coefficients->values[3] > 0)
        {
            cloudAbovePlane->points[j].x = (*cloud_outer).points[i].x;
            cloudAbovePlane->points[j].y = (*cloud_outer).points[i].y;
            cloudAbovePlane->points[j].z = (*cloud_outer).points[i].z;
            ++j;
        }
        else
        {
            cloudBelowPlane->points[k].x = (*cloud_outer).points[i].x;
            cloudBelowPlane->points[k].y = (*cloud_outer).points[i].y;
            cloudBelowPlane->points[k].z = (*cloud_outer).points[i].z;
            ++k;
        }
    }
    std::cerr <<cloudBelowPlane->points.size()<<endl;

    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_above(cloudAbovePlane, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_below(cloudBelowPlane, 0, 0, 255);
    viewer.addPointCloud<pcl::PointXYZ>(cloudAbovePlane, cloud_above, "cloudAbovePlane");
    viewer.addPointCloud<pcl::PointXYZ>(cloudBelowPlane, cloud_below, "cloudBelowPlane");
    pcl::io::savePCDFileASCII("./NG/16.pcd", *cloudBelowPlane);

    viewer.spin();
    return (0);
}
 */
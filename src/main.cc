/*
 * @Author: WJM
 * @Date: 2023-03-02 21:10:50
 * @LastEditors: WJM
 * @LastEditTime: 2024-03-05 17:38:22
 * @Description:
 * @FilePath: /rknn_yolox_detector_fp16/src/main.cc
 * @custom_string: http://www.aiar.xjtu.edu.cn/
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <vector>
#include <queue>
#include <thread>

#include "RgaUtils.h"
#include "im2d.h"
#include <opencv2/opencv.hpp>
#include "idetector.h"
int main(int argc, char **argv)
{
    // 线程池操作
    std::vector<bbox_t> boxs;
    std::queue<cv::Mat> frameQueue;

    // 预定义一个包含12种醒目颜色的列表
    const std::vector<cv::Scalar> predefinedColors = {
        cv::Scalar(255, 0, 0),   // 红色
        cv::Scalar(0, 255, 0),   // 绿色
        cv::Scalar(0, 0, 255),   // 蓝色
        cv::Scalar(255, 255, 0), // 黄色
        cv::Scalar(255, 0, 255), // 粉色
        cv::Scalar(0, 255, 255), // 青色
        cv::Scalar(255, 127, 0), // 橙色
        cv::Scalar(127, 0, 255), // 紫色
        cv::Scalar(0, 127, 255), // 天蓝色
        cv::Scalar(127, 255, 0), // 酸橙色
        cv::Scalar(255, 0, 127), // 玫瑰红
        cv::Scalar(0, 255, 127), // 春绿色
        // ... 或许还可以添加更多的颜色，如果类别有增加
    };

    // char model_path[256] = "/home/rpdzkj/code/yolov8_rknn_Cplusplus/examples/rknn_yolov8_demo_open/model/RK3588/yolov8n_ZQ.rknn";
    char model_path[256] = "/home/rpdzkj/wjm/pinlingv2.3.1/pinlingv2.3/rknn_yolox_detector_fp16/model/yolox_RK3588_fp.rknn";
    char save_image_path[256] = "/home/rpdzkj/test_result.jpg";
    char image_path[256] = "/home/rpdzkj/code/obj_detect_multhread/examples/rknn_yolov8_demo_open/test.jpg";

    cv::VideoWriter writer;
    int codec = cv::VideoWriter::fourcc('M','P','E','G');         // 选择合适的编解码器
    writer.open("output.avi", codec, 30, cv::Size(1280, 720), true); // 假设我们想保存彩色视频

    if (!writer.isOpened())
    {
        std::cerr << "ERROR: Can't initialize video writer" << std::endl;
        return 1;
    }

    cv::Mat img_1 = cv::Mat::zeros(1280, 720, CV_8UC3);
    cv::Mat frame;
    // 打开视频文件
    cv::VideoCapture cap("/home/rpdzkj/video/01_1.mp4");
    idetector *idet = new idetector(model_path, 3, 12, 0.45, 0.2);
    idet->init();
    // 检查视频是否成功打开
    if (!cap.isOpened())
    {
        std::cout << "无法打开视频文件" << std::endl;
        return -1;
    }
    int frame_count = 0;


    while (cap.read(frame)&&frame_count<400)
    // int i=0;
    // while(i<100)
    {
        if (frame.empty()){
            printf("input img empty, quit\n");
            continue;
        }
        frameQueue.push(frame);
        idet->process(frameQueue.back(),boxs);

        if(frameQueue.size()<4){
            continue;
        }
        if (boxs.size() > 0)
        {
            for (auto &box : boxs)
            {
                // 假设box对象有一个obj_id成员
                size_t colorIndex = box.obj_id % predefinedColors.size();
                cv::Scalar color = predefinedColors[colorIndex];

                cv::rectangle(frameQueue.front(), cv::Point(box.x, box.y), cv::Point(box.x + box.w, box.y + box.h), color, 2, 8);

                // 在框的左上角添加obj_id文本
                std::string id_text = std::to_string(box.obj_id);
                cv::putText(frameQueue.front(), id_text, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
            }
        }
        // cv::imwrite("img/"+std::to_string(frame_count)+".jpg",frameQueue.front());
        frame_count++;
        writer.write(frameQueue.front());
    }
    delete idet;
    // writer.release();
    std::cout << "main exit" << std::endl;

    cap.release();
    std::cout << "main exit" << std::endl;
    return 0;
}

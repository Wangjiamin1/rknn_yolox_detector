/*
 * @Author: WJM
 * @Date: 2024-03-04 17:33:23
 * @LastEditors: WJM
 * @LastEditTime: 2024-03-08 15:21:04
 * @Description:
 * @FilePath: /pinlingv2.3.1/pinlingv2.3/rknn_yolox_detector_fp16/include/model_inference.h
 * @custom_string: http://www.aiar.xjtu.edu.cn/
 */
#ifndef __IDETECTOR_H__
#define __IDETECTOR_H__

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
// #include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"
#include "common.h"
#include <dirent.h>

inline double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

class modelInference
{
public:
    modelInference(char *model_path_, int c_model_input_img_size, int thread_index_, int OBJ_CLASS_NUM, float NMS_THRESHOLD, float CONF_THRESHOLD);
    virtual ~modelInference();
    int init();
    int detect();

    cv::Mat preprocess(const cv::Mat originalImage, int &startX, int &startY, float &ratio);

    int process_fp32(float *input, int grid_h, int grid_w, int height, int width, int stride,
                     std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold);

    int post_process(rknn_output *outputs, float ratio, int startX, int startY);

    int process_i8(int8_t *input, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale);

    int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
            int filterId, float threshold);
    cv::Mat c_img;
    std::vector<bbox_t> c_boxs;
    // bool run_flag;

private:
    char *model_path;
    unsigned char *model_data;
    int c_img_width;
    int c_img_height;
    int c_img_channel;
    std::vector<float> DetectiontRects;
    MODEL_INFO model_info;
    int c_thread_index;
    int c_obj_class_num;
    float c_nms_threshold;
    float c_conf_threshold;
    int c_model_input_img_size;
    object_detect_result_list c_detect_result_group;

    void *c_resize_buf = nullptr;
    float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1);
    int quick_sort_indice_inverse(
        std::vector<float> &input,
        int left,
        int right,
        std::vector<int> &indices);
    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);

    int read_data_from_file(const char *path, char **out_data);
    int clamp(float val, int min, int max);

    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);
    int32_t __clip(float val, float min, float max);
};

#endif
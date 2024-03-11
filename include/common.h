/*
 * @Author: WJM
 * @Date: 2023-03-03 02:04:26
 * @LastEditors: WJM
 * @LastEditTime: 2023-03-03 02:47:17
 * @Description: 
 * @FilePath: /rknn_yolox_detector_fp16/include/common.h
 * @custom_string: http://www.aiar.xjtu.edu.cn/
 */
#ifndef __COMMON_H__
#define __COMMON_H__


#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 200
#define MAX_OUTPUTS 3


typedef enum {
    IMAGE_FORMAT_GRAY8,
    IMAGE_FORMAT_RGB888,
    IMAGE_FORMAT_RGBA8888,
    IMAGE_FORMAT_YUV420SP_NV21,
    IMAGE_FORMAT_YUV420SP_NV12,
} image_format_t;

typedef struct {
    int width;
    int height;
    int width_stride;
    int height_stride;
    image_format_t format;
    unsigned char* virt_addr;
    int size;
    int fd;
} image_buffer_t;

struct bbox_t
{
    unsigned int x, y, w, h;     // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                  // confidence - probability that the object was found correctly
    unsigned int obj_id;         // class of object - from range [0, classes-1]
    unsigned int track_id;       // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter; // counter of frames on which the object was detected
    float x_3d, y_3d, z_3d;      // center of object (in Meters) if ZED 3D Camera is used
    bbox_t(unsigned int xx, unsigned int yy, unsigned int ww, unsigned int hh, unsigned int cls, unsigned int id, float conf) : x(xx), y(yy), w(ww), h(hh), obj_id(cls), track_id(id), prob(conf){};
    bbox_t() {}
};

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

typedef struct {
    int id;
    int count;
    bbox_t results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef enum
{
    NORMAL_API = 0,
    ZERO_COPY_API,
} API_TYPE;


typedef struct _RKDEMO_INPUT_PARAM
{
    uint8_t pass_through;
    rknn_tensor_format layout_fmt;
    rknn_tensor_type dtype;

    API_TYPE api_type;
    bool enable = false;
    bool _already_init = false;
} RKDEMO_INPUT_PARAM;

typedef struct _RKDEMO_OUTPUT_PARAM
{
    uint8_t want_float;

    API_TYPE api_type;
    bool enable = false;
    bool _already_init = false;
} RKDEMO_OUTPUT_PARAM;

struct MODEL_INFO
{

    rknn_context ctx;
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    bool is_quant;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_channel;
    int model_width;
    int model_height;

};

#endif
/*
 * @Author: WJM
 * @Date: 2024-03-04 17:33:24
 * @LastEditors: WJM
 * @LastEditTime: 2024-03-05 16:42:03
 * @Description: 
 * @FilePath: /rknn_yolox_detector_fp16/src/model_inference.cpp
 * @custom_string: http://www.aiar.xjtu.edu.cn/
 */

#include "model_inference.h"
#include <chrono>
#include <thread>

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

cv::Mat modelInference::preprocess(const cv::Mat originalImage, int &startX, int &startY, float &ratio)
{
    int originalWidth = originalImage.cols;
    int originalHeight = originalImage.rows;
    // 创建一个新的 640x640 大小的黑色图像
    cv::Mat resizedImage(c_model_input_img_size, c_model_input_img_size, CV_8UC3, cv::Scalar(0, 0, 0));

    // 计算调整大小后的图像的宽度和高度
    int resizedWidth, resizedHeight;
    if (originalWidth > originalHeight)
    {
        resizedWidth = c_model_input_img_size;
        ratio =  static_cast<float>(originalWidth) / static_cast<float>(c_model_input_img_size);
        resizedHeight = originalHeight * c_model_input_img_size / static_cast<float>(originalWidth);
    }
    else
    {
        resizedHeight = c_model_input_img_size;
         ratio =  static_cast<float>(originalHeight) / static_cast<float>(c_model_input_img_size);
        resizedWidth = originalWidth * c_model_input_img_size / static_cast<float>(originalHeight);;
    }
    // 计算调整大小后图像的起始坐标
    startX = (c_model_input_img_size - resizedWidth) / 2;
    startY = (c_model_input_img_size - resizedHeight) / 2;

    // 调整大小并将原始图像复制到新图像中
    cv::resize(originalImage, resizedImage(cv::Rect(startX, startY, resizedWidth, resizedHeight)), cv::Size(resizedWidth, resizedHeight));
    return resizedImage;
}

int modelInference::quick_sort_indice_inverse(
    std::vector<float> &input,
    int left,
    int right,
    std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}
inline float modelInference::CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}
int modelInference::nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] == -1 || classIds[i] != filterId)
        {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

inline int modelInference::clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
inline int32_t modelInference::__clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

inline float modelInference::deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

int8_t modelInference::qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}
/**
 * @description:
 * @param {char} *model_path_
 * @param {int} thread_index_
 * @param {int} OBJ_CLASS_NUM
 * @param {float} NMS_THRESHOLD
 * @param {float} CONF_THRESHOLD
 * @return {*}
 */
modelInference::modelInference(char *model_path_, int c_model_input_img_size, int thread_index, int OBJ_CLASS_NUM, float NMS_THRESHOLD, float CONF_THRESHOLD)
{
    this->model_path = (char *)malloc(strlen(model_path_) + 1); // 需要加1，以便为字符串结尾的'\0'预留空间

    // 复制原始字符串到临时变量
    strcpy(model_path, model_path_);
    this->c_thread_index = thread_index;
    this->c_obj_class_num = OBJ_CLASS_NUM;
    this->c_nms_threshold = NMS_THRESHOLD;
    this->c_conf_threshold = CONF_THRESHOLD;
    this->c_model_input_img_size = c_model_input_img_size;
}

int modelInference::read_data_from_file(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

int modelInference::init()
{
    int ret = 0;
    int model_len = 0;
    char *model;
    model_info.ctx = 0;

    memset(&c_detect_result_group, 0x00, sizeof(c_detect_result_group));

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    rknn_core_mask core_mask;
    if (c_thread_index == 0)
        core_mask = RKNN_NPU_CORE_0;
    else if (c_thread_index == 1)
        core_mask = RKNN_NPU_CORE_1;
    else
        core_mask = RKNN_NPU_CORE_2;

    // core_mask = RKNN_NPU_CORE_AUTO;
    ret = rknn_init(&(model_info.ctx), model, model_len, 0, NULL);
    rknn_set_core_mask(model_info.ctx, core_mask);
    free(model);

    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    /* Query sdk version */
    rknn_sdk_version version;
    ret = rknn_query(model_info.ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(model_info.ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(model_info.ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(model_info.ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // 此处判断模型是否使用int8或者FP16
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    {
        model_info.is_quant = true;
    }
    else
    {
        model_info.is_quant = false;
    }

    model_info.io_num = io_num;
    model_info.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(model_info.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    model_info.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(model_info.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        model_info.model_channel = input_attrs[0].dims[1];
        model_info.model_height = input_attrs[0].dims[2];
        model_info.model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        model_info.model_height = input_attrs[0].dims[1];
        model_info.model_width = input_attrs[0].dims[2];
        model_info.model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           model_info.model_height, model_info.model_width, model_info.model_channel);

    return 0;
}
int modelInference::detect()
{

    int startX, startY;
    float ratio;
    int ret;

    // Set Input Data
    rknn_input inputs[model_info.io_num.n_input];
    rknn_output outputs[model_info.io_num.n_output];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));


    cv::Mat dst = preprocess(c_img, startX, startY, ratio);
    // std::cout<<"startX,startY,ratio:"<<startX<<" "<<startY<<" "<<ratio;
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_info.model_width * model_info.model_height * model_info.model_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    
    inputs[0].buf = dst.data;
    ret = rknn_inputs_set(model_info.ctx, model_info.io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    ret = rknn_run(model_info.ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));

    for (int i = 0; i < model_info.io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!model_info.is_quant);
    }

    ret = rknn_outputs_get(model_info.ctx, model_info.io_num.n_output, outputs, NULL);

    post_process(outputs,ratio, startX, startY);

    // std::cout<<"detect_result_group.count: "<<c_detect_result_group.count<<std::endl;
    c_boxs.clear();
    for (int i = 0; i < c_detect_result_group.count; i++)
    {
        if (c_detect_result_group.results[i].prob > c_conf_threshold)
        {
            c_boxs.push_back(c_detect_result_group.results[i]);
            // rectangle(img, cv::Point(boxs.back().x, boxs.back().y), cv::Point(boxs.back().x + boxs.back().w, boxs.back().y + boxs.back().h), cv::Scalar(255, 0, 0), 2);
        }
    }
    rknn_outputs_release(model_info.ctx, model_info.io_num.n_output, outputs);
    return ret;
}

int modelInference::post_process(rknn_output *outputs,float ratio, int startX, int startY)
{
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = model_info.model_width;
    int model_in_h = model_info.model_height;

    memset(&c_detect_result_group, 0, sizeof(object_detect_result_list));

    for (int i = 0; i < 3; i++)
    {
        grid_h = model_info.output_attrs[i].dims[2];
        grid_w = model_info.output_attrs[i].dims[3];
        stride = model_in_h / grid_h;

        if (model_info.is_quant)
        {
            validCount += process_i8((int8_t *)outputs[i].buf, grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
                                     classId, c_conf_threshold, model_info.output_attrs[i].zp, model_info.output_attrs[i].scale);
        }
        else
        {
            validCount += process_fp32((float *)outputs[i].buf, grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
                                       classId, c_conf_threshold);
        }
    }

    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, c_nms_threshold);
    }

    int last_count = 0;
    c_detect_result_group.count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - startX;
        float y1 = filterBoxes[n * 4 + 1] - startY;
        float w = filterBoxes[n * 4 + 2];
        float h = filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        c_detect_result_group.results[last_count].x = (int)(clamp(x1, 0, model_in_w) * ratio);
        c_detect_result_group.results[last_count].y = (int)(clamp(y1, 0, model_in_h) * ratio);
        c_detect_result_group.results[last_count].w = w*ratio;
        c_detect_result_group.results[last_count].h = h*ratio;
        c_detect_result_group.results[last_count].prob = obj_conf;
        c_detect_result_group.results[last_count].obj_id = id;

        last_count++;
    }
    c_detect_result_group.count = last_count;
    return 0;
}

int modelInference::process_fp32(float *input, int grid_h, int grid_w, int height, int width, int stride,
                                 std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            float box_confidence = input[4 * grid_len + i * grid_w + j];
            if (box_confidence >= threshold)
            {
                int offset = i * grid_w + j;
                float *in_ptr = input + offset;
                float box_x = *in_ptr;
                float box_y = in_ptr[grid_len];
                float box_w = in_ptr[2 * grid_len];
                float box_h = in_ptr[3 * grid_len];
                box_x = (box_x + j) * (float)stride;
                box_y = (box_y + i) * (float)stride;
                box_w = exp(box_w) * stride;
                box_h = exp(box_h) * stride;
                box_x -= (box_w / 2.0);
                box_y -= (box_h / 2.0);

                float maxClassProbs = in_ptr[5 * grid_len];
                int maxClassId = 0;
                for (int k = 1; k < c_obj_class_num; ++k)
                {
                    float prob = in_ptr[(5 + k) * grid_len];
                    if (prob > maxClassProbs)
                    {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }
                if (maxClassProbs > threshold)
                {
                    objProbs.push_back(maxClassProbs * box_confidence);
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

int modelInference::process_i8(int8_t *input, int grid_h, int grid_w, int height, int width, int stride,
                               std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                               int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    for (int i = 0; i < grid_h; ++i)
    {
        for (int j = 0; j < grid_w; ++j)
        {
            int8_t box_confidence = input[4 * grid_len + i * grid_w + j];
            if (box_confidence >= thres_i8)
            {
                int offset = i * grid_w + j;
                int8_t *in_ptr = input + offset;

                int8_t maxClassProbs = in_ptr[5 * grid_len];
                int maxClassId = 0;
                for (int k = 1; k < c_obj_class_num; ++k)
                {
                    int8_t prob = in_ptr[(5 + k) * grid_len];
                    if (prob > maxClassProbs)
                    {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }

                if (maxClassProbs > thres_i8)
                {
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale));
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale));
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale));
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale));
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

modelInference::~modelInference()
{
    if (model_info.ctx != 0)
    {
        rknn_destroy(model_info.ctx);
        model_info.ctx = 0;
    }
    if (model_info.input_attrs != NULL)
    {
        free(model_info.input_attrs);
        model_info.input_attrs = NULL;
    }
    if (model_info.output_attrs != NULL)
    {
        free(model_info.output_attrs);
        model_info.output_attrs = NULL;
    }
    
    std::cout << "destroy modelInference!" << std::endl;

}
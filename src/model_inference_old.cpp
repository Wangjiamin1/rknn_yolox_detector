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
    cv::Mat resizedImage(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    // 计算调整大小后的图像的宽度和高度
    int resizedWidth, resizedHeight;
    if (originalWidth > originalHeight)
    {
        resizedWidth = 640;
        ratio = originalWidth / 640.0;
        resizedHeight = originalHeight * 640 / originalWidth;
    }
    else
    {
        resizedWidth = originalWidth * 640 / originalHeight;
        ratio = originalHeight / 640.0;
        resizedHeight = 640;
    }
    // 计算调整大小后图像的起始坐标
    startX = (640 - resizedWidth) / 2;
    startY = (640 - resizedHeight) / 2;

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

int modelInference::process_i8(int8_t *input, int *anchor, int anchor_per_branch, int grid_h, int grid_w, int height, int width, int stride,
                               std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId,
                               float threshold, int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = threshold;
    auto thres_i8 = qnt_f32_to_affine(thres, zp, scale);
    for (int a = 0; a < anchor_per_branch; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {

            for (int j = 0; j < grid_w; j++)
            {

                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                // printf("The box confidence in i8: %d\n", box_confidence);
                if (box_confidence >= thres_i8)
                {
                    // printf("box_conf %u, thres_i8 %u\n", box_confidence, thres_i8);
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }

                    float box_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                    float class_prob_f32 = deqnt_affine_to_f32(maxClassProbs, zp, scale);
                    float limit_score = 0;
                    limit_score = box_conf_f32 * class_prob_f32;
                    // printf("limit score: %f\n", limit_score);
                    if (limit_score > CONF_THRESHOLD)
                    {
                        float box_x, box_y, box_w, box_h;
                        box_x = deqnt_affine_to_f32(*in_ptr, zp, scale);
                        box_y = deqnt_affine_to_f32(in_ptr[grid_len], zp, scale);
                        box_w = deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale);
                        box_h = deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale);
                        box_w = exp(box_w) * stride;
                        box_h = exp(box_h) * stride;

                        box_x = (box_x + j) * (float)stride;
                        box_y = (box_y + i) * (float)stride;
                        box_w *= (float)anchor[a * 2];
                        box_h *= (float)anchor[a * 2 + 1];
                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                        boxScores.push_back(box_conf_f32 * class_prob_f32);
                        classId.push_back(maxClassId);
                        validCount++;
                    }
                }
            }
        }
    }
    return validCount;
}

int modelInference::post_process(void **rk_outputs, detect_result_group_t *group, float ratio, int startX, int startY)
{
    memset(group, 0, sizeof(detect_result_group_t));

    std::vector<float> filterBoxes;
    std::vector<float> boxesScore;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int *anchors;
    for (int i = 0; i < model_info->n_output; i++)
    {
        stride = model_info->strides[i];
        grid_h = model_info->height / stride;
        grid_w = model_info->width / stride;
        anchors = &(model_info->anchors[i * 2 * model_info->anchor_per_branch]);
        validCount = validCount + process_i8((int8_t *)rk_outputs[i], anchors, model_info->anchor_per_branch, grid_h, grid_w, model_info->height, model_info->width, stride,
                                             filterBoxes, boxesScore, classId, CONF_THRESHOLD, model_info->out_attr[i].zp, model_info->out_attr[i].scale);
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

    quick_sort_indice_inverse(boxesScore, 0, validCount - 1, indexArray);

    nms(validCount, filterBoxes, classId, indexArray, NMS_THRESHOLD, true);

    int last_count = 0;
    group->count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {

        if (indexArray[i] == -1 || boxesScore[i] < CONF_THRESHOLD || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];

        int id = classId[n];

        group->results[last_count].x = ((int)((x1 - startX) * ratio)) > 0 ? (int)((x1 - startX) * ratio) : 0;
        group->results[last_count].y = (int)((y1 - startY) * ratio) > 0 ? (int)((y1 - startY) * ratio) : 0;
        group->results[last_count].w = ((int)((x2 - startX) * ratio) > 0 ? (int)((x2 - startX) * ratio) : 0) - group->results[last_count].x;
        group->results[last_count].h = ((int)((y2 - startY) * ratio) > 0 ? (int)((y2 - startY) * ratio) : 0) - group->results[last_count].y;
        group->results[last_count].prob = boxesScore[i];
        group->results[last_count].obj_id = id;
        last_count++;
    }
    group->count = last_count;

    return 0;
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

modelInference::modelInference(char *model_path_, int thread_index_, int OBJ_CLASS_NUM)
{
    model_path = (char *)malloc(strlen(model_path_) + 1); // 需要加1，以便为字符串结尾的'\0'预留空间

    // 复制原始字符串到临时变量
    strcpy(model_path, model_path_);
    thread_index = thread_index_;
    model_info = new MODEL_INFO;
    this->OBJ_CLASS_NUM = OBJ_CLASS_NUM;
    PROP_BOX_SIZE = OBJ_CLASS_NUM + 5;
    // run_flag = true;
}
int modelInference::init()
{
    int ret = 0;
    int model_data_size = 0;
    for (int i = 0; i < 18; i++)
    {
        model_info->anchors[i] = 1;
    }
    model_info->anchor_per_branch = 1;
    model_data = load_model(model_path, &model_info->init_flag);
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    std::cout << model_path << std::endl;

    rknn_core_mask core_mask;
    // if (thread_index == 0)
    // core_mask = RKNN_NPU_CORE_0;
    // else if (thread_index == 1)
    //     core_mask = RKNN_NPU_CORE_1;
    // else
    //     core_mask = RKNN_NPU_CORE_2;
    core_mask = RKNN_NPU_CORE_AUTO;

    ret = rknn_init(&(model_info->ctx), (void *)model_data, model_info->init_flag, 0, NULL);

    rknn_set_core_mask(model_info->ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    /* Query sdk version */
    rknn_sdk_version version;
    ret = rknn_query(model_info->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    /* Get input,output attr */
    rknn_input_output_num io_num;
    ret = rknn_query(model_info->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    model_info->n_input = io_num.n_input;
    model_info->n_output = io_num.n_output;

    model_info->inputs = (rknn_input *)malloc(sizeof(rknn_input) * model_info->n_input);
    model_info->in_attr = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * model_info->n_input);
    for (int i = 0; i < model_info->n_input; i++)
    {
        memset(&(model_info->inputs[i]), 0, sizeof(rknn_input));
        // memset(&(model_info->rkdmo_input_param[i]), 0, sizeof(RKDEMO_INPUT_PARAM));
        model_info->in_attr[i].index = i;
        ret = rknn_query(model_info->ctx, RKNN_QUERY_INPUT_ATTR, &(model_info->in_attr[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        if (model_info->verbose_log == true)
        {
            dump_tensor_attr(&model_info->in_attr[i]);
        }
    }

    model_info->outputs = (rknn_output *)malloc(sizeof(rknn_output) * model_info->n_output);
    model_info->out_attr = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * model_info->n_output);

    for (int i = 0; i < model_info->n_output; i++)
    {
        memset(&(model_info->outputs[i]), 0, sizeof(rknn_output));
        model_info->out_attr[i].index = i;
        // printf("model_info->outputs[%d].index = %d\n", i, model_info->outputs[i].want_float);
        ret = rknn_query(model_info->ctx, RKNN_QUERY_OUTPUT_ATTR, &(model_info->out_attr[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        if (model_info->verbose_log == true)
        {
            dump_tensor_attr(&model_info->out_attr[i]);
        }
    }

    if (model_info->in_attr[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        model_channel = model_info->in_attr[0].dims[1];
        model_height = model_info->in_attr[0].dims[2];
        model_width = model_info->in_attr[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        model_height = model_info->in_attr[0].dims[1];
        model_width = model_info->in_attr[0].dims[2];
        model_channel = model_info->in_attr[0].dims[3];
    }
    model_info->channel = model_channel;
    model_info->height = model_height;
    model_info->width = model_width;

    // resize_buf = malloc(model_height * model_width * model_channel);
    // memset(resize_buf, 0x00, model_width * model_height * model_channel);
    printf("model input height=%d, width=%d, channel=%d\n", model_height, model_width, model_channel);
}
int modelInference::detect()
{

    img_width = img.cols;
    img_height = img.rows;
    // cv::Mat det_img;
    // cv::cvtColor(img, det_img, cv::COLOR_BGR2RGB);
    int startX, startY;
    float ratio;

    image_buffer_t dst_img;
    letterbox_t letter_box;
    int bg_color = 114;

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_width * model_height * model_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // Pre Process
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // cv::Mat dst = preprocess(img, startX, startY, ratio);
    ret = convert_image_with_letterbox(img.data, &dst_img, &letter_box, bg_color);

    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }
    inputs[0].buf = dst.data;
    rknn_inputs_set(model_info->ctx, model_info->n_input, inputs);
    int ret = 0;
    ret = rknn_run(model_info->ctx, NULL);

    ret = rknn_outputs_get(model_info->ctx, model_info->n_output, model_info->outputs, NULL);

    /* Post process */
    void *rk_outputs_buf[model_info->n_output];
    detect_result_group_t detect_result_group;
    for (auto i = 0; i < model_info->n_output; i++)
        rk_outputs_buf[i] = model_info->outputs[i].buf;
    post_process(rk_outputs_buf, &detect_result_group, ratio, startX, startY);
    // std::cout<<"detect_result_group.count: "<<detect_result_group.count<<std::endl;
    for (int i = 0; i < detect_result_group.count; i++)
    {
        if (detect_result_group.results[i].prob > CONF_THRESHOLD)
        {
            boxs.push_back(detect_result_group.results[i]);
            // rectangle(img, cv::Point(boxs.back().x, boxs.back().y), cv::Point(boxs.back().x + boxs.back().w, boxs.back().y + boxs.back().h), cv::Scalar(255, 0, 0), 2);
        }
    }
    ret = rknn_outputs_release(model_info->ctx, model_info->n_output, model_info->outputs);
    return 0;
}

modelInference::~modelInference()
{
    // run_flag = false;

    free(model_path);
    if (model_info->ctx > 0)
    {
        rknn_destroy(model_info->ctx);
    }

    if (model_data)
    {
        free(model_data);
    }

    if (model_info->in_attr)
    {
        free(model_info->in_attr);
    }

    if (model_info->out_attr)
    {
        free(model_info->out_attr);
    }

    // if (resize_buf)
    // {
    //     free(resize_buf);
    // }
    if (model_info->inputs)
    {
        free(model_info->inputs);
    }
    if (model_info->outputs)
    {
        free(model_info->outputs);
    }
    std::cout << "destroy modelInference!" << std::endl;
}
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <iostream>
#include <sys/mman.h>
#include <cstdint>
#include <cstring>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

struct buffer
{
    void *start;
    struct v4l2_plane *planes_buffer;
};

class V4L2Capture
{
public:
    V4L2Capture(const char *device, int width, int height, int fps) : device_(device), width_(width), height_(height), fps_(fps)
    {
        file_fd_raw = fopen("raw.yuv", "wb+");
        if (!file_fd_raw)
        {
            printf("open save_file: %s fail\n", "raw.yuv");
        }
        // memset(&modifiedYData, 0, sizeof(modifiedYData));
    }

    bool openDevice()
    {
        fd_ = open(device_, O_RDWR);
        if (fd_ < 0)
        {
            std::cerr << "Failed to open device" << std::endl;
            return false;
        }
        return true;
    }

    bool queryCapability()
    {
        memset(&cap, 0, sizeof(cap));

        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0)
        {
            printf("Get video capability error!\n");
            return false;
        }
        printf("driver : %s\n", cap.driver);
        printf("device : %s\n", cap.card);
        printf("bus : %s\n", cap.bus_info);
        printf("version : %d n", cap.version);
        if (!(cap.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE))
        {
            printf("Video device not support capture!\n");
            return false;
        }
        return true;
    }

    bool initDevice()
    {
        memset(&format, 0, sizeof(format));
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        format.fmt.pix.width = width_;
        format.fmt.pix.height = height_;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
        format.fmt.pix.field = V4L2_FIELD_ANY;

        if (ioctl(fd_, VIDIOC_S_FMT, &format) < 0)
        {
            std::cerr << "Failed to set format" << std::endl;
            return false;
        }
        printf("width = %d\n", format.fmt.pix_mp.width);
        printf("height = %d\n", format.fmt.pix_mp.height);
        printf("nmplane = %d\n", format.fmt.pix_mp.num_planes);

        // TODO: More initialization (e.g., buffer allocation) goes here

        return true;
    }

    bool mmap_v4l2_buffer()
    {
        int buf_num = 4;
        buffers = (struct buffer *)calloc(buf_num, sizeof(*buffers));
        if (!buffers)
        {
            printf("calloc \"frame buffer \" error : Out of memory\n");
            return false;
        }
        req.count = buf_num;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        req.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0)
        {
            printf("Reqbufs fail\n");
            return -1;
        }
        for (int i = 0; i < req.count; i++)
        {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane *planes_buffer;
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            planes_buffer = (struct v4l2_plane *)calloc(1, sizeof(*planes_buffer));
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            v4l2_buf.memory = V4L2_MEMORY_MMAP;
            v4l2_buf.m.planes = planes_buffer;
            v4l2_buf.length = 1;
            v4l2_buf.index = i;
            if (-1 == ioctl(fd_, VIDIOC_QUERYBUF, &v4l2_buf))
            {
                printf("Querybuf fail\n");
                req.count = i;
                return false;
            }
            printf("plane[%d]: length = %d\n", 1, (planes_buffer)->length);
            printf("plane[%d]: offset = %d\n", 1, (planes_buffer)->m.mem_offset);
            (buffers + i)->planes_buffer = planes_buffer;
            (buffers + i)->start = mmap(NULL /* start anywhere */,
                                        (planes_buffer)->length,
                                        PROT_READ | PROT_WRITE /* required */,
                                        MAP_SHARED /* recommended */,
                                        fd_,
                                        (planes_buffer)->m.mem_offset);
            if (MAP_FAILED == (buffers + i)->start)
            {
                printf("mmap failed\n");
                req.count = i;
                return false;
            }
            else
            {
                if (ioctl(fd_, VIDIOC_QBUF, &v4l2_buf) < 0)
                {
                    printf("VIDIOC_QBUF failed\n");
                    return false;
                }
            }
        }
        return true;
    }

    bool startCapture()
    {
        // TODO: Start capturing frames
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0)
        {
            printf("VIDIOC_STREAMON failed\n");
            return false;
        }
        return true;
    }

    cv::Mat nv12ToMat(uint8_t *nv12Data, int width, int height)
    {
        cv::Mat nv12Image(height + height / 2, width, CV_8UC1, (unsigned char *)nv12Data);
        cv::Mat bgrImage;
        cv::cvtColor(nv12Image, bgrImage, cv::COLOR_YUV2BGR_NV12);

        return bgrImage;
    }

    cv::Mat writeVideoFrame()
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane *planes_buffer;
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        planes_buffer = (struct v4l2_plane *)calloc(1, sizeof(*planes_buffer));
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        v4l2_buf.memory = V4L2_MEMORY_MMAP;
        v4l2_buf.m.planes = planes_buffer;
        v4l2_buf.length = 1;
        if (ioctl(fd_, VIDIOC_DQBUF, &v4l2_buf) < 0)
            printf("dqbuf fail\n");
        // printf("plane[%d] start = %p, bytesused = %d\n", 1, (buffers + v4l2_buf.index)->start, planes_buffer->bytesused);
        // fwrite((buffers + v4l2_buf.index)->start, planes_buffer->bytesused, 1, file_fd_raw);
        // for (int index_ = 0; index_ < 192 * 256; index_++)
        // {

        //     modifiedYData[index_] = ((uint16_t)(*((uint8_t *)(buffers + v4l2_buf.index)->start + index_ * 2)) & 0xFF);
        //     modifiedYData[index_] |= (((uint16_t)(*((uint8_t *)(buffers + v4l2_buf.index)->start + index_ * 2 + 1)) & 0x3F) << 8);
        //     modifiedYData[index_] = (uint16_t)(modifiedYData[index_]) << 2;
        // }
        // cv::Mat mat(1920, 1080, CV_16U, modifiedYData);
        cv::Mat mat = nv12ToMat((uint8_t *)((buffers + v4l2_buf.index)->start),1920,1080);
        if (ioctl(fd_, VIDIOC_QBUF, &v4l2_buf) < 0)
        {
            printf("failture VIDIOC_QBUF\n");
        }
        free(planes_buffer);
        return mat;
    }

    void stopCapture()
    {
        // TODO: Stop capturing frames
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0)
            printf("VIDIOC_STREAMOFF fail\n");
    }

    bool unmap_v4l2_buffer()
    {
        for (int i = 0; i < req.count; i++)
        {
            if (MAP_FAILED != (buffers + i)->start)
            {
                if (-1 == munmap((buffers + i)->start, ((buffers + i)->planes_buffer)->length))
                    printf("munmap error\n");
                return false;
            }
        }

        for (int i = 0; i < req.count; i++)
        {
            free((buffers + i)->planes_buffer);
            free((buffers + i)->start);
        }
        return true;
    }

    ~V4L2Capture()
    {
        if (fd_ >= 0)
        {
            unmap_v4l2_buffer();
            close(fd_);
        }
    }

private:
    int fd_ = -1;
    const char *device_;
    int width_, height_, fps_;
    struct v4l2_capability cap;
    struct v4l2_format format;
    struct v4l2_requestbuffers req;
    struct buffer *buffers;
    FILE *file_fd_raw;
    // uint16_t modifiedYData[192 * 256];
};

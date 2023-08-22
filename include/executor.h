//
// Created by huangyuyang on 6/13/23.
//

#ifndef FASTLLM_EXECUTOR_H
#define FASTLLM_EXECUTOR_H

#include "device.h"

namespace fastllm {
    class Executor {
    private:
        std::vector <BaseDevice*> devices;
        std::map <std::string, float> profiler; // 统计每一个算子耗时,相同算子采用累加形式
        // 统计某算子多次调用的数据维度
        std::map <std::string, std::vector<DataDict>> op_profile;
        // 统计某算子多次调用耗费时间
        std::map<std::string, std::vector<float>> op_profile_t;

    public:
        Executor (); // 创建默认的Executor

        ~Executor(); // 析构

        void ClearDevices(); // 清空 devices

        void AddDevice(BaseDevice *device); // 增加一个device

        void SetFirstDevice(const std::string &device); // 设定优先的device

        std::vector <int> GetDeviceIds(const std::string &device); // 获取指定device的deviceIds

        // 运行一个op
        void Run(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                 const fastllm::IntDict &intParams);

        void ClearProfiler();

        void PrintProfiler();
    };
}

#endif //FASTLLM_EXECUTOR_H

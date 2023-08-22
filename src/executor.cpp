//
// Created by huangyuyang on 6/13/23.
//

#include "utils.h"

#include "executor.h"

#include "devices/cpu/cpudevice.h"

#ifdef USE_CUDA
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {
    Executor::Executor() {
        this->devices.clear();
#ifdef USE_CUDA
        this->devices.push_back((BaseDevice*) new CudaDevice());
#endif
        this->devices.push_back((BaseDevice*) new CpuDevice());
    }

    Executor::~Executor() {
        for (int i = 0; i < devices.size(); i++) {
            delete devices[i];
        }
    }

    void Executor::ClearDevices() {
        this->devices.clear();
    }

    void Executor::AddDevice(fastllm::BaseDevice *device) {
        this->devices.push_back(device);
    }

    void Executor::SetFirstDevice(const std::string &device) {
        auto temp = this->devices;
        this->devices.clear();
        for (int i = 0; i < temp.size(); i++) {
            if (StartWith(device, temp[i]->deviceType)) {
                this->devices.push_back(temp[i]);
                this->devices.back()->deviceIds = ParseDeviceIds(device, temp[i]->deviceType);
            }
        }
        for (int i = 0; i < temp.size(); i++) {
            if (!StartWith(device, temp[i]->deviceType)) {
                this->devices.push_back(temp[i]);
            }
        }
    }

    std::vector <int> Executor::GetDeviceIds(const std::string &device) {
        for (int i = 0; i < devices.size(); i++) {
            if (StartWith(devices[i]->deviceType, device)) {
                return devices[i]->deviceIds;
            }
        }
        return {0};
    }

    void Executor::Run(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                       const fastllm::IntDict &intParams) {

        auto st = std::chrono::system_clock::now();
        bool lockInCPU = false;
        for (auto &it: datas) {
            if (intParams.find(it.first + "___batch") != intParams.end()) {
                int batch = intParams.find(it.first + "___batch")->second;
                for (int i = 0; i < batch; i++) {
                    lockInCPU |= ((Data**)it.second)[i]->lockInCPU;
                }
            } else {
                lockInCPU |= it.second->lockInCPU;
            }
        }
        for (auto device: devices) {
            if (lockInCPU && device->deviceType != "cpu") {
                continue;
            }
            if (device->CanRun(opType, datas, floatParams, intParams)) {
#ifdef USE_CUDA
                if (device->deviceType == "cuda" && device->deviceIds.size() > 0) {
                    FastllmCudaSetDevice(device->deviceIds[0]);
                }
#endif
                for (auto &it: datas) {
                    if (intParams.find(it.first + "___batch") != intParams.end()) {
                        int batch = intParams.find(it.first + "___batch")->second;
                        for (int i = 0; i < batch; i++) {
                            ((Data**)it.second)[i]->ToDevice((void *) device);
                        }
                    } else {
                        it.second->ToDevice((void *) device);
                    }
                }
                device->Reshape(opType, datas, floatParams, intParams);
                device->Run(opType, datas, floatParams, intParams);
                break;
            }
        }
        float spend = GetSpan(st, std::chrono::system_clock::now());

        // 在这里统计输入形状,并且只是统计opType为Linear的算子
        op_profile[opType].push_back(datas);
        op_profile_t[opType].push_back(spend);
        // 输入数据和权重和bias都在DataDict中存储着
        profiler[opType] += spend;
    }

    void Executor::ClearProfiler() {
        profiler.clear();
    }

    void Executor::PrintProfiler() {

        // 只打印Linear算子的耗时和维度
        std::vector<fastllm::DataDict> dicv = op_profile.at("Linear");
        std::vector<float> timev =  op_profile_t.at("Linear");
        if(dicv.size() != timev.size()){
            printf("Error!");
            // return;
        }
        
        float sum_1 = 0.0;
        for(int i = 0; i < timev.size(); ++i){
            fastllm::DataDict dic = dicv[i];
            sum_1 +=  timev[i];
            printf("Linear %d spend %fs \n",i + 1, timev[i]);
            
        }
        float sum = 0.0;
        for (auto &it : profiler) {
            printf("%s spend %fs \n", it.first.c_str(), it.second);
            sum += it.second;
        }
        // printf("total spend %fs\n", sum);
        printf("total spend %fs and %fs \n", sum, sum_1);
    }
}
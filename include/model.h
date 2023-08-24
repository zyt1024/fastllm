//
// Created by huangyuyang on 6/20/23.
//

#ifndef FASTLLM_MODEL_H
#define FASTLLM_MODEL_H

#include "basellm.h"
#include <sstream>
namespace fastllm {
    std::unique_ptr<basellm> CreateLLMModelFromFile(const std::string &fileName);

    std::unique_ptr<basellm> CreateEmptyLLMModel(const std::string &modelType);
}

#endif //FASTLLM_MODEL_H

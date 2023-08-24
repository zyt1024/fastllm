//
// Created by huangyuyang on 6/9/23.
//

#include "model.h"
#include "utils.h"
#include "fstream"

struct BenchmarkConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    int threads = 4; // 使用的线程数
    int limit = -1; // 输出token数限制，如果 < 0 则代表无限制
    int batch = -1; // batch数, -1时使用文件中的行数作为batch
    std::string file; // 输入文件
    std::string output; // 输出文件，如果不设定则输出到屏幕
    bool printProfile = false; // 是否打印性能分析
    bool print_perf = false; // 是否输出性能信息
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--limit> <args>:          输出token数限制" << std::endl;
    std::cout << "<-b|--batch> <args>:          batch数"      << std::endl;
    std::cout << "<-f|--file> <args>:           输入文件，文件中每行一个prompt，如果行数不足batch则用之前的prompt补充"      << std::endl;
    std::cout << "<--print_profiler>:           打印推理各个算子时间" << std::endl;
    std::cout << "<--print_perf>:               选项打开时，输出性能信息" << std::endl;
}

double GetSpan(std::chrono::high_resolution_clock::time_point time1, std::chrono::high_resolution_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};
void ParseArgs(int argc, char **argv, BenchmarkConfig &config) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        }
        else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--limit") {
            config.limit = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-b" || sargv[i] == "--batch") {
            config.batch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-f" || sargv[i] == "--file") {
            config.file = sargv[++i];
        } else if (sargv[i] == "-o" || sargv[i] == "--output") {
            config.output = sargv[++i];
        } else if (sargv[i] == "--print_profiler"){
            config.printProfile = true;
        }  else if (sargv[i] == "--print_perf") {
            config.print_perf = true;
        } else {
            Usage();
            exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    BenchmarkConfig config;
    ParseArgs(argc, argv, config);
    fastllm::SetThreads(config.threads);
    auto model = fastllm::CreateLLMModelFromFile(config.path);
    fastllm::GenerationConfig generationConfig;
    generationConfig.output_token_limit = config.limit;

    fastllm::PrintInstructionInfo();
    std::vector <std::string> inputs;
    if (config.file != "") {
        std::ifstream finputs(config.file, std::ios::in);
        while (true) {
            std::string input = "";
            std::getline(finputs, input);
            if (input == "") {
                break;
            } else {
                inputs.push_back(input);
            }
        }
    } else {
        inputs.push_back("您好");
    }
    if (config.batch < 0) {
        config.batch = inputs.size();
    }
    while (inputs.size() < config.batch) {
        inputs.push_back(inputs[rand() % inputs.size()]);
    }
    if (inputs.size() > config.batch && config.batch != -1) {
        inputs.resize(config.batch);
    }

    int promptTokenNum = 0;
    // for (int i = 0; i < inputs.size(); i++) {
    //     inputs[i] = model->MakeInput("", 0, inputs[i]);
    //     promptTokenNum += model->weight.tokenizer.Encode(inputs[i]).Count(0);
    // }
    auto model_input = model->MakeInput("",0,inputs[0]);
    std::vector <std::string> outputs;
    static int tokens = 0;
    //可重复执行2, 4, 8, 16, 32, , 256, 512, 1024
    std::vector<int> values = {32 , 64, 128, 512};
    for(int on = 0; on < values.size(); on++){
        for(int ok = 0; ok < values.size(); ok++){
            for(int om = 0; om < values.size(); om++){
                for(int nt = 0; nt < values.size(); nt++){
                    for(int kt = 0; kt < values.size(); kt++){
                        for(int mt = 0; mt < values.size(); mt++){
                            printf("%d,%d,%d,%d,%d,%d\n",values[on],values[ok],values[om],values[nt],values[kt],values[mt]);
                            tokens = 0;
                            fastllm::setOutNTIleSize(values[on]);
                            fastllm::setOutKTIleSize(values[ok]);
                            fastllm::setOutMTIleSize(values[om]);
                            fastllm::setNTIleSize(values[nt]);
                            fastllm::setKTIleSize(values[kt]);
                            fastllm::setMTIleSize(values[mt]);
                            auto ts_0 = std::chrono::high_resolution_clock::now();
                            static auto ts_1 = ts_0;
                            static auto ts_2 = ts_0;    
                            // static auto promptTime = st;

                            std::string ret = model->Response(model_input,[](int index, const char* content) {
                                    if (index == 0) {
                                        ts_1 = std::chrono::high_resolution_clock::now();
                                        printf("%s:%s", "chatGLM2", content);
                                        fflush(stdout);
                                    }
                                    if (index > 0) {
                                        printf("%s", content);
                                        fflush(stdout);
                                    }
                                    if (index == -1) {
                                        printf("\n");
                                        ts_2 = std::chrono::high_resolution_clock::now();
                                    }
                            }, generationConfig);

                            if (config.output != "") {
                                FILE *fo = fopen(config.output.c_str(), "w");
                                for (int i = 0; i < outputs.size(); i++) {
                                    fprintf(fo, "[ user: \"%s\", model: \"%s\"]\n", inputs[i].c_str(), outputs[i].c_str());
                                }
                                fclose(fo);
                            } else {
                                for (int i = 0; i < outputs.size(); i++) {
                                    printf("[ user: \"%s\", model: \"%s\"]\n", inputs[i].c_str(), outputs[i].c_str());
                                }
                            }
                            auto num_prefilling_tokens = model->weight.tokenizer.Encode(model_input).Count(0);
                            auto num_generated_tokens = model->weight.tokenizer.Encode(ret).Count(0);
                            auto prefilling_secs = GetSpan(ts_0, ts_1);
                            auto decoding_secs = GetSpan(ts_1, ts_2);
                            printf(
                                "[ pref #%ld, %f sec, %f t/s | dec #%ld, %f sec, %f t/s]\n",
                                num_prefilling_tokens, prefilling_secs, num_prefilling_tokens / prefilling_secs,
                                num_generated_tokens, decoding_secs, num_generated_tokens / decoding_secs);
                            // printf("batch: %d\n", (int)inputs.size());
                            // printf("prompt token number = %d\n", promptTokenNum);
                            // printf("prompt use %f s\n", promptSpend);
                            // printf("prompt speed = %f tokens / s\n", (float)promptTokenNum / promptSpend);
                            // printf("output %d tokens\nuse %f s\nspeed = %f tokens / s\n", tokens, spend, tokens / spend);
                        }
                    }
                }
            }
        }
    }


    if(config.printProfile){
        printf("==================== per op sepend ====================\n");
        fastllm::PrintProfiler();
    }
    return 0;
}
#include "model.h"
#include <sstream>

struct RunConfig {
	std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
	int threads = 4; // 使用的线程数
	bool lowMemMode = false; // 是否使用低内存模式
    bool no_history = false; // 是否每轮都清空历史
    bool print_perf = false; // 是否输出性能信息
    std::string pro_param = ""; //优化参数
};

void Usage() {
	std::cout << "Usage:" << std::endl;
	std::cout << "[-h|--help]:                  显示帮助" << std::endl;
	std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
	std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
	std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--top_p> <args>:             采样参数top_p" << std::endl;
    std::cout << "<--top_k> <args>:             采样参数top_k" << std::endl;
    std::cout << "<--temperature> <args>:       采样参数温度，越高结果越不固定" << std::endl;
    std::cout << "<--repeat_penalty> <args>:    采样参数重复惩罚" << std::endl;
    std::cout << "<--no_history>:               选项打开时，每轮对话都清空历史" << std::endl;
    std::cout << "<--print_perf>:               选项打开时，输出性能信息" << std::endl;
    std::cout << "<--pro_param>:                选项打开时，传入矩阵分块参数" << std::endl;
}
static double GetSpan(std::chrono::high_resolution_clock::time_point time1, std::chrono::high_resolution_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};

void ParseArgs(int argc, char **argv, RunConfig &config, fastllm::GenerationConfig &generationConfig) {
	std::vector <std::string> sargv;
	for (int i = 0; i < argc; i++) {
		sargv.push_back(std::string(argv[i]));
	}
	for (int i = 1; i < argc; i++) {
		if (sargv[i] == "-h" || sargv[i] == "--help") {
			Usage();
			exit(0);
		} else if (sargv[i] == "-p" || sargv[i] == "--path") {
			config.path = sargv[++i];
		} else if (sargv[i] == "-t" || sargv[i] == "--threads") {
			config.threads = atoi(sargv[++i].c_str());
		} else if (sargv[i] == "-l" || sargv[i] == "--low") {
			config.lowMemMode = true;
		} else if (sargv[i] == "-m" || sargv[i] == "--model") {
            i++;
        } else if (sargv[i] == "--top_p") {
            generationConfig.top_p = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--top_k") {
            generationConfig.top_k = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--temperature") {
            generationConfig.temperature = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--repeat_penalty") {
            generationConfig.repeat_penalty = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--no_history") {
            config.no_history = true;
        } else if (sargv[i] == "--print_perf") {
            config.print_perf = true;
        } else if (sargv[i] == "--pro_param") {
            config.pro_param = sargv[++i];
        } else {
			Usage();
			exit(-1);
		}
	}
}

int main(int argc, char **argv) {
    int round = 0;
    std::string history = "";

    RunConfig config;
    fastllm::GenerationConfig generationConfig;
	ParseArgs(argc, argv, config, generationConfig);

    if (config.pro_param != ""){
        printf("1");
        std::vector<int> numbers;
        std::stringstream ss(config.pro_param);
        int number;
        while (ss >> number) {
            numbers.push_back(number);
        }
        fastllm::setOutNTIleSize(numbers[0]);
        fastllm::setOutKTIleSize(numbers[1]);
        fastllm::setOutMTIleSize(numbers[2]);
        fastllm::setNTIleSize(numbers[3]);
        fastllm::setKTIleSize(numbers[4]);
        fastllm::setMTIleSize(numbers[5]);
    }

    fastllm::PrintInstructionInfo();
    fastllm::SetThreads(config.threads);
    fastllm::SetLowMemMode(config.lowMemMode);
    auto model = fastllm::CreateLLMModelFromFile(config.path);

    // static std::string modelType = model->model_type;
    // printf("欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出程序.\n", model->model_type.c_str());
    // static std::string modelType = model->model_type;
    static std::string modelType = "ChatGLM2";
    if (config.no_history) {
        printf("欢迎使用 %s 模型. 输入内容，stop退出程序.\n", modelType.c_str());
    } else {
        printf("欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出程序.\n", modelType.c_str());
    }
    while (true) {
        printf("用户: ");
        std::string input;
        std::getline(std::cin, input);
        if (input == "reset" && !config.no_history) {
            history = "";
            round = 0;
            continue;
        } else if (input == "stop") {
            break;
        }
        auto ts_0 = std::chrono::high_resolution_clock::now();
        static auto ts_1 = ts_0;
        static auto ts_2 = ts_0;
        auto model_input = model->MakeInput(history, round, input);

        std::string ret = model->Response(model_input, [](int index, const char* content) {
            if (index == 0) {
                ts_1 = std::chrono::high_resolution_clock::now();
                printf("%s:%s", modelType.c_str(), content);
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
        history = model->MakeHistory(history, round, input, ret);
        // round++;
        if (config.print_perf) {
            auto num_prefilling_tokens = model->weight.tokenizer.Encode(model_input).Count(0);
            auto num_generated_tokens = model->weight.tokenizer.Encode(ret).Count(0);
            auto prefilling_secs = GetSpan(ts_0, ts_1);
            auto decoding_secs = GetSpan(ts_1, ts_2);
            printf(
                "[ pref #%ld, %f sec, %f t/s | dec #%ld, %f sec, %f t/s]\n",
                num_prefilling_tokens, prefilling_secs, num_prefilling_tokens / prefilling_secs,
                num_generated_tokens, decoding_secs, num_generated_tokens / decoding_secs);
        }
        if (!config.no_history) {
            history = model->MakeHistory(history, round, input, ret);
            round++;
        }
    }

	return 0;
}
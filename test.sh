#!/bin/bash

usage() {
    echo "====================================="
    echo "使用方法: $(basename "$0") [选项] 参数"
    echo "选项:"
    echo "  -h, --help     显示帮助信息"
    echo "参数:"
    echo "  -p, --path VALUE [required]   模型的路径"
    echo "  --taskset  VALUE [required]   将任务绑定到执行核心"
    echo "====================================="
    # 添加其他参数的描述
    exit 1
}
# 解析命令行参数
OPTIONS=$(getopt -o h,p: --long help,taskset: -n "$(basename "$0")" -- "$@")
if [ $? -ne 0 ]; then
    usage
fi

eval set -- "$OPTIONS"
# 设置必需选项的初始值为空
path=""
bind_thread=""
# 处理命令行参数
while true; do
    case "$1" in
        -h | --help)
            usage
            ;;
        --taskset)
            echo "taskset , argument $2"
            bind_thread=$2
			shift 1
            ;;
        -p)
            echo "Path:"
            path=$2
            shift 1
            ;;
        --)
            shift
            break
            ;;
    esac
    shift
done

# 检查必需选项是否已设置
if [ -z "$path" ]; then
    echo "错误：必需的路径选项 -p/--path 未提供"
    usage
fi
if [ -z "$bind_thread" ]; then
    echo "错误：必需的 taskset 选项 --taskset 未提供"
    usage
fi


# 定义参数列表
parameters=(
    "-p $path --print_perf --threads 1 -b 1 --exe_num 2 --pro_param '2048 2048 2048 2048 2048 2048'"
    "-p $path --print_perf --threads 1 -b 1 --exe_num 2 --pro_param '2048 2048 2048 2048 1024 2048'"
    # 添加更多参数组合...
)

# 可执行文件路径
executable_file="taskset -c $bind_thread ./benchmark_zyt"

# 循环遍历参数列表
for param in "${parameters[@]}"; do
    # 构建命令行参数
    command=("$executable_file" "$param")
    echo ${command[@]}
    # 执行命令
    eval ${command[@]}
done

import sys
from transformers import AutoTokenizer, AutoModel
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    basePath = "/home/student/zyt/glm/models/chatglm2-3b/"
    tokenizer = AutoTokenizer.from_pretrained(basePath, trust_remote_code=True)
    model = AutoModel.from_pretrained(basePath, trust_remote_code=True)
    model = model.eval()

    dtype = sys.argv[2] if len(sys.argv) >= 3 else "float16"
    exportPath = sys.argv[1] if len(sys.argv) >= 2 else "chatglm-6b-' + dtype + '.flm"
    exportPath = basePath+exportPath
    torch2flm.tofile(exportPath, model, tokenizer, dtype = dtype)

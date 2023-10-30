import torch, pickle
from collections import OrderedDict
import argparse

class UnpicklerWrapper(pickle.Unpickler):
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)

pickle.Unpickler = UnpicklerWrapper
final = {}

def get_all(d, prefix_keys=[]):
    for key in d.keys():
        if isinstance(d[key], OrderedDict) or isinstance(d[key], dict):
            get_all(d[key], prefix_keys + [key])
        else:
            final['.'.join(prefix_keys + [key])] = d[key]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    megatron_dics = torch.load(args.input, pickle_module=pickle)
    get_all(megatron_dics['model'])
    state_dict = {}
    for key in final.keys():
        state_dict[key.replace('language_model.', 'transformer.')] = final[key]
    torch.save(state_dict, args.output)
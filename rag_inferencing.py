import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModel, AutoTokenizer


class LLM:

    def load_os_llm(self, model_name, load_in_4bit=False, load_in_8bit=False, device_map='cpu', low_cpu_mem_usage=False,
                 torch_dtype=torch.float32, trust_remote_code=True):

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype
            )
        except NameError:
            try:

                model = AutoModel.from_pretrained(
                    model_name,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype
                )
            except:
                model = LlamaForCausalLM.from_pretrained(model_name,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype
                )
        return model

    def load_tokenizer(self, model_name,use_fast_tokenizer=True,revision=True,trust_remote_code=True):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=use_fast_tokenizer,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False, revision=revision, trust_remote_code=True
            )

        return tokenizer

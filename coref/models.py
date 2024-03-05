import torch
import re
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoModelForCausalLM, AutoTokenizer
import transformer_lens.HookedTransformer as HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name
import os
from functools import cache


LLAMA_PATH = os.environ["LLAMA_WEIGHTS"] if "LLAMA_WEIGHTS" in os.environ else None
HF_PATH = os.environ["HF_CACHE"] if "HF_CACHE" in os.environ else None
TULU_PATH = os.environ["TULU_WEIGHTS"] if "TULU_WEIGHTS" in os.environ else None

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]
LLAMA_MODELS = [
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
]
TULU_MODELS = [
    "tulu-7b",
    "tulu-13b",
    "tulu-30b",
    "tulu-65b",
]


def fetch_model(model_tag, device="cuda", num_devices=1, dtype=None):
    if dtype is None:
        dtype = torch.float16 if '65b' in model_tag else torch.float32
    print(f'Loading {model_tag} in precision {dtype}')
    torch.set_grad_enabled(False)
    if model_tag in PYTHIA_MODELS:
        hf_model = AutoModelForCausalLM.from_pretrained(
            get_official_model_name(model_tag),
            **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
            low_cpu_mem_usage=True,
            torch_dtype=dtype
        )
        model = HookedTransformer.from_pretrained(
            model_tag,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            hf_model=hf_model,
            dtype=dtype
        )
    elif model_tag in LLAMA_MODELS:
        LLAMA_SIZE = re.match(r"llama-(\d+)b-hf", model_tag).group(1)
        MODEL_PATH = os.path.join(LLAMA_PATH, f"{LLAMA_SIZE}B")
        tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_PATH)
        hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True, torch_dtype=dtype)
        model = HookedTransformer.from_pretrained(
            model_tag,
            hf_model=hf_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=dtype
        )
        model.tokenizer = tokenizer
        model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif model_tag in TULU_MODELS:
        model_size = re.match(r"tulu-(\d+)b", model_tag).group(1)
        llama_tag = f"llama-{model_size}b-hf"
        model_path = os.path.join(TULU_PATH, f"tulu-merged-{model_size}b")
        # probably okay to use llama tokenizer instead of tulu tokenizer
        # should be just a difference of a padding token at the end
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        hf_model = LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=dtype)
        model = HookedTransformer.from_pretrained(
            llama_tag,
            hf_model=hf_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=dtype
        )
        model.tokenizer = tokenizer
        # the new model already has this i think
        # model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    else:
        raise Exception(f"Unknown model tag {model_tag}")
    model.eval()
    model.model_tag = model_tag
    return model

@cache
def get_llama_tokenizer():
    return LlamaTokenizerFast.from_pretrained(
        os.path.join(LLAMA_PATH, "7B")
    )
@cache
def get_pythia_tokenizer():
    return AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-2.8b",
        **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
    )
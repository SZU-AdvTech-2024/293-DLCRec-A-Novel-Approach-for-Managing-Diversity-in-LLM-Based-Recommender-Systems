from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "../Meta-Llama-3-8B-Instruct/", torch_dtype=torch.float16, device_map="auto"
)

peft_model_id = "./model/movie_sgcate/BERT_GF_d_vlGF_d_ep25_1e-3/"
model = PeftModel.from_pretrained(base_model, peft_model_id, adapter_name="sft")

weighted_adapter_name = "sft-dpo"
model.load_adapter("./model/movie_sgcate/BERT_GF_n_vlGF_n_ep25_1e-3/", adapter_name="dpo")
model.add_weighted_adapter(
    adapters=["sft", "dpo"],
    weights=[0.5, 0.5],
    adapter_name=weighted_adapter_name,
    combination_type="linear"
)
model.set_adapter(weighted_adapter_name)

merged_model_path = "./model/movie_sgcate/BERT_GF_nd_vlGF_nd_ep25_1e-3/"
model.save_pretrained(merged_model_path)
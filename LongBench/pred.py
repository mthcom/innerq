import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.multiprocessing as mp

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")

    # quantization arguments
    parser.add_argument('--k_bits', type=int, default=None)
    parser.add_argument('--v_bits', type=int, default=None)
    parser.add_argument('--key_group_size', type=int, default=32)
    parser.add_argument('--value_group_size', type=int, default=32)
    parser.add_argument('--key_w_recent', type=int, default=32)
    parser.add_argument('--value_w_recent', type=int, default=32)
    parser.add_argument('--key_w_sink', type=int, default=0)
    parser.add_argument('--value_w_sink', type=int, default=0)
    parser.add_argument('--key_flush_size', type=int, default=32)
    parser.add_argument('--value_flush_size', type=int, default=32)
    parser.add_argument('--key_normalize', type=str2bool, default=False)
    parser.add_argument('--key_quantize_over_token', type=str2bool, default=False)
    parser.add_argument('--value_quantize_over_token', type=str2bool, default=True)
    parser.add_argument('--key_quantize_mode', type=str, default='symmetric')
    parser.add_argument('--value_quantize_mode', type=str, default='symmetric')
    return parser.parse_args()

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(args, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:0')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, args)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def get_pred_llama3_1(args, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:0')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, args)
    
    # 1. Explicitly define the Llama 3.1 terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for json_obj in tqdm(data):
        raw_prompt = prompt_format.format(**json_obj)

        # 2. Safely truncate the raw text BEFORE applying the chat template
        tokenized_raw_prompt = tokenizer(raw_prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_raw_prompt) > max_length:
            half = int(max_length / 2)
            raw_prompt = tokenizer.decode(tokenized_raw_prompt[:half], skip_special_tokens=True) + \
                         tokenizer.decode(tokenized_raw_prompt[-half:], skip_special_tokens=True)

        # 3. Build the conversational structure
        messages = []

        # Guide the model to be concise on datasets that normally trigger chatty filler
        if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant. Provide a direct, concise answer to the prompt without any introductory filler, apologies, or extra explanation."
            })

        messages.append({"role": "user", "content": raw_prompt})

        # 4. Apply the native Llama 3.1 chat template
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Tokenize the fully templated string (avoids missing attention_mask warnings)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        context_length = inputs.input_ids.shape[-1]

        # 5. Generate with guaranteed terminators
        if dataset == "samsum": 
            # SAMSum specific newline stop + Llama 3.1 stops
            samsum_stops = terminators + [tokenizer.encode("\n", add_special_tokens=False)[-1]]
            output = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=samsum_stops,
            )[0]
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=terminators, # Terminators are now strictly enforced here
            )[0]

        # 6. Decode strictly the newly generated tokens
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred, 
                "answers": json_obj["answers"], 
                "all_classes": json_obj.get("all_classes", []), 
                "length": json_obj.get("length", 0)
            }, f, ensure_ascii=False)
            f.write('\n')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_model_config(path, args):
    config = AutoConfig.from_pretrained(path)

    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.key_group_size = args.key_group_size
    config.value_group_size = args.value_group_size
    config.key_w_recent = args.key_w_recent
    config.value_w_recent = args.value_w_recent
    config.key_w_sink = args.key_w_sink
    config.value_w_sink = args.value_w_sink
    config.key_flush_size = args.key_flush_size
    config.value_flush_size = args.value_flush_size
    config.key_normalize = args.key_normalize
    config.key_quantize_over_token = args.key_quantize_over_token
    config.value_quantize_over_token = args.value_quantize_over_token
    config.key_quantize_mode = args.key_quantize_mode
    config.value_quantize_mode = args.value_quantize_mode
    # config._attn_implementation = 'flash_attention_2'

    return config


def load_model_and_tokenizer(path, model_name, device, args):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name or "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if args.k_bits != None:
            from innerq.models.llama_innerq import InnerqLlamaForCausalLM
            config = get_model_config(path, args)
            model = InnerqLlamaForCausalLM.from_pretrained(path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = 1
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        if args.dataset:
            if args.dataset not in datasets:
                raise ValueError(f'Invalid dataset name: {args.dataset}')
            datasets = [args.dataset]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"{args.output_dir}"):
                os.makedirs(f"{args.output_dir}")
            out_path = f"{args.output_dir}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
            if not os.path.exists(f"{args.output_dir}"):
                os.makedirs(f"{args.output_dir}")
            out_path = f"{args.output_dir}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        if 'llama3' in model_name:
            get_pred_llama3_1(args, world_size, data_subsets[0], max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path)
        else:
            get_pred(args, world_size, data_subsets[0], max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path)

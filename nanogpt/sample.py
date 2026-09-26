"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import fire


def load_tokenizer(init_from="resume", meta_path=None):
    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and meta_path:
        # meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        max_token = enc.n_vocab - 1
        decode = lambda l: enc.decode([t for t in l if t <= max_token])
    return encode, decode


def load_model(init_from="resume", model_dir=None, device="cuda", compile=False):
    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(model_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    encoder, decoder = load_tokenizer(init_from=init_from, 
                                      meta_path=meta_path)
    return model, encoder, decoder


def generate(model, model_input, context, decoder, num_samples=5, 
             max_new_tokens=500, temperature=0.8, top_k=200):
    # DDP intentionally does not proxy arbitrary module methods such as
    # reset_state() and generate(). Text sampling runs only on the caller
    # (rank 0 during training), so invoke those methods on the wrapped GPT.
    generation_model = model.module if hasattr(model, "module") else model
    # run generation
    with torch.no_grad():
        with context:
            outputs = []
            for k in range(num_samples):
                print("k=", k)
                generation_model.reset_state()
                y = generation_model.generate(
                    model_input, max_new_tokens,
                    temperature=temperature, top_k=top_k)
                print(decoder(y[0].tolist()))
                print('---------------')
                outputs.append(y)
            return outputs


def encode_prompt(prompt, encoder, device):
    # encode the beginning of the prompt
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()
    prompt_ids = encoder(prompt)
    model_input = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    model_input = (model_input[None, ...])
    return model_input


def run(prompt="\n", out_dir="out", init_from="resume", 
        num_samples=5, max_new_tokens=500, temperature=0.8, 
        top_k=200, device="cuda", seed=1337):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model, encoder, decoder = load_model(model_dir=out_dir, 
                                         init_from=init_from, 
                                         device=device)
    model_input = encode_prompt(prompt, encoder, device)
    _ = generate(model, model_input, ctx, decoder,
                 num_samples=num_samples, 
                 max_new_tokens=max_new_tokens, 
                 temperature=temperature, 
                 top_k=top_k)


if __name__ == '__main__':
    fire.Fire(run)

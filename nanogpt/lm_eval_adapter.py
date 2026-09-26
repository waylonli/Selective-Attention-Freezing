"""EleutherAI lm-eval adapter for native nanoGPT checkpoints."""

import contextlib
import pathlib
import sys

import torch
import tiktoken

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_nanogpt_logprobs import load_model

try:
    from lm_eval import utils
    from lm_eval.api.model import TemplateLM
except ImportError as exc:  # optional dependency; training does not require it
    raise ImportError(
        "lm-eval is required: install requirements_downstream.txt") from exc


class NanoGPTLM(TemplateLM):
    """Score standard lm-eval tasks without converting the custom model to HF."""

    backend = "causal"

    def __init__(self, ckpt, *, device="cuda", batch_size=8,
                 dtype="bfloat16", max_gen_toks=128):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)
        self._dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self.model, self.config, self.checkpoint = load_model(
            ckpt, str(self._device))
        if self.config.state_size:
            raise ValueError("stateful checkpoints are not supported by lm-eval")
        self.model.eval()
        self.tokenizer = tiktoken.get_encoding("gpt2")

    @property
    def eot_token_id(self):
        return self.tokenizer.eot_token

    @property
    def max_length(self):
        return self.config.block_size

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        del add_special_tokens, kwargs
        return self.tokenizer.encode(
            string, allowed_special={"<|endoftext|>"})

    def tok_decode(self, tokens):
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens)

    def _autocast(self):
        if self.device.type != "cuda" or self._dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", dtype=self._dtype)

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False,
                              override_bs=None, **kwargs):
        del disable_tqdm, kwargs
        batch_size = int(override_bs or self.batch_size)
        answers = []
        for start in range(0, len(requests), batch_size):
            batch = requests[start:start + batch_size]
            prepared = []
            for cache_key, context, continuation in batch:
                if not continuation:
                    raise ValueError("continuation must contain at least one token")
                if len(continuation) > self.max_length:
                    raise ValueError(
                        "continuation is longer than the model context window")
                # Keep every continuation token plus as much rightmost context as
                # fits. The final token is a target and is not fed to the model.
                sequence = (context + continuation)[-(self.max_length + 1):]
                inputs = sequence[:-1]
                if len(inputs) > self.max_length:
                    inputs = inputs[-self.max_length:]
                targets = sequence[-len(continuation):]
                prepared.append((cache_key, inputs, targets))

            max_input = max(len(item[1]) for item in prepared)
            input_ids = torch.full(
                (len(prepared), max_input), self.eot_token_id,
                dtype=torch.long, device=self.device)
            for row, (_, inputs, _) in enumerate(prepared):
                input_ids[row, :len(inputs)] = torch.tensor(
                    inputs, dtype=torch.long, device=self.device)

            with self._autocast():
                if (hasattr(self.model, "forward_hidden")
                        and hasattr(self.model, "lm_head")):
                    hidden = self.model.forward_hidden(input_ids)
                    score_hidden = []
                    for row, (_, inputs, targets) in enumerate(prepared):
                        end = len(inputs)
                        score_hidden.append(
                            hidden[row, end - len(targets):end])
                    packed_logits = self.model.lm_head(
                        torch.cat(score_hidden, dim=0))
                    selected_by_row = list(packed_logits.split(
                        [len(targets) for _, _, targets in prepared], dim=0))
                else:  # small contract-test models and legacy adapters
                    logits = self.model.forward_all_positions(input_ids)
                    selected_by_row = []
                    for row, (_, inputs, targets) in enumerate(prepared):
                        end = len(inputs)
                        selected_by_row.append(
                            logits[row, end - len(targets):end])
            for row, (cache_key, inputs, targets) in enumerate(prepared):
                selected = selected_by_row[row].float()
                log_probs = torch.log_softmax(selected, dim=-1)
                target_ids = torch.tensor(
                    targets, dtype=torch.long, device=self.device)
                score = float(log_probs.gather(
                    -1, target_ids[:, None]).sum())
                greedy = bool(torch.equal(selected.argmax(-1), target_ids))
                answer = (score, greedy)
                answers.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial(
                        "loglikelihood", cache_key, answer)
        return answers

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for request in requests:
            (string,) = request.args
            windows = list(map(
                utils.make_disjoint_window,
                utils.get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            ))
            scores = self._loglikelihood_tokens(
                [(None, context, continuation)
                 for context, continuation in windows],
                disable_tqdm=disable_tqdm)
            total = sum(score for score, _ in scores)
            results.append(total)
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), total)
        return results

    @torch.inference_mode()
    def generate_until(self, requests, disable_tqdm=False):
        del disable_tqdm
        outputs = []
        tokenizer_vocab = self.tokenizer.n_vocab
        for request in requests:
            context, generation_kwargs = request.args
            unsupported = set(generation_kwargs) - {
                "until", "max_gen_toks", "do_sample", "temperature",
            }
            if unsupported:
                raise ValueError(
                    "NanoGPTLM.generate_until does not support generation "
                    f"arguments: {sorted(unsupported)}")
            if generation_kwargs.get("do_sample", False):
                raise ValueError(
                    "NanoGPTLM.generate_until currently supports deterministic "
                    "greedy generation only (do_sample=False)")
            temperature = generation_kwargs.get("temperature", 0)
            if temperature not in (None, 0, 0.0, 1, 1.0):
                raise ValueError(
                    "temperature is only accepted for compatibility with "
                    "greedy generation; stochastic sampling is unsupported")
            stops = generation_kwargs.get("until") or []
            if isinstance(stops, str):
                stops = [stops]
            elif not isinstance(stops, (list, tuple)):
                raise TypeError("until must be a string, list, tuple, or None")
            max_new = int(generation_kwargs.get(
                "max_gen_toks", self.max_gen_toks))
            if not 0 <= max_new < self.max_length:
                raise ValueError(
                    f"max_gen_toks must be in [0, {self.max_length - 1}]")
            tokens = self.tok_encode(context) or [self.eot_token_id]
            tokens = tokens[-max(1, self.max_length - max_new):]
            idx = torch.tensor(tokens, dtype=torch.long,
                               device=self.device)[None, :]
            generated = []
            for _ in range(max_new):
                with self._autocast():
                    logits, _ = self.model(idx[:, -self.max_length:])
                next_token = int(logits[0, -1, :tokenizer_vocab].argmax())
                if next_token == self.eot_token_id:
                    break
                generated.append(next_token)
                idx = torch.cat((idx, torch.tensor(
                    [[next_token]], device=self.device)), dim=1)
                text = self.tok_decode(generated)
                if any(stop and stop in text for stop in stops):
                    break
            text = self.tok_decode(generated)
            stop_positions = [text.find(stop) for stop in stops if stop in text]
            if stop_positions:
                text = text[:min(stop_positions)]
            outputs.append(text)
            self.cache_hook.add_partial(
                "generate_until", (context, generation_kwargs), text)
        return outputs

from qlora import get_accelerate_model
import pickle as pkl
import os
from transformers import AutoTokenizer, LlamaTokenizer
import transformers
from typing import Dict
from tqdm import trange


DEVICE = "cuda:0"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_tokenizer(args, model):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,  # Fast tokenizer giving issues.
        tokenizer_type="llama"
        if "llama" in args.model_name_or_path
        else None,  # Needed for HF name change
    )

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print("Adding special tokens.")
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id
                    if model.config.pad_token_id != -1
                    else tokenizer.pad_token_id
                ),
            }
        )
    return tokenizer


class InferenceWrapper:
    def __init__(
        self,
        model_dir,
        step,
        template_path="/scratch/users/ruiqi-zhong/descriptive_clustering/templates/gpt_validator.txt",
    ):
        arg_path = os.path.join(model_dir, f"args.pkl")
        checkpoint_dir = os.path.join(model_dir, f"checkpoint-{step}/")
        with open(arg_path, "rb") as f:
            args = pkl.load(f)

        self.model = get_accelerate_model(args, checkpoint_dir=checkpoint_dir)
        self.tokenizer = get_tokenizer(args, self.model)
        with open(template_path, "r") as f:
            self.template = f.read()
        self.valid_responses = ["Yes", "No"]

        self.yes_no_idxes = [self.tokenizer.encode(r)[1] for r in self.valid_responses]

    def validate(self, hypothesis_text_dicts, bsize=8, verbose=True):
        prompts = [self.template.format(**d) for d in hypothesis_text_dicts]
        return self.get_yes_no(prompts, bsize, verbose=verbose)

    def get_yes_no_batch(self, prompts):

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.001,
            return_dict_in_generate=True,
            output_scores=True,
        )

        logits = outputs["scores"]
        results = []

        for i in range(len(prompts)):
            yes_no_logits = logits[0][i][self.yes_no_idxes].cpu().numpy()
            results.append(yes_no_logits[0] > yes_no_logits[1])
        return results

    def get_yes_no(self, prompts, batch_size, verbose=True):
        pbar = (
            trange(0, len(prompts), batch_size)
            if verbose
            else range(0, len(prompts), batch_size)
        )
        for i in pbar:
            batch_results = self.get_yes_no_batch(prompts[i : i + batch_size])
            for b in batch_results:
                yield b

    def get_generation(self, prompts, max_new_tokens, temperature=1.0):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_return_sequences=1,
            temperature=temperature,
            return_dict_in_generate=True,
        )
        output_sequences = outputs["sequences"]
        input_lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        returned_strings = [
            self.tokenizer.decode(
                output_sequences[i][input_lengths[i] :], skip_special_tokens=True
            )
            for i in range(len(prompts))
        ]
        return returned_strings


if __name__ == "__main__":
    model_dir = "output/verifier/"
    step = 6000

    validator = InferenceWrapper(model_dir, step)
    hypothesis_text_dicts = [
        {"hypothesis": "sounds happy", "text": "yeah!!!!"},
        {"hypothesis": "sounds happy", "text": "sh**t"},
        {"hypothesis": "sounds happy", "text": "f**k"},
    ] * 100

    for b in validator.validate(hypothesis_text_dicts, verbose=False):
        print(b)

    results = list(validator.validate(hypothesis_text_dicts, verbose=True))
    print(results)

    prompts = [
        "My name is Ruiqi Zhong, I'm from ",
        "My name is Jacob Steinhardt, I'm from ",
        "My name is Erik Jones, I'm from ",
    ]
    print(validator.get_generation(prompts, 10))

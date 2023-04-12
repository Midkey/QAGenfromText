import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


class LLM(object):
    def __init__(self,
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    ) -> None:
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

        self.prompter = Prompter(prompt_template)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        
        # unwind broken decapoda-research config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        if not load_8bit:
            self.model.half()  # seems to fix bugs for some users.

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def evaluate(
        self,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        output_list = []
        for s in generation_output.sequences:
            output = self.tokenizer.decode(s)
            output_list.append(output.split("### Response:")[1].strip())
        return output_list

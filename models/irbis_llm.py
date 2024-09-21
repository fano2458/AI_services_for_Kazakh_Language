from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch


class LLM():
    def __init__(self):
        self.model_name = "IrbisAI/Irbis-7b-v0.1"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def ask_llm(self, prompt):
        prompt = self.init_prompt(prompt)
        input_ids = self.tokenizer([prompt], return_tensors = "pt")["input_ids"].to("cuda")
        generation_config = GenerationConfig(
            temperature=0.6,
            repetition_penalty=1.15
        )

        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = []
        for s in generation_output.sequences:
            response.append(self.tokenizer.decode(s)) 

        return response

    def init_prompt(self, prompt):
        return f"Сұрақ: {prompt}\nЖауап: "

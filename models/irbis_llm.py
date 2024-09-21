from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

model_name = "IrbisAI/Irbis-7b-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Сұрақ: Шөп неге жасыл?\nЖауап: "

input_ids = tokenizer([prompt], return_tensors = "pt")["input_ids"].to("cuda")

generation_config = GenerationConfig(
    temperature=0.6,
    repetition_penalty=1.15
)
print("Generating...")
generation_output = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=2048,
    pad_token_id=tokenizer.eos_token_id,
)
for s in generation_output.sequences:
    print(tokenizer.decode(s)) # Өсімдіктер ауасыз өмір сүре алмайды, сондықтан олар жасыл түсті болады.

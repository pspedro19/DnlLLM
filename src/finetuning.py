import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

def load_model_and_tokenizer(model_name, bnb_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to be the same as the EOS token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def add_adopter_to_model(model):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    model = get_peft_model(model, peft_config)
    return model, peft_config

def set_hyperparameters():
    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        #report_to="wandb"
    )
    return training_arguments

def train_model(model, dataset, peft_config, tokenizer, training_arguments):
    tokenizer.padding_side = 'right'  # Ensure the padding side is set to 'right'
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=1024,  # Set the max sequence length
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    trainer.train()
    return trainer


def save_and_push_model(trainer, new_model_name):
    model_dir = os.path.join("..", "models", new_model_name)
    trainer.model.save_pretrained(model_dir)
    trainer.model.push_to_hub(new_model_name, use_temp_dir=False)

def main():
    model_name = "mistralai/Mistral-7B-v0.1"
    dataset_name = "mlabonne/guanaco-llama2-1k"
    new_model_name = "mistral_7b_guanaco"
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model, tokenizer = load_model_and_tokenizer(model_name, bnb_config)
    model, peft_config = add_adopter_to_model(model)
    training_arguments = set_hyperparameters()
    dataset = load_dataset(dataset_name, split="train")
    trainer = train_model(model, dataset, peft_config, tokenizer, training_arguments)
    save_and_push_model(trainer, new_model_name)

if __name__ == "__main__":
    main()

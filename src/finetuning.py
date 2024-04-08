import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

def load_model_and_tokenizer(model_name, bnb_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
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
        report_to="wandb"
    )
    return training_arguments

def train_model(model, dataset, peft_config, tokenizer, training_arguments):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=None,
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
    # Set up
    model_name = os.path.join("..", "models", "mistral", "pytorch", "7b-v0.1-hf", "1")
    dataset_name = "mlabonne/guanaco-llama2-1k"
    new_model_name = "mistral_7b_guanaco"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, bnb_config)

    # Add adopter to the model
    model, peft_config = add_adopter_to_model(model)

    # Set hyperparameters
    training_arguments = set_hyperparameters()

    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    # Train model
    trainer = train_model(model, dataset, peft_config, tokenizer, training_arguments)

    # Save and push model
    save_and_push_model(trainer, new_model_name)

    # Evaluation and inference can be performed as needed

if __name__ == "__main__":
    main()

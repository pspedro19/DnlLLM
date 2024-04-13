import os
import datetime
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfFolder
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch
import wandb

# Suppress specific UserWarnings
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")

def load_model_and_tokenizer(model_name, bnb_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
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

def set_hyperparameters(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
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
        report_to="wandb",
    )

def train_model(model, train_dataset, eval_dataset, peft_config, tokenizer, training_arguments):
    tokenizer.padding_side = 'right'
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=1024,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
    )
    trainer.train()
    return trainer

def save_and_push_model(trainer, model_name, output_dir, hf_token):
    model_path = os.path.join(output_dir, model_name)
    trainer.model.save_pretrained(model_path)
    trainer.tokenizer.save_pretrained(model_path)

    # Verify the presence of necessary files
    required_files = ['config.json', 'pytorch_model.bin']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        print("Missing files in the model directory:", ', '.join(missing_files))
    else:
        print("All necessary model files are saved successfully.")
    
    try:
        HfFolder.save_token(hf_token)
        trainer.model.push_to_hub(model_name)
        print("Model pushed to Hugging Face Hub successfully.")
    except Exception as e:
        print(f"Failed to save or push model: {e}")

def main():
    hf_token = input("Enter your Hugging Face token: ")
    wb_token = input("Enter your Weights & Biases token: ")
    wandb.login(key=wb_token)

    model_name = "mistralai/Mistral-7B-v0.1"
    dataset_name = "mlabonne/guanaco-llama2-1k"
    new_model_name = "DnlModel"
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", timestamp)
    dataset = load_dataset(dataset_name, split="train")
    train_dataset = dataset.train_test_split(test_size=0.1)["train"]
    eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

    try:
        model, tokenizer = load_model_and_tokenizer(model_name, bnb_config)
        model, peft_config = add_adopter_to_model(model)
        training_arguments = set_hyperparameters(output_dir)
        trainer = train_model(model, train_dataset, eval_dataset, peft_config, tokenizer, training_arguments)
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)
        wandb.log(eval_results)
        save_and_push_model(trainer, new_model_name, output_dir, hf_token)
    except Exception as e:
        print(f"An error occurred during training or evaluation: {e}")

    wandb.finish()

if __name__ == "__main__":
    main()

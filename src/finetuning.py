import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import DPOTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from accelerate import Accelerator
import torch
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def format_dataset(example, tokenizer):
    instruccion = example["instruccion"]
    input_text = example["input"]
    output_text = example["output"]

    # Tokenizar las instrucciones
    instruccion_encoding = tokenizer(instruccion, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    # Tokenizar el texto de entrada y salida
    input_encoding = tokenizer(input_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    output_encoding = tokenizer(output_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    # Asegurar que 'labels' sea una lista de enteros
    labels = output_encoding["input_ids"].squeeze().tolist()

    # Devolver un diccionario con las codificaciones y las etiquetas
    return {
        "instruccion": instruccion_encoding["input_ids"].squeeze(),
        "input_ids": input_encoding["input_ids"].squeeze(),
        "attention_mask": input_encoding["attention_mask"].squeeze(),
        "labels": labels
    }





def fine_tune_mistral(model_name, dataset_path, output_dir, epochs=1, batch_size=4, learning_rate=5e-5):
    # Initialize the Accelerator
    accelerator = Accelerator()
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess the dataset
    dataset = load_dataset("json", data_files=dataset_path, field="data")["train"]
    formatted_dataset = dataset.map(lambda x: format_dataset(x, tokenizer))


    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=10,
        evaluation_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(output_dir)

    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, "fine_tuned_model"))

    # Prepare the model for Qlora fine-tuning quantization
    model = prepare_model_for_kbit_training(model)
    qlora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    model = get_peft_model(model, qlora_config)

    # Save the model with Qlora quantization
    model.save_pretrained(os.path.join(model_dir, "qlora_quantized_model"))


if __name__ == "__main__":
    # Use relative paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.join(script_dir, '..')
    data_dir = os.path.join(project_dir, 'data')
    model_dir = os.path.join(project_dir, 'models')

    # Set environment variables based on the operating system
    if os.name == 'nt':  # Windows
        os.environ['DEBUGPY_LOG_DIR'] = os.path.join(os.environ['USERPROFILE'], '.vscode', 'extensions', 'ms-python.vscode-pylance-2024.3.1')
    else:  # Linux and other Unix-like systems
        os.environ['DEBUGPY_LOG_DIR'] = os.path.join(os.environ['HOME'], '.vscode', 'extensions', 'ms-python.vscode-pylance-2024.3.1')

    # Fine-tune the Mistral model
    fine_tune_mistral(
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_path=os.path.join(data_dir, "dataset.json"),
        output_dir=os.path.join(model_dir, "fine_tuned_model"),
        epochs=1,
        batch_size=8,
        learning_rate=5e-5,
    )

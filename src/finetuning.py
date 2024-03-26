import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

def format_dataset(example):
    return {
        "prompt": example["instruccion"] + " " + example["input"],
        "chosen": example["output"],
        "rejected": "Respuesta incorrecta."
    }

def fine_tune_mistral(model_name, dataset_path, output_dir, epochs=1, batch_size=8, learning_rate=5e-5):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load and preprocess the dataset
    dataset = load_dataset("json", data_files=dataset_path, field="data")["train"]
    formatted_dataset = dataset.map(format_dataset)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=10,
        evaluation_strategy="no",
        report_to="none",
    )

    # Initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
        beta=0.1,
    )

    # Fine-tune the model
    dpo_trainer.train()

    # Save the fine-tuned model
    dpo_trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    project_dir = r'C:\Users\pedro\Documents\ProyectoFinal\DnlLLM'
    data_dir = os.path.join(project_dir, 'data')
    model_dir = os.path.join(project_dir, 'models')

    os.environ['DEBUGPY_LOG_DIR'] = r'c:\Users\pedro\.vscode\extensions\ms-python.vscode-pylance-2024.3.1'
    os.add_dll_directory(r'C:\Users\pedro\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\bitsandbytes')

    # Fine-tune the Mistral model
    fine_tune_mistral(
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_path=os.path.join(data_dir, "dataset.json"),
        output_dir=os.path.join(model_dir, "fine_tuned_model"),
        epochs=1,
        batch_size=8,
        learning_rate=5e-5,
    )

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
    model = get_peft_model(model, qlora_config)

    # Save the model with Qlora quantization
    model.save_pretrained(os.path.join(model_dir, "qlora_quantized_model"))

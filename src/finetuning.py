import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from accelerate import Accelerator

def format_dataset(examples, tokenizer):
    # Listas para almacenar los resultados tokenizados
    instrucciones = []
    inputs = []
    outputs = []
    attention_masks = []
    labels = []

    # Iterar sobre cada ejemplo en la lista 'data'
    for example in examples["data"]:
        instruccion = example["instruccion"]
        input_text = example["input"]
        output_text = example["output"]

        # Tokenizar las instrucciones, el texto de entrada y salida con truncación y padding
        instruccion_encoding = tokenizer(instruccion, truncation=True, padding="max_length", max_length=512)
        input_encoding = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
        output_encoding = tokenizer(output_text, truncation=True, padding="max_length", max_length=128)

        # Agregar los resultados tokenizados a las listas correspondientes
        instrucciones.append(instruccion_encoding["input_ids"])
        inputs.append(input_encoding["input_ids"])
        attention_masks.append(input_encoding["attention_mask"])
        labels.append(output_encoding["input_ids"])

    # Devolver un diccionario con las listas de codificaciones y etiquetas
    return {
        "instruccion": instrucciones,
        "input_ids": inputs,
        "attention_mask": attention_masks,
        "labels": labels
    }

def fine_tune_mistral(model_name, dataset_path, output_dir, epochs=1, batch_size=4, learning_rate=5e-5):
    # Inicializar el Accelerator
    accelerator = Accelerator()

    # Cargar el tokenizador y el modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Establecer el token de padding
    tokenizer.pad_token = tokenizer.eos_token

    # Cargar y preprocesar el dataset
    dataset = load_dataset("json", data_files=dataset_path, field="data")["train"]
    formatted_dataset = dataset.map(lambda x: format_dataset({"data": [x]}, tokenizer), batched=False)

    # Definir los argumentos de entrenamiento
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
        load_best_model_at_end=False,
    )

    # Inicializar el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer,
    )

    # Fine-tuning del modelo
    trainer.train()

    # Guardar el modelo fine-tuned
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(output_dir)

if __name__ == "__main__":
    # Usar rutas relativas
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.join(script_dir, '..')
    data_dir = os.path.join(project_dir, 'data')
    model_dir = os.path.join(project_dir, 'models')

    # Fine-tuning del modelo Mistral
    fine_tune_mistral(
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_path=os.path.join(data_dir, "dataset.json"),
        output_dir=os.path.join(model_dir, "fine_tuned_model"),
        epochs=1,
        batch_size=8,
        learning_rate=5e-5,
    )

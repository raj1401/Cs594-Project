import torch
from datasets import load_dataset
from evaluate import load
from datasets import Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import numpy as np

# Load accuracy metric
accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=preds, references=labels)

class LogAccuracyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            acc = metrics.get("accuracy", None)
            print(f">>> Epoch {int(state.epoch)} - Accuracy: {acc:.4f}")

# 1. Load STS-B Dataset
dataset = load_dataset("glue", "stsb")
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

# Convert regression scores (0.0–5.0) to integer classes (0–5)
def discretize_label(example):
    example["label"] = int(round(example["label"]))
    return example

dataset = dataset.map(discretize_label)

def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset = tokenized_dataset.cast_column("labels", ClassLabel(num_classes=6))

# 2. BitsAndBytes Quantization Config (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 3. Load Model for Classification (output_dim=6)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-large-uncased",
    num_labels=6,  # Classification
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Prepare Model for QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. Set up Trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="epoch",
    num_train_epochs=10,
    learning_rate=1e-4,
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LogAccuracyCallback()]
)

# 6. Start Training
trainer.train()

# 7. Save the fine-tuned model
model.save_pretrained("bert-stsb-qlora-accuracy")
tokenizer.save_pretrained("bert-stsb-qlora-accuracy")

# 8. Evaluate the model
results = trainer.evaluate()
print(f"Final Validation Accuracy: {results['accuracy']:.4f}")

import torch
from datasets import load_dataset
from evaluate import load
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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Custom callback to log accuracy after each epoch
class LogAccuracyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_accuracy" in metrics:
            print(f">>> Epoch {int(state.epoch)} - Validation Accuracy: {metrics['eval_accuracy']:.4f}")

# 1. Load MNLI Dataset
dataset = load_dataset("glue", "mnli")
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

def tokenize_fn(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# 2. BitsAndBytes Quantization Config (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 3. Load Model with 3 output labels (MNLI)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-large-uncased",
    num_labels=3,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Prepare Model for QLoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],  # Layers in BERT to LoRA-ize
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
    eval_dataset=tokenized_dataset["validation_matched"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LogAccuracyCallback()]
)

# 6. Start Training
trainer.train()

# 7. Save the fine-tuned model
model.save_pretrained("bert-mnli-qlora")
tokenizer.save_pretrained("bert-mnli-qlora")

# 8. Evaluate the model
results = trainer.evaluate()
print(f"Final Validation Accuracy: {results['eval_accuracy']:.4f}")

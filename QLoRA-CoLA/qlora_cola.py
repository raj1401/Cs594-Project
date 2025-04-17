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

# Load Matthews correlation metric for CoLA
matthews_metric = load("matthews_correlation")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return matthews_metric.compute(predictions=predictions, references=labels)

# Custom callback to log correlation after each epoch
class LogCorrelationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_matthews_correlation" in metrics:
            print(f">>> Epoch {int(state.epoch)} - Validation Matthews Correlation: {metrics['eval_matthews_correlation']:.4f}")

# 1. Load CoLA Dataset
dataset = load_dataset("glue", "cola")
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

def tokenize_fn(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# 2. BitsAndBytes Quantization Config (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 3. Load Model with 4-bit Quantization
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-large-uncased",
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
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LogCorrelationCallback()]  # Update callback
)

# 6. Start Training
trainer.train()

# 7. Save the fine-tuned model
model.save_pretrained("bert-cola-qlora")
tokenizer.save_pretrained("bert-cola-qlora")

# 8. Evaluate the model
results = trainer.evaluate()
print(f"Final Validation Matthews Correlation: {results['eval_matthews_correlation']:.4f}")

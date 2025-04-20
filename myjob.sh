#!/bin/bash
#SBATCH --job-name=llm-finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  # request GPUs
#SBATCH --time=24:00:00

# Load necessary modules
module load python/3.8 cuda/11.2

# Set up environment
python -m venv llm-env
source llm-env/bin/activate
pip install torch transformers datasets

# Run fine-tuning
python - << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and prepare HellaSwag
dataset = load_dataset("hellaswag")
# ... preprocess data ...

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
EOF
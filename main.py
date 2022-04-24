import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from wechsel import WECHSEL, load_embeddings

source_tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")

target_tokenizer = source_tokenizer.train_new_from_iterator(
    load_dataset("oscar", "unshuffled_deduplicated_ko", split="train")["text"],
    vocab_size=len(source_tokenizer)
)

wechsel = WECHSEL(
    load_embeddings("en"),
    load_embeddings("ko"),
    bilingual_dictionary="korean"
)

target_embeddings, info = wechsel.apply(
    source_tokenizer,
    target_tokenizer,
    model.get_input_embeddings().weight.detach().numpy(),
)

model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)

num_gpu = torch.cuda.device_count()

data_collator = DataCollatorForLanguageModeling(
	tokenizer=target_tokenizer,
	mlm=False,
	)

train_dataset=load_dataset("oscar", "unshuffled_deduplicated_ko", split="train")


training_args = TrainingArguments(
    output_dir="./wechsel-gpt2-ko",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=800,
    warmup_ratio=0.1,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model()

# -*- coding: utf-8 -*-
# ===================================  MT5  ===================================
import math
import random
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader


# Helper class for splitting data into batches
class GreeklishDataset(Dataset):
    def __init__(self, source_encodings, target_encodings):
        assert len(source_encodings) == len(target_encodings)
        self.source_encodings = source_encodings
        # we don't need the mask for the target sequence
        self.target_encodings = target_encodings['input_ids']

    def __getitem__(self, idx):
        # don't forget we need both the idxs and the attention mask from the source
        source = {key: torch.tensor(val[idx]) for key, val in self.source_encodings.items()}
        target = torch.tensor(self.target_encodings[idx])
        return source, target

    def __len__(self):
        return len(self.target_encodings)


# path = "/content/"
load_path = "/content/drive/MyDrive/"
save_path = "/content/drive/MyDrive/ByT5_Greeklish/"

# import data
with open(load_path + "greek_europarl_training_100k.txt", "r", encoding="utf-8") as file:
    train_y = []
    for line in file:
        train_y.append(line)
with open(load_path + "greeklish_europarl_training_100k.txt", "r", encoding="utf-8") as file:
    train_x = []
    for line in file:
        train_x.append(line)

version = "small"

# choose training/val datasets
random.seed(12345)
random.shuffle(train_x)
random.seed(12345)
random.shuffle(train_y)

data_size = 12500

# 80/20 split
train_sources = train_x[: math.ceil(data_size * 0.8)]
train_targets = train_y[: math.ceil(data_size * 0.8)]
val_sources = train_x[math.ceil(data_size * 0.8): data_size]
val_targets = train_y[math.ceil(data_size * 0.8): data_size]

# ================ Fine-Tuning =================
model = T5ForConditionalGeneration.from_pretrained("google/byt5-{}".format(version))
tokenizer = AutoTokenizer.from_pretrained("google/byt5-{}".format(version))

# test
encoded_sample = tokenizer(train_y[0])
print(tokenizer.decode(encoded_sample['input_ids']))

# Padded to the length of the longest sentence
train_source_encodings = tokenizer(train_sources, padding=True)
train_target_encodings = tokenizer(train_targets, padding=True)
val_source_encodings = tokenizer(val_sources, padding=True)
val_target_encodings = tokenizer(val_targets, padding=True)

train_dataset = GreeklishDataset(train_source_encodings, train_target_encodings)
val_dataset = GreeklishDataset(val_source_encodings, val_target_encodings)

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 11
learning_rate = 5e-5
epochs = 33
accumulation_steps = 9

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print(len(train_loader))
print(len(val_loader))

# Optimizer & lr decay
optim = AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=11, gamma=0.5)  # decrease the lr to half every 11 epochs

training_losses = []
validation_losses = []

model.zero_grad(set_to_none=True)
print("Training started")
best_loss = None
for epoch in range(1, epochs + 1):
    print("\nEpoch {}/{}".format(epoch, epochs))
    train_loss = 0
    for i, (sources, targets) in enumerate(train_loader):
        # Print progress of epoch
        print("\r[{}{}] Batch {}/{}".format(math.ceil((i + 1) / len(train_loader) * 40) * "=",
                                            (40 - math.ceil((i + 1) / len(train_loader) * 40)) * " ", i + 1,
                                            len(train_loader)), end="")
        # optim.zero_grad()

        # Unpack the batch & push to device
        input_ids = sources['input_ids'].to(device)
        attention_mask = sources['attention_mask'].to(device)
        labels = targets.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        train_loss += loss  # For loss loging at the end of the epoch

        # Grad accumulation
        if (i / accumulation_steps == len(train_loader) / accumulation_steps) and (
                len(train_loader) % accumulation_steps != 0):
            loss = loss / (len(train_loader) % accumulation_steps)  # The last batches may not be enough for a full accumulation step
        else:
            loss = loss / accumulation_steps

        loss.backward()

        # if (i+1) % accumulation_steps == 0:
        if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:  # If the finally batches can't if a full accumulation step
            optim.step()
            model.zero_grad(set_to_none=True)

    # Validation
    model.zero_grad()
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for i, (sources, targets) in enumerate(val_loader):
            input_ids = sources['input_ids'].to(device)
            attention_mask = sources['attention_mask'].to(device)
            labels = targets.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            val_loss += loss

    # Calculate losses for the epoch
    epoch_train_loss = train_loss / len(train_loader)
    epoch_val_loss = val_loss / len(val_loader)

    training_losses.append(epoch_train_loss.item())
    validation_losses.append(epoch_val_loss.item())

    print(" Loss: (train){} (val){}".format(epoch_train_loss, epoch_val_loss))
    print("train: {}".format(training_losses))
    print("val: {}".format(validation_losses))

    # Save the models (if val loss has improved)
    if not best_loss:
        best_loss = epoch_val_loss

        tokenizer.save_pretrained(save_path + "ByT5_{}_{}_samples".format(version, data_size))
        model.save_pretrained(save_path + "ByT5_{}_{}_samples".format(version, data_size))
    elif (best_loss > epoch_val_loss):
        best_loss = epoch_val_loss

        tokenizer.save_pretrained(save_path + "ByT5_{}_{}_samples".format(version, data_size))
        model.save_pretrained(save_path + "ByT5_{}_{}_samples".format(version, data_size))

    scheduler.step()

# =============== Training Done ===============
# Translate a sentence
with torch.no_grad():
    model.eval()
    sample = "Ena paradeigma sta ellinika."
    print("\n{}".format(sample))
    encoded_sample = tokenizer(sample, return_tensors="pt").input_ids.to(device)
    output = model.generate(encoded_sample, max_length=200)
    print(tokenizer.decode(output[0]))


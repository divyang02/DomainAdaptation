import argparse
import constant as config
import torch
from util.dataset import read_dataset, sampling_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from trainer import Trainer
# from util.augment import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Define the dataset path and class number
Labeled_dataset_path = "./Dataset/Large_dataset/Val_large.csv"
Unlabeled_dataset_path = "./Dataset/Large_dataset/Unlabeled_large.csv"
save_path = "./save"

# Load the dataset
train_df = pd.read_csv(Labeled_dataset_path)
unlabeled_df = pd.read_csv(Unlabeled_dataset_path)

# Modifiy the labels from -1 to 0 since in bert model, target should contain indices in the range [0, nb_classes-1].
train_df["Helpful"] = train_df["Helpful"].apply(lambda x: 1 if x == 1 else 0)

# Define the training and validation sets
train_texts, train_labels = train_df["comments"].tolist(), train_df["Helpful"].tolist()
labeled_texts, remain_texts, labeled_labels, remain_labels = train_test_split(train_texts, train_labels, test_size=0.4,
                                                                              random_state=2023)
dev_texts, test_texts, dev_labels, test_labels = train_test_split(remain_texts, remain_labels, test_size=0.5,
                                                                  random_state=2023)

# Tokenizing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labeled_encodings = tokenizer(labeled_texts, truncation=True, padding=True)
labeled_dataset = Dataset(s, labeled_labels)

dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)
dev_dataset = Dataset(dev_encodings, dev_labels)

test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = Dataset(test_encodings, test_labels)

# Initialize the Bert model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num)

# Criterion & optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # or AdamW

trainer = Trainer(config, model, loss_function, optimizer, save_path, dev_dataset, test_dataset)

#Initial training
trainer.initial_train(labeled_dataset)

# load checkpoint
checkpoint_path = trainer.sup_path +'/checkpoint.pt'
checkpoint = torch.load(checkpoint_path)

del model, optimizer, trainer.model, trainer.optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.model = model
trainer.optimizer = optimizer

# eval supervised trained model
trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)

# self-training
trainer.self_train(labeled_dataset, unlabeled_dataset)

# eval semi-supervised trained model
checkpoint_path = trainer.ssl_path +'/checkpoint.pt'
checkpoint = torch.load(checkpoint_path)

del model, optimizer, trainer.model, trainer.optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config.class_num).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

trainer.model = model
trainer.optimizer = optimizer

trainer.evaluator.evaluate(trainer.model, trainer.test_loader, is_test=True)
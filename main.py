from src.resnet_transformer_model import ResNetTransformerEncoderDecoder
from src.dataset import get_datasets
from src.trainer import Trainer
from src.utils import CollateFn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from src.model_lstm import LSTMEncoderDecoder

import wandb

def main():
    train, val, _ = get_datasets()
    wandb.login(key='ключ от wandb')
    pad_idx = train.vocab.stoi["<PAD>"]
    batch_size = 64
    batch_size = 64
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=CollateFn(pad_idx)
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        collate_fn=CollateFn(pad_idx)
    )

    embed_size = 512
    num_layers = 3
    vocab_size = len(train.vocab)
    num_epochs = 20
    num_heads = 8
    learning_rate = 2e-5
    train_CNN = True
    beam_width = 3

    model = ResNetTransformerEncoderDecoder(
        embed_size=embed_size,
        num_heads=num_heads,
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=0.4,
        num_encoder_layers=3,
        train_CNN=train_CNN
    )

    criterion = nn.CrossEntropyLoss(ignore_index=train.vocab.stoi["<PAD>"], label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.03)

    train_dataset_len = len(train_loader.dataset)

    total_steps = (train_dataset_len // batch_size) * num_epochs
    total_steps = len(train_loader) * num_epochs

    warmup_steps = 500


    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
        last_epoch=-1
    )


    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        save_path="./save_dir",
        pad_idx=pad_idx,
        patience=5,
        max_gen_len=25,
        tokenizer=train.vocab,
        num_epochs=num_epochs,
        beam_width=beam_width
            )
    
    trainer.train()


if __name__ == "__main__":
    main()
import comet_ml
from src.resnet_lstm_model import LSTMEncoderDecoder
from src.dataset import get_datasets
from src.lstm_trainer import LSTMTrainer
from src.utils import CollateFn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def main():
    train, val, _ = get_datasets("Flickr8k")
    pad_idx = train.vocab.stoi["<PAD>"]
    start_idx = train.vocab.stoi.get("<START>", 1)
    end_idx = train.vocab.stoi.get("<END>", 2)
    
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
    hidden_size = 512
    num_layers = 2
    vocab_size = len(train.vocab)
    num_epochs = 20
    num_heads = 8
    learning_rate = 1e-4
    train_CNN = True
    beam_width = 3
    
    num_encoder_layers = 2
    
    dropout = 0.5

    model = LSTMEncoderDecoder(
        embed_size=embed_size,
        num_heads=num_heads,
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout,
        num_encoder_layers=num_encoder_layers,
        train_CNN=train_CNN
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx, 
        label_smoothing=0.05
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = min(1000, int(0.1 * total_steps))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
        last_epoch=-1
    )

    trainer = LSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        save_path="./save_dir_lstm",
        patience=7,
        pad_idx=pad_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        max_gen_len=25,
        tokenizer=train.vocab,
        num_epochs=num_epochs,
        beam_width=beam_width,
        grad_clip=1.0
    )
    
    trainer.train()


if __name__ == "__main__":
    main()

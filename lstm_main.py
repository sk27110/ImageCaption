import comet_ml
from src.resnet_lstm_model import LSTMEncoderDecoder  # предполагаемый путь к вашей новой модели
from src.dataset import get_datasets
from src.lstm_trainer import LSTMTrainer  # новый тренер
from src.utils import CollateFn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def main():
    # Загрузка данных
    train, val, _ = get_datasets("Flickr8k")
    pad_idx = train.vocab.stoi["<PAD>"]
    start_idx = train.vocab.stoi.get("<START>", 1)  # Проверяем наличие START токена
    end_idx = train.vocab.stoi.get("<END>", 2)      # Проверяем наличие END токена
    
    batch_size = 64  # Уменьшил batch_size для LSTM
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

    # Параметры модели LSTM
    embed_size = 512           # Размер эмбеддингов
    hidden_size = 512          # Размер скрытого состояния LSTM
    num_layers = 2             # Количество слоев LSTM (2-3 оптимально)
    vocab_size = len(train.vocab)
    num_epochs = 20            # LSTM может потребовать больше эпох
    num_heads = 8              # Для энкодера (трансформера) - можно оставить
    learning_rate = 1e-4       # Чуть выше, чем для трансформера
    train_CNN = True
    beam_width = 3             # LSTM хорошо работает с beam_width 3-7
    
    # Количество слоев трансформер-энкодера
    num_encoder_layers = 2     # Можно уменьшить до 1-2 слоев
    
    # Dropout для регуляризации (LSTM склонен к переобучению)
    dropout = 0.5

    # Создание модели
    model = LSTMEncoderDecoder(
        embed_size=embed_size,
        num_heads=num_heads,
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout,
        num_encoder_layers=num_encoder_layers,
        train_CNN=train_CNN
    )

    # Функция потерь (можно добавить label smoothing для LSTM)
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx, 
        label_smoothing=0.05  # LSTM склонен к переуверенности
    )
    
    # Оптимизатор - AdamW с умеренным weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,  # Меньше weight decay для LSTM
        betas=(0.9, 0.999)   # Стандартные беты
    )

    # Планировщик learning rate
    total_steps = len(train_loader) * num_epochs
    warmup_steps = min(1000, int(0.1 * total_steps))  # 10% или 1000 шагов
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
        last_epoch=-1
    )

    # Создание тренера
    trainer = LSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        save_path="./save_dir_lstm",  # Другой путь для LSTM модели
        patience=7,                   # Чуть больше patience для LSTM
        pad_idx=pad_idx,
        start_idx=start_idx,          # Добавляем start_idx
        end_idx=end_idx,              # Добавляем end_idx
        max_gen_len=25,
        tokenizer=train.vocab,
        num_epochs=num_epochs,
        beam_width=beam_width,
        grad_clip=1.0                 # Gradient clipping для стабильности
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
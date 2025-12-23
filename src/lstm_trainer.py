import comet_ml
import torch
import numpy as np
from tqdm.auto import tqdm
import os


class LSTMTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, save_path,
                 patience=5, pad_idx=0, start_idx=1, end_idx=2, max_gen_len=20, tokenizer=None, 
                 num_epochs=100, beam_width=3, grad_clip=1.0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.save_path = save_path
        self.patience = patience
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_gen_len = max_gen_len
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.beam_width = beam_width
        self.grad_clip = grad_clip
        
        # Пути для сохранения
        self.tokenizer_path = os.path.join(self.save_path, "tokenizer.pth")
        self.model_path = os.path.join(self.save_path, "best_model.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Перенос модели на устройство
        self.model.to(self.device)
        os.makedirs(self.save_path, exist_ok=True)

        # === Comet ML initialization ===
        self.experiment = comet_ml.Experiment(
            api_key="MbL2psOHT82Uc7ML5Cd7TSvmR",
            project_name="image_captioning",
            auto_metric_logging=False,
            auto_param_logging=False,
            auto_histogram_tensorboard_logging=False,
        )

        # Логируем гиперпараметры
        self.experiment.log_parameters({
            "model_type": "LSTM_Attention",
            "embed_size": model.embed_size,
            "num_layers": model.num_layers,
            "beam_width": beam_width,
            "patience": patience,
            "grad_clip": grad_clip,
            "max_gen_len": max_gen_len
        })

        # Сохраняем и логируем токенизатор
        vocab_state = {
            "itos": self.tokenizer.itos,
            "stoi": self.tokenizer.stoi,
            "freq_threshold": self.tokenizer.freq_threshold if hasattr(tokenizer, 'freq_threshold') else None,
            "special_tokens": {
                "pad": self.pad_idx,
                "start": self.start_idx,
                "end": self.end_idx,
                "unk": self.tokenizer.stoi.get("<UNK>", 3) if hasattr(tokenizer, 'stoi') else 3,
            }
        }
        torch.save(vocab_state, self.tokenizer_path)
        self.experiment.log_asset(self.tokenizer_path, file_name="tokenizer.pth")

    def _train_one_epoch(self, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        total_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            images, captions = batch
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Подготовка входных данных для teacher forcing
            # Вход: captions без последнего токена, цель: captions без первого токена
            tgt_input = captions[:, :-1]
            targets = captions[:, 1:]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, tgt_input)
            
            # Вычисление потерь
            # outputs shape: (batch_size, seq_len, vocab_size)
            # targets shape: (batch_size, seq_len)
            loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), 
                                 targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Оптимизация
            self.optimizer.step()
            
            # Обновление learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Логирование метрик
            current_lr = self.optimizer.param_groups[0]["lr"]
            batch_loss = loss.detach().item()
            total_loss += batch_loss
            
            # Логирование в Comet ML
            self.experiment.log_metrics({
                "batch_train_loss": batch_loss,
                "learning_rate": current_lr,
                "grad_norm": grad_norm.item()
            })
            
            # Обновление progress bar
            progress_bar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "lr": f"{current_lr:.6f}"
            })
        
        avg_loss = total_loss / total_batches
        return avg_loss

    def _validate(self, epoch):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        examples = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            
            for batch_idx, batch in enumerate(progress_bar):
                images, captions = batch
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                # Подготовка данных для валидации
                tgt_input = captions[:, :-1]
                targets = captions[:, 1:]
                
                # Forward pass для вычисления потерь
                outputs = self.model(images, tgt_input)
                loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), 
                                     targets.reshape(-1))
                total_loss += loss.item()
                
                # Собираем примеры предсказаний для первого батча
                if batch_idx == 0:
                    for i in range(min(20, images.size(0))):  # Берем 5 примеров
                        if i%5==0:
                            img_tensor = images[i].unsqueeze(0)
                            
                            # Greedy decoding
                            greedy_ids = self.model.generate(
                                img_tensor, 
                                max_len=self.max_gen_len,
                                start_token=self.start_idx,
                                end_token=self.end_idx
                            )[0]  # Берем первый (и единственный) элемент списка
                            
                            # Beam search decoding
                            beam_ids = self.model.generate_beam(
                                img_tensor,
                                beam_width=self.beam_width,
                                max_len=self.max_gen_len,
                                start_token=self.start_idx,
                                end_token=self.end_idx
                            )[0]  # Лучшая последовательность из beam search
                            
                            # Конвертируем токены в слова
                            greedy_tokens = []
                            for idx in greedy_ids:
                                if idx == self.end_idx:
                                    greedy_tokens.append("<END>")
                                    break
                                if hasattr(self.tokenizer, 'itos') and idx < len(self.tokenizer.itos):
                                    greedy_tokens.append(self.tokenizer.itos[idx])
                                else:
                                    greedy_tokens.append(f"[{idx}]")
                            
                            beam_tokens = []
                            for idx in beam_ids:
                                if idx == self.end_idx:
                                    beam_tokens.append("<END>")
                                    break
                                if hasattr(self.tokenizer, 'itos') and idx < len(self.tokenizer.itos):
                                    beam_tokens.append(self.tokenizer.itos[idx])
                                else:
                                    beam_tokens.append(f"[{idx}]")
                            
                            ground_truth_tokens = []
                            for idx in captions[i].cpu().numpy():
                                if idx == self.pad_idx:
                                    continue
                                if idx == self.end_idx:
                                    ground_truth_tokens.append("<END>")
                                    break
                                if hasattr(self.tokenizer, 'itos') and idx < len(self.tokenizer.itos):
                                    ground_truth_tokens.append(self.tokenizer.itos[idx])
                                else:
                                    ground_truth_tokens.append(f"[{idx}]")
                            
                            examples.append({
                                "image_idx": i,
                                "greedy": " ".join(greedy_tokens),
                                "beam": " ".join(beam_tokens),
                                "ground_truth": " ".join(ground_truth_tokens)
                            })
                            
                            # Останавливаемся после сбора нужного количества примеров
                            if len(examples) >= 5:
                                break
            
            avg_loss = total_loss / len(self.val_loader)
            
            # Логирование примеров в Comet ML
            for ex in examples:
                self.experiment.log_text(
                    text=ex["greedy"],
                    metadata={
                        "type": "greedy_prediction",
                        "epoch": epoch,
                        "image_idx": ex["image_idx"]
                    }
                )
                self.experiment.log_text(
                    text=ex["beam"],
                    metadata={
                        "type": "beam_prediction",
                        "epoch": epoch,
                        "image_idx": ex["image_idx"]
                    }
                )
                self.experiment.log_text(
                    text=ex["ground_truth"],
                    metadata={
                        "type": "ground_truth",
                        "epoch": epoch,
                        "image_idx": ex["image_idx"]
                    }
                )
            
            return avg_loss, examples

    def train(self):
        """Основной цикл обучения"""
        print(f"Начинаем обучение на устройстве: {self.device}")
        print(f"Размер словаря: {self.model.vocab_size}")
        print(f"Размер эмбеддинга: {self.model.embed_size}")
        print(f"Количество слоев LSTM: {self.model.num_layers}")
        
        best_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Эпоха {epoch}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Обучение
            train_loss = self._train_one_epoch(epoch)
            
            # Валидация
            val_loss, examples = self._validate(epoch)
            
            # Логирование метрик эпохи
            self.experiment.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            }, epoch=epoch)
            
            # Вывод результатов
            print(f"\nРезультаты эпохи {epoch}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            
            # Вывод примеров
            print(f"\nПримеры предсказаний (из валидации):")
            for i, ex in enumerate(examples):
                print(f"\nПример {i+1}:")
                print(f"  Greedy:  {ex['greedy']}")
                print(f"  Beam:    {ex['beam']}")
                print(f"  Ground:  {ex['ground_truth']}")
            
            # Сохранение лучшей модели
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                
                # Сохраняем модель
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "model_args": {
                        "embed_size": self.model.embed_size,
                        "num_layers": self.model.num_layers,
                        "vocab_size": self.model.vocab_size,
                        "num_encoder_layers": self.model.num_encoder_layers,
                        "num_heads": self.model.num_heads,
                        "dropout": self.model.dropout,
                        "train_CNN": self.model.train_CNN
                    }
                }, self.model_path)
                
                print(f"\n✓ Сохранена лучшая модель с Val Loss: {val_loss:.4f}")
                
                # Логируем модель в Comet ML
                self.experiment.log_model(
                    "best_lstm_model",
                    self.model_path,
                    metadata={"val_loss": val_loss, "epoch": epoch}
                )
            else:
                wait += 1
                print(f"\nVal Loss не улучшился. Патience: {wait}/{self.patience}")
            
            # Early stopping
            if wait >= self.patience:
                print(f"\nEarly stopping на эпохе {epoch}")
                break
        
        # Финальные логи
        print(f"\n{'='*60}")
        print(f"Обучение завершено!")
        print(f"Лучшая модель на эпохе {best_epoch} с Val Loss: {best_val_loss:.4f}")
        print(f"Модель сохранена в: {self.model_path}")
        print(f"{'='*60}")
        
        # Завершаем эксперимент Comet ML
        self.experiment.log_parameter("best_epoch", best_epoch)
        self.experiment.log_parameter("best_val_loss", best_val_loss)
        self.experiment.end()
        
        return best_val_loss

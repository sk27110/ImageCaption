import wandb
import torch
import numpy as np
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, save_path,
                 patience=5, pad_idx=0, end_idx=2, max_gen_len=20, tokenizer=None, num_epochs=100, beam_width = 3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.save_path = save_path
        self.patience = patience
        self.pad_idx = pad_idx
        self.end_idx = end_idx
        self.max_gen_len = max_gen_len
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.beam_width = beam_width
        self.tokenizer_path = os.path.join(self.save_path, "tokenizer.pth")
        self.model_path = os.path.join(self.save_path, "best_model.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        os.makedirs(self.save_path, exist_ok=True)

        wandb.init(project="image_captioning")

        vocab_state = {
            "itos": self.tokenizer.itos,
            "stoi": self.tokenizer.stoi,
            "freq_threshold": self.tokenizer.freq_threshold,
            "special_tokens": {
                "pad": self.tokenizer.stoi["<PAD>"],
                "start": self.tokenizer.stoi["<START>"],
                "end": self.tokenizer.stoi["<END>"],
                "unk": self.tokenizer.stoi["<UNK>"],
            }
        }
        torch.save(vocab_state, self.tokenizer_path)
        artifact = wandb.Artifact('tokenizer', type='tokenizer')
        artifact.add_file(self.tokenizer_path)
        wandb.log_artifact(artifact)


    def _get_tgt_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device), diagonal=1)

    def _get_padding_mask(self, captions):
        return (captions == self.pad_idx).to(self.device)


    def _train_one_epoch(self):
        self.model.train()
        train_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            images, captions = batch
            images = images.to(self.device)
            captions = captions.to(self.device)

            tgt_input = captions[:, :-1]
            targets = captions[:, 1:]

            tgt_mask = self._get_tgt_mask(tgt_input.shape[1])
            tgt_key_padding_mask = self._get_padding_mask(tgt_input)


            self.optimizer.zero_grad()

            outputs = self.model(images, tgt_input, tgt_mask, tgt_key_padding_mask)
            loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            wandb.log({
                "batch_train_loss": loss.detach().item(),
                "grad_norm": grad_norm.item(),
                "learning_rate": current_lr
            })
            train_loss += loss.detach().item()

        avg_train_loss = train_loss / len(self.train_loader)
        return avg_train_loss


    def _validate(self):
        self.model.eval()
        val_loss = 0
        examples = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                images, captions = batch
                images = images.to(self.device)
                captions = captions.to(self.device)

                tgt_input = captions[:, :-1]
                targets = captions[:, 1:]

                tgt_mask = self._get_tgt_mask(tgt_input.shape[1])
                tgt_key_padding_mask = self._get_padding_mask(tgt_input)

                outputs = self.model(images, tgt_input, tgt_mask, tgt_key_padding_mask)
                loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
                val_loss += loss.detach().item()

                if batch_idx < 1:
                    for i in range(min(20, images.size(0))):
                        if i%5==0:
                            gen_ids = self.model.generate(images[i].unsqueeze(0), max_len=self.max_gen_len,
                                                           start_token=self.tokenizer.stoi["<START>"],
                                                           end_token=self.end_idx)[0]
                            gen_ids_beam = self.model.generate_beam(
                                images[i].unsqueeze(0), 
                                max_len=self.max_gen_len,
                                beam_width = self.beam_width,                          
                                start_token=self.tokenizer.stoi["<START>"],
                                end_token=self.end_idx
                            )
                            gen_beam_tokens = [self.tokenizer.itos[idx] for idx in gen_ids_beam if idx != self.tokenizer.stoi["<PAD>"]]
                            gen_tokens = [self.tokenizer.itos[idx] for idx in gen_ids if idx != self.tokenizer.stoi["<PAD>"]]
                            true_tokens = [self.tokenizer.itos[idx] for idx in captions[i].cpu().numpy() if idx != self.tokenizer.stoi["<PAD>"]]
    
                            examples.append({
                                "beam_prediction": " ".join(gen_beam_tokens),
                                "prediction": " ".join(gen_tokens),
                                "ground_truth": " ".join(true_tokens)
            
                            })

            log_dict = {}
            for idx, ex in enumerate(examples):
                log_dict[f"example_{idx}_beam_prediction"] = ex["pbeam_rediction"]
                log_dict[f"example_{idx}_prediction"] = ex["prediction"]
                log_dict[f"example_{idx}_ground_truth"] = ex["ground_truth"]

            wandb.log(log_dict)

        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss, examples

    def train(self):
        best_val = np.inf
        wait = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_one_epoch()
            val_loss, examples = self._validate()

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save({
                "model_state_dict": self.model.state_dict(),
                "model_class": "EncoderDecoder",
                "model_args": {
                    "embed_size": self.model.embed_size,
                    "num_heads": self.model.num_heads,
                    "num_layers": self.model.num_layers,
                    "num_encoder_layers": self.model.num_encoder_layers,
                    "vocab_size": self.model.vocab_size,
                    "dropout": self.model.dropout,
                    "train_CNN": self.model.train_CNN
                }
            }, self.model_path)
                artifact = wandb.Artifact('my_model', type='model')
                artifact.add_file(self.model_path)
                wandb.log_artifact(artifact)
                best_checkpoint = self.model_path
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            for idx, ex in enumerate(examples):
                log_dict[f"example_{idx}_prediction"] = ex["prediction"]
                log_dict[f"example_{idx}_beam_prediction"] = ex["beam_prediction"]
                log_dict[f"example_{idx}_ground_truth"] = ex["ground_truth"]

            wandb.log(log_dict)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            for ex in examples:
                print(f"Pred: {ex['prediction']}")
                print(f"Beam pred: {ex['beam_prediction']}")
                print(f"GT  : {ex['ground_truth']}")
                print("-"*40)

        print(f"Training finished. Best checkpoint saved at: {best_checkpoint}")
        wandb.finish()
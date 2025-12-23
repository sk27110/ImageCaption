import torch
import torch.nn as nn
import torchvision.models as models
import math


class Encoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False, num_encoder_layers=3, num_heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.train_CNN = train_CNN
        resnet = models.resnet50(weights="DEFAULT")

        for param in resnet.parameters():
            param.requires_grad = False

        if train_CNN:
            for param in resnet.layer3.parameters():
                param.requires_grad = True
            for param in resnet.layer4.parameters():
                param.requires_grad = True

        modules = list(resnet.children())[:-2]
        
        self.cnn = nn.Sequential(*modules)
        self.conv_proj = nn.Conv2d(2048, embed_size, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pos_encoder = PositionalEncoding(embed_size, max_len=49)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True  
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        nn.init.xavier_uniform_(self.conv_proj.weight)
        nn.init.zeros_(self.conv_proj.bias)

    def forward(self, images):
        features = self.cnn(images)
        features = self.adaptive_pool(features) 
        features = self.conv_proj(features) 
        features = self.relu(features)
        features = self.dropout(features)

        batch_size = features.size(0)
        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)
        features = self.pos_encoder(features.transpose(0, 1)).transpose(0, 1)
        features = self.transformer_encoder(features) 

        return features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Decoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size, dropout):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(self.hidden_size * 2, vocab_size)  # Concat hidden and context
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.normal_(self.embed.weight, mean=0, std=0.01)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features, captions):
        batch_size, seq_len = captions.shape
        hidden = self.init_hidden(batch_size, features)
        outputs = []

        for t in range(seq_len):
            input_token = captions[:, t]
            embeddings = self.embed(input_token)
            embeddings = self.dropout(embeddings)
            lstm_input = embeddings.unsqueeze(1)  # (batch, 1, embed)
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            h = lstm_out.squeeze(1)  # (batch, hidden)
            attn_scores = torch.bmm(features, h.unsqueeze(2)).squeeze(2)  # (batch, 49)
            attn_alpha = nn.functional.softmax(attn_scores, dim=1)
            context = torch.bmm(attn_alpha.unsqueeze(1), features).squeeze(1)  # (batch, embed)
            merged = torch.cat((h, context), dim=1)
            logit = self.linear(merged)
            outputs.append(logit)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq, vocab)
        return outputs

    def init_hidden(self, batch_size, features):
        device = features.device
        mean_features = features.mean(dim=1)
        h = mean_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = mean_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return (h.to(device), c.to(device))

    def step(self, input_token, features, hidden):
        """Один шаг декодера для генерации"""
        embeddings = self.embed(input_token)
        embeddings = self.dropout(embeddings)
        lstm_input = embeddings.unsqueeze(1)  # (batch, 1, embed)
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        h = lstm_out.squeeze(1)  # (batch, hidden)
        attn_scores = torch.bmm(features, h.unsqueeze(2)).squeeze(2)  # (batch, 49)
        attn_alpha = nn.functional.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_alpha.unsqueeze(1), features).squeeze(1)  # (batch, embed)
        merged = torch.cat((h, context), dim=1)
        logit = self.linear(merged)
        log_probs = torch.log_softmax(logit, dim=-1)
        return log_probs, hidden


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, vocab_size, num_layers, dropout=0.4, 
                 num_encoder_layers=3, train_CNN=False):
        super().__init__()
        self.encoder = Encoder(embed_size, train_CNN=train_CNN, 
                              num_encoder_layers=num_encoder_layers, 
                              num_heads=num_heads, dropout=dropout)
        self.decoder = Decoder(embed_size, num_heads, num_layers, vocab_size, dropout)
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.train_CNN = train_CNN

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate(self, images, max_len=50, start_token=1, end_token=2):
        """Greedy decoding"""
        with torch.no_grad():
            batch_size = images.size(0)
            features = self.encoder(images)
            hidden = self.decoder.init_hidden(batch_size, features)
            generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=images.device)
            input_token = generated[:, 0]

            for _ in range(max_len):
                log_probs, hidden = self.decoder.step(input_token, features, hidden)
                next_token = log_probs.argmax(dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)
                input_token = next_token.squeeze(1)

                if torch.all(next_token == end_token):
                    break

            return [seq.tolist() for seq in generated]

    def generate_beam(self, images, beam_width=3, max_len=50, start_token=1, end_token=2):
        """Beam search decoding - адаптированная версия из трансформерного кода"""
        with torch.no_grad():
            batch_size = images.size(0)
            device = images.device
            
            # Получаем фичи от энкодера
            features = self.encoder(images)  # (batch_size, 49, embed_size)
            
            # Для каждого примера в батче запускаем beam search
            all_captions = []
            
            for i in range(batch_size):
                # Берем фичи для текущего изображения
                img_features = features[i:i+1]  # (1, 49, embed_size)
                
                # Инициализируем начальную последовательность
                start_seq = torch.tensor([[start_token]], dtype=torch.long, device=device)
                
                # Инициализируем начальное скрытое состояние
                init_hidden = self.decoder.init_hidden(1, img_features)
                
                # Получаем log probabilities для первого шага
                log_probs, hidden = self.decoder.step(
                    torch.tensor([start_token], device=device), 
                    img_features, 
                    init_hidden
                )
                
                # Выбираем топ-K кандидатов для первого шага
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=-1)
                
                # Инициализируем лучи
                beams = []
                for k in range(beam_width):
                    sequence = [start_token, top_indices[0, k].item()]
                    score = top_log_probs[0, k].item()
                    # Для каждого луча нужно свое скрытое состояние
                    # Придется пересчитать hidden для каждого луча
                    _, beam_hidden = self.decoder.step(
                        torch.tensor([start_token], device=device), 
                        img_features, 
                        init_hidden
                    )
                    beams.append({
                        'sequence': sequence,
                        'score': score,
                        'hidden': beam_hidden,
                        'ended': False
                    })
                
                # Продолжаем расширять лучи
                for step in range(2, max_len):
                    candidates = []
                    
                    # Если все лучи закончились, останавливаемся
                    if all(beam['ended'] for beam in beams):
                        break
                    
                    for beam_idx, beam in enumerate(beams):
                        if beam['ended']:
                            # Добавляем законченный луч в кандидаты без изменений
                            candidates.append(beam)
                            continue
                        
                        # Последний токен в текущей последовательности
                        last_token = torch.tensor([beam['sequence'][-1]], device=device)
                        
                        # Делаем шаг для этого луча
                        log_probs, new_hidden = self.decoder.step(
                            last_token, 
                            img_features, 
                            beam['hidden']
                        )
                        
                        # Добавляем padding token (обычно 0) в конец, если нужно
                        log_probs = log_probs.squeeze(0)  # (vocab_size,)
                        
                        # Выбираем топ-K продолжений для этого луча
                        top_log_probs_step, top_indices_step = log_probs.topk(beam_width)
                        
                        for k in range(beam_width):
                            new_sequence = beam['sequence'] + [top_indices_step[k].item()]
                            new_score = beam['score'] + top_log_probs_step[k].item()
                            
                            # Проверяем, закончилась ли последовательность
                            ended = (top_indices_step[k].item() == end_token)
                            
                            candidates.append({
                                'sequence': new_sequence,
                                'score': new_score,
                                'hidden': new_hidden if not ended else beam['hidden'],
                                'ended': ended
                            })
                    
                    # Сортируем кандидатов по score и выбираем топ-K
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    beams = candidates[:beam_width]
                
                # Выбираем лучший луч для этого изображения
                best_beam = beams[0]
                all_captions.append(best_beam['sequence'])
            
            return all_captions

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
        self.hidden_size = embed_size  # Set hidden size equal to embed size for simplicity
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
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


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, embed_size, num_heads, vocab_size, num_layers, dropout=0.4, num_encoder_layers=3, train_CNN = False):
        super().__init__()
        self.encoder = Encoder(embed_size, train_CNN=train_CNN, num_encoder_layers=num_encoder_layers, num_heads=num_heads, dropout=dropout)
        self.decoder = Decoder(embed_size, num_heads, num_layers, vocab_size, dropout)

        self.embed_size = embed_size
        self.vocab_size = vocab_size

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate(self, images, max_len=50, start_token=1, end_token=2):
        with torch.no_grad():
            batch_size = images.size(0)
            features = self.encoder(images)
            hidden = self.decoder.init_hidden(batch_size, features)
            generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=images.device)
            input_token = generated[:, 0]

            for _ in range(max_len):
                embeddings = self.decoder.embed(input_token)
                embeddings = self.decoder.dropout(embeddings)
                lstm_input = embeddings.unsqueeze(1)
                lstm_out, hidden = self.decoder.lstm(lstm_input, hidden)
                h = lstm_out.squeeze(1)
                attn_scores = torch.bmm(features, h.unsqueeze(2)).squeeze(2)
                attn_alpha = nn.functional.softmax(attn_scores, dim=1)
                context = torch.bmm(attn_alpha.unsqueeze(1), features).squeeze(1)
                merged = torch.cat((h, context), dim=1)
                logit = self.decoder.linear(merged)
                next_token = logit.argmax(dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)
                input_token = next_token.squeeze(1)

                if torch.all(next_token == end_token):
                    break

            return [seq.tolist() for seq in generated]
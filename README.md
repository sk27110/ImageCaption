# Image captioning

Команда: \
Кочетков Александр, 3 курс Падии \
Хорошенко Дмитрий, 3 курс Падии \

### Струкрура проекта

├── src/ # Исходный код проекта \
│ ├── init.py # Пакетный файл Python \
│ ├── clip_transformer_model.py # Модель  CLIP + Transformer \
│ ├── dataset.py # Класс для загрузки и обработки данных \
│ ├── lstm_trainer.py M Тренировка модели LSTM \
│ ├── resnet_lstm_model.py # Модель ResNet + LSTM \
│ ├── resnet_transformer_model.py # Модель ResNet + Transformer \
│ ├── transformer_trainer.py M Тренировка Transformer модели \
│ ├── utils.py # Вспомогательные функции \
│ └── lstm_main.py # Главный скрипт для LSTM \
│\
├── README.md M Документация проекта \
├── requirements.txt # Зависимости Python \
├── test_clip_transformer.ipynb # Тестирование CLIP + Transformer \
├── test_lstm.ipynb # Тестирование LSTM модели \
├── test_resnet_transformer.ipynb # Тестирование ResNet + Transformer \
└── transformer_main.py # Главный скрипт для Transformer \

### Датасет
[flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

Корпус имеет структуру
```
image, caption
```
Для каждого изображения в корпусе по 5 различных описаний. 
### Архитектура моделей

1) ResNet + Transformer
2) ResNet + LSTM + Attention
3) CLIP + Transformer


### How to use

Установка зависимости:
```
pip install -r requirements.txt
```

Скачивание токенизатора:
```
python -m spacy download en_core_web_sm 
```


Запусу обучения:
```
python model_type_main.py
```

Запуск инференса:
```
gradio app.py
```

### Результаты
1) ResNet + Transformer: 
```
    BLEU-4 :  0.229 
    METEOR : 0.232 
    ROUGE-L: 0.477 
    CIDEr  : 0.613 
```
2) ResNet + LSTM: 
```
    BLEU-4 : 0.194 
    METEOR : 0.216 
    ROUGE-L: 0.461 
    CIDEr  : 0.516 
```
3) CLIP + Transformer: 
```
    BLEU-4 : 0.253 
    METEOR : 0.245 
    ROUGE-L: 0.501 
    CIDEr  : 0.684 
```


# Image captioning

### Датасет
[flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

### Архитектура
ResNet50 (feature map с 4-ого слоя) + Transformer

### How to use

```
pip install -r requirements.txt
```

```
python main.py()
```

### Результаты
Графики обучения есть в HW5.pdf или [тут](https://api.wandb.ai/links/sashakoch27-/cjg0jcme)
Примеры генерации есть в log-ах для каждой эпохи

Метрики:

    BLEU-4: 0.179
    METEOR: 0.217
    ROUGE-L: 0.456
    CIDEr: 0.534

### Возможные улучшения

1) Добавить beam search
2) Заменить ResNet50 на ResNet101 или ViT модель
3) Применить современные методы обучения, в которых мне пока лень разбираться
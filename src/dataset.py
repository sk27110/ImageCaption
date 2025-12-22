from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import kagglehub
from io import StringIO
import spacy
import torch
import torchvision.transforms as T



val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.9, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomGrayscale(p=0.05),
    T.GaussianBlur(kernel_size=3),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



class Vocabulary:
    def __init__(self, freq_threshold=4):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.spacy_eng = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, df, root_path ,freq_threshold=4, transform=None, vocab=None, return_img_idx = False):
        self.df = df.reset_index(drop=True)
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold=freq_threshold)
            self.vocab.build_vocabulary(self.df['caption'].tolist())
        else:
            self.vocab = vocab
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        caption = row['caption']
        img_path = os.path.join(self.root_path, "Images", row['image'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(numericalized_caption)



class FlickrTestDataset(Dataset):
    def __init__(self, df_grouped, root_path, transform=None):
        self.df = df_grouped.reset_index(drop=True)
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row['image']
        captions = row['captions']

        img_path = os.path.join(self.root_path, "Images", img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)


        return image, captions, img_name


def get_datasets(dataset_type='Flickr8k'):
    if dataset_type=='Flickr8k':
        path = kagglehub.dataset_download("adityajn105/flickr8k")
    if dataset_type=='Flickr30k':
        path = kagglehub.dataset_download("adityajn105/flickr30k")

    print("Path to dataset files:", path)

    df = pd.read_csv(StringIO(open(path + "/captions.txt").read()))

    all_images = df["image"].unique()

    train_imgs, tmp_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(tmp_imgs, test_size=0.5, random_state=42)

    df_train = df[df["image"].isin(train_imgs)].reset_index(drop=True)
    df_val   = df[df["image"].isin(val_imgs)].reset_index(drop=True)

    train_dataset = FlickrDataset(df_train, path, 4, train_transform)
    val_dataset = FlickrDataset(df_val, path, 4, val_transform, train_dataset.vocab)
    df_test_grouped = (
        df[df["image"].isin(test_imgs)]
        .groupby("image")["caption"]
        .apply(list)
        .reset_index(name="captions") 
    )

    assert all(len(captions) == 5 for captions in df_test_grouped["captions"])

    test_dataset = FlickrTestDataset(df_test_grouped, path, val_transform)

    return train_dataset, val_dataset, test_dataset
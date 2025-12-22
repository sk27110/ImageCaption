import torch


class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        targets_list = [item[1] for item in batch]

        imgs = torch.cat([img.unsqueeze(0) for img in images], dim=0)

        targets = torch.nn.utils.rnn.pad_sequence(
            targets_list,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return imgs, targets
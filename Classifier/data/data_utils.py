import os
import pandas as pd
import datasets
import torch
from datasets import load_dataset, Dataset
from datasets import disable_caching
disable_caching()

def collect_fn(batch_data):
    batch = {key:[] for key in batch_data[0].keys()}
    input_lens = list(map(len, [ele['input_ids'] for ele in batch_data]))
    max_input_len = max(input_lens)
    for key, pad in zip(batch.keys(), [-1, 0, 0, 1]):
        for idx, ele in enumerate(batch_data):
            if pad >= 0:
                batch[key].append(ele[key] + [pad] * (max_input_len - input_lens[idx]))
            else:
                batch[key].append(ele[key])

        batch[key] = torch.LongTensor(batch[key])
    return batch

class Financial_Dataset(Dataset):
    def __init__(self, args, tokenizer, data_path, split='train'):
        dataset = load_dataset("csv", data_files=data_path, split=split, encoding='GBK')
        dataset = dataset.map(
            lambda x: self.convert_to_id(x, tokenizer, maxlen=args.max_length), num_proc=8, batched=False)

        self.dataset = dataset
        print(self.dataset)

    def convert_to_id(self, example, tokenizer, maxlen):
        outputs = tokenizer(example['Text'], max_length=maxlen, truncation=True, padding=True)
        example['input_ids'] = outputs['input_ids']
        example['attention_mask'] = outputs['attention_mask']
        example['token_type_ids'] = outputs['token_type_ids']
        del example['Text']
        return example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


if __name__ == '__main__':
    data_path = './train.CSV'
    dataset = Financial_Dataset(data_path, None)
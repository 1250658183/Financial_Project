import os
import pandas as pd
import datasets
import torch
from datasets import load_dataset, Dataset
from datasets import disable_caching
disable_caching()

def collect_fn(batch_data):
    batch = {key:[] for key in sorted(list(batch_data[0].keys()))}
    input_lens = list(map(len, [ele['input_ids'] for ele in batch_data]))
    max_input_len = max(input_lens)
    for key, pad in zip(batch.keys(), [0, 0, -1, 1]):
        for idx, ele in enumerate(batch_data):
            if pad >= 0:
                batch[key].append(ele[key] + [pad] * (max_input_len - input_lens[idx]))
            else:
                batch[key].append(ele[key])
        batch[key] = torch.LongTensor(batch[key])
    return batch

class Financial_Dataset(Dataset):
    def __init__(self, args, tokenizer, data_path, split='train'):
        if split == 'train':
            dataset = load_dataset("csv", data_files=data_path, split=split, encoding='utf-8', sep="	")
            dataset = dataset.map(lambda x: self.convert_to_id(x, tokenizer, maxlen=args.max_length), num_proc=8, batched=False)
        else:
            dataset = load_dataset("csv", data_files=data_path, split='train', encoding='gbk', sep="	")
            dataset = dataset.rename_column('reason', 'Text')
            dataset = dataset.map(lambda x, idx: self.convert_to_id(x, tokenizer, maxlen=args.max_length, idx=idx), with_indices=True, num_proc=1, batched=False)
        self.dataset = dataset
        print(self.dataset)

    def convert_to_id(self, example, tokenizer, maxlen, idx=None):
        outputs = tokenizer(example['Text'], max_length=maxlen, truncation=True, padding=True)
        # print(idx, example['Text'])
        example['input_ids'] = outputs['input_ids']
        example['attention_mask'] = outputs['attention_mask']
        example['token_type_ids'] = outputs['token_type_ids']
        del example['Text']
        if 'id' in example:
            del example['id']
        if idx is not None: example['labels'] = idx
        return example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


if __name__ == '__main__':
    data_path = './train.CSV'
    dataset = Financial_Dataset(data_path, None)
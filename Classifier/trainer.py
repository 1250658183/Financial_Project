# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '../')))

import argparse
from collections import namedtuple
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

import torch
from torch import optim
import logging, json, re, math, shutil
from transformers import BertForSequenceClassification, BertConfig, BertTokenizerFast, BertTokenizer
from data.data_utils import Financial_Dataset, collect_fn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

class Financial_Task(pl.LightningModule):
    def __init__(self, args):
        super(Financial_Task, self).__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        else:
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        # Model
        self.config = BertConfig.from_pretrained(args.bert_path)
        setattr(self.config, 'num_labels', args.num_labels)
        if not args.TestModel:
            self.model = BertForSequenceClassification.from_pretrained(args.bert_path, config=self.config)
        else:
            self.model = BertForSequenceClassification(self.config)
        format = '%(asctime)s - %(name)s - %(message)s'
        if not args.TestModel:
            logging.basicConfig(format=format, filename=os.path.join(self.args.save_path, "eval_result_log.txt"), level=logging.INFO)
            self.result_logger = logging.getLogger(__name__)
            self.result_logger.setLevel(logging.INFO)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), lr=self.args.lr, eps=self.args.adam_epsilon, )

        # num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        if isinstance(self.args.gpus, int): num_gpus = self.args.gpus if self.args.gpus >= 0 else torch.distributed.get_world_size()
        elif isinstance(self.args.gpus, str): num_gpus = len(self.args.gpus.split(','))
        elif isinstance(self.args.gpus, list): num_gpus = len(self.args.gpus)
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus)) * self.args.epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.no_lr_scheduler:
            return [optimizer]
        else:
            warm_up_with_cosine_lr = lambda epoch: epoch / warmup_steps if epoch <= warmup_steps else \
                0.5 * (math.cos((epoch - warmup_steps) / (t_total * 4/5) * math.pi) + 1)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output['loss']
        self.log('train_loss', loss)
        self.log('train_lr', self.trainer.optimizers[0].param_groups[0]["lr"])
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output['loss']
        pred = torch.argmax(output['logits'], dim=-1)
        label = batch['labels']
        return {"loss": loss, 'pred': pred, 'label': label}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.sum(torch.stack([x["loss"] for x in outputs]))
        preds = torch.cat([x["pred"] for x in outputs], dim=0)
        labels = torch.cat([x["label"] for x in outputs], dim=0)
        preds = distributed_concat(preds, int(1e9)).tolist()
        labels = distributed_concat(labels, int(1e9)).tolist()

        dev_f1 = f1_score(labels, preds, average='micro')
        self.result_logger.info(
            f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step}"
            f" , current_lr is: {self.trainer.optimizers[0].param_groups[0]['lr']}"
            f" , current_val_loss is: {avg_loss}"
        )

        if self.local_rank <= 0:
            print(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step}"
            f" , current_lr is: {self.trainer.optimizers[0].param_groups[0]['lr']}"
            f" , current_val_loss is: {avg_loss}, dev_f1 is: {dev_f1}, ")

        self.log('val_loss', avg_loss)
        self.log('dev_f1', dev_f1)
        return {"val_loss": avg_loss, 'dev_f1': dev_f1}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        indices = batch.pop('labels')
        output = self.model(**batch)
        pred = torch.argmax(output['logits'], dim=-1)
        return {'pred': pred, 'indices': indices}

    def train_dataloader(self):
        return self.get_dataloader("train")
    def val_dataloader(self):
        return self.get_dataloader("dev")
    def test_dataloader(self):
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train", limit=None):
        tokenizer = BertTokenizer.from_pretrained(self.args.bert_path)
        if prefix == "train":
            train_dataset = Financial_Dataset(self.args, tokenizer, data_path=os.path.join(self.args.data_root, 'train.txt'), split='train')
            dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.workers, drop_last=False, shuffle=False, collate_fn=collect_fn)
        elif prefix == 'dev':
            dev_dataset = Financial_Dataset(self.args, tokenizer, data_path=os.path.join(self.args.data_root, 'dev.txt'), split='train')
            dataloader = DataLoader(dev_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.workers, drop_last=False, shuffle=False, collate_fn=collect_fn)
        else:
            dev_dataset = Financial_Dataset(self.args, tokenizer, data_path=os.path.join(self.args.data_root, 'test.txt'), split='test')
            dataloader = DataLoader(dev_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.workers, drop_last=False, shuffle=False, collate_fn=collect_fn)
        return dataloader

def str2bool(v):
    return v.lower() in ('true')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='/home/data_ti4_c/caoyc/Pretrains/chinese-roberta-wwm-ext-large')
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--TestModel', type=str2bool, default=False)
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    return parser

def main(gpus=None):
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # args.gpus = gpus if gpus is not None else len(over_gpus.split(','))
    args.gpus = -1
    cache_path = './models'
    args.cache_path = cache_path
    args.workers = 4
    args.save_path = './checkpoints/run1'
    args.save_top_k = 1
    args.monitor = 'dev_f1'
    args.mode = 'max'
    args.warmup_proportion = 0.01
    args.no_lr_scheduler = False
    args.weight_decay = 0.001
    args.lr = 5e-5
    args.adam_epsilon = 1e-8
    args.num_sanity_val_steps = 10
    args.val_check_interval = 1.0

    args.accumulate_grad_batches = 1
    args.precision = 32
    args.accelerator = 'ddp'
    args.gradient_clip_val = 5.0

    if not args.TestModel:
        if os.path.exists(os.path.join(args.save_path, 'eval_result_log.txt')):
            os.remove(os.path.join(args.save_path, 'eval_result_log.txt'))
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if not os.path.exists(os.path.join(args.save_path, 'checkpoint')):
            os.mkdir(os.path.join(args.save_path, 'checkpoint'))

        logger = TensorBoardLogger(save_dir=args.save_path, name='log')

    model = Financial_Task(args)

    if not args.TestModel:
        checkpoint_callback = ModelCheckpoint(
            dirpath = os.path.join(args.save_path, "checkpoint"),
            filename = '{epoch:02d}-{dev_f1:.4f}',
            save_top_k=args.save_top_k,
            save_last=True,
            monitor=args.monitor,
            mode=args.mode,
            verbose=True,
            save_weights_only=True,
        )

        with open(os.path.join(args.save_path, "checkpoint", "args.json"), "w") as f:
            args_dict = args.__dict__
            del args_dict["tpu_cores"]
            json.dump(args_dict, f, indent=4)

        trainer = Trainer.from_argparse_args(args,
                                             callbacks=[checkpoint_callback],
                                             logger=logger,
                                             deterministic=True,
                                             # plugins = DDPPlugin(find_unused_parameters=False),
                                             )
        trainer.fit(model)
    else:
        best_model_path = './checkpoints/run1/checkpoint/epoch=01-dev_f1=0.7109.ckpt'
        checkpoint = torch.load(best_model_path)
        model_weights = checkpoint["state_dict"]
        for key in list(model_weights):
            model_weights[key.replace("model.", "")] = model_weights.pop(key)
        model.model.load_state_dict(model_weights)
        trainer = Trainer.from_argparse_args(args)
        predictions = trainer.predict(model, model.get_dataloader('test'))
        preds = torch.cat([x["pred"] for x in predictions], dim=0).tolist()
        indices = torch.cat([x["indices"] for x in predictions], dim=0).tolist()
        preds = sorted([(a, b) for a, b in zip(indices, preds)], key=lambda x: x[0])
        for idx, p in preds:
            print(idx, p)

def out_file_call(gpus):
    from multiprocessing import freeze_support
    freeze_support()
    main(gpus)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
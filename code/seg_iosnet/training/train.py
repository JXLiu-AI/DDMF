import argparse
import torch
import os
from dataset import get_loader
import pytorch_lightning as pl
from pl_model import SegDGCNN
from pytorch_lightning.loggers import WandbLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Tooth Seg Model")
    parser.add_argument('-g', '--num-gpus', type=int, default=1, help='Number of GPUs [default: 1]')
    parser.add_argument('--model-path', type=str, default=None, help='path to restore the model [default: None]')
    parser.add_argument('--num-points', type=int, default=10000, help='Point Number [default: 10000]')
    parser.add_argument('--max-epoch', type=int, default=16, help='Epoch to run [default: 16]')
    parser.add_argument('--lr', type=float, default=3e-3, help='StartLR [default: 3e-3]')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay of AdamW [default: 1e-3]')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch Size during training [default: 10]')
    parser.add_argument('--eval-per-n-epoch', type=int, default=1, help='evaluate after every n epoch [default: 1]')
    parser.add_argument('--k', type=int, default=25, help='K neighboring nodes to look at [default: 25]')
    parser.add_argument('--num-parts', type=int, default=33, help='Number of parts for segmentation [default: 33]')
    parser.add_argument('--num-workers', type=int, default=32, help='Number of workers [default: 32]')
    parser.add_argument('--num-features', type=int, default=15, help='Number of features per point')
    parser.add_argument('--num-categories', type=int, default=2, help='Number of categories')
    parser.add_argument('--train-set', type=str, help='Path to training set', required=True)
    parser.add_argument('--val-set', type=str, help='Path to testing set', required=True)
    parser.add_argument('--model-save-dir', type=str, default="./saved", help='Dir to save the trained model')

    args = parser.parse_args()
    return args

args = parse_args()

train_loader = get_loader(args.batch_size, args.train_set, True, args.num_workers)
val_loader = get_loader(args.batch_size, args.val_set, False, args.num_workers)

M = SegDGCNN(
        args.num_features, args.num_parts, args.num_categories, args.k, args.lr, args.weight_decay
    ) if args.model_path is None else SegDGCNN.load_from_checkpoint(args.model_path,
    start_lr=args.lr, k=args.k, weight_decay=args.weight_decay)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
mc = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=None,
    filename='{epoch}-{val_acc:.2f}-{val_loss:.2f}-{train_acc:.2f}-{train_loss:.2f}',
    monitor=None,
    verbose=True,
    save_top_k=-1,
    every_n_val_epochs=1)

trainer = pl.Trainer(gpus=args.num_gpus, default_root_dir=args.model_save_dir, callbacks=[lr_monitor, mc],
                     accelerator="dp", check_val_every_n_epoch=args.eval_per_n_epoch, max_epochs=args.max_epoch,
                     logger=WandbLogger(project='experiment'))
trainer.fit(M, train_dataloader=train_loader, val_dataloaders=val_loader)
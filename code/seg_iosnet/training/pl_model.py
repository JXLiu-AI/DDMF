import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DGCNN_partseg, knn
import pytorch_lightning as pl

class SegDGCNN(pl.LightningModule):

    def __init__(self, num_features, num_parts, num_categories, k, start_lr=3e-3, weight_decay=1e-3):
        super().__init__()
        self.start_lr = start_lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.dgcnn = DGCNN_partseg(
            num_features,
            num_parts,
            num_categories,
            k
        )

    def forward(self, x):
        y = self.dgcnn(x[0], x[1])
        y = F.softmax(y, 1)
        return y

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        nl_loss_crit = nn.NLLLoss()
        x, l, y = batch # (bs, 15, pts), (bs, cats), (bs, pts)
        inf_result = self.dgcnn(x, l)
        inf_result_softmax = F.softmax(inf_result, 1)
        inf_result_softmax_log = torch.log_softmax(inf_result, 1) # (bs, 33, num_points)
        ce_loss = nl_loss_crit(inf_result_softmax_log, y)

        # loop over bs
        bs = x.size(0)
        num_pts = x.size(2)
        centroid_dist = []

        neighbor_k = 5

        neighbor_idx = knn(x[:, :3, :], neighbor_k) # (batch_size, num_points, 5)
        boundary_loss = 0.0

        for b in range(bs):

            # centroid loss
            for uid in torch.unique(y[b]):
                if uid == 0:
                    # do not process jaw
                    continue

                # Get GT centroid
                batch_pts = x[b, :3] #(3, num_pts)
                gt_uid_pts_id = (y[b] == uid).nonzero(as_tuple=True)[0]
                gt_uid_pts = batch_pts[:, gt_uid_pts_id]
                gt_centroid = torch.mean(gt_uid_pts, 1)

                # Get PRED centroid
                uid_pred_prob = F.threshold(inf_result_softmax[b, uid], 0.38, 0.0) #(num_pts),
                uid_pred_prob = -F.threshold(-uid_pred_prob, -0.6, -1.0)
                pred_centroid = torch.sum(uid_pred_prob.view(1, -1) * batch_pts, 1) / F.threshold(torch.sum(uid_pred_prob), 1e-8, 1e-8)
                centroid_dist.append(torch.sqrt(torch.sum(torch.square(pred_centroid - gt_centroid))))

            source_distrib_log = inf_result_softmax_log[b].transpose(0, 1).contiguous()
            source_distrib_log = source_distrib_log.view(num_pts, 1, -1).repeat(1, neighbor_k, 1) #(num_pts, 5, 33)

            target_distrib = inf_result_softmax[b].transpose(0, 1).contiguous()[neighbor_idx[b]]
            kl_div_ = torch.clamp(F.kl_div(source_distrib_log,
                                           target_distrib,
                                           reduction='none'), max=8.0).mean(dim=-1).max(dim=-1)[0].clone().detach() #(num_pts)
            kl_div_thres = torch.quantile(kl_div_, 1.0 - 0.05) # top 5% is boundary points
            boundary_id = (kl_div_ > kl_div_thres).nonzero(as_tuple=True)[0]
            boundary_loss += F.nll_loss(inf_result_softmax_log[b, :, boundary_id].transpose(0, 1), y[b, boundary_id])

        centroid_loss = sum(centroid_dist) / len(centroid_dist)
        boundary_loss /= bs

        loss = ce_loss + centroid_loss + boundary_loss
        inf_argmax = torch.argmax(inf_result, 1)
        inf_acc = torch.mean((y == inf_argmax).float())
        self.log('train_acc', inf_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_centroid_loss', centroid_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_boundary_loss', boundary_loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seg_loss = nn.CrossEntropyLoss()
        x, l, y = batch
        inf_result = self.dgcnn(x, l)
        loss = seg_loss(inf_result, y)
        inf_argmax = torch.argmax(inf_result, 1)
        inf_acc = torch.mean((y == inf_argmax).float())
        self.log('val_acc', inf_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.start_lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, min_lr=1e-5, factor=0.3)
        return ({"optimizer": optimizer, "lr_scheduler":{"scheduler": scheduler, "monitor": "train_loss"}},)

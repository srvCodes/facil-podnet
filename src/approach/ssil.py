import pdb

import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from collections import defaultdict

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform


class Appr(Inc_Learning_Appr):
    """
    """

    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, ratio=4, distill_temp=2.0, lamb=1.0):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.exemplars_loader = None
        self.exemplars_iter = None
        self.ratio = ratio
        self.distill_temp = distill_temp
        self.lamb = lamb

        self.current_epoch = 1
        self.additional_stats = {'train': defaultdict(list), 'eval': defaultdict(list)}

        # SS-IL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument('--ratio', default=4, type=float, required=False,
                            help="write-me") # TODO: write me
        parser.add_argument('--distill-temp', default=2., type=float, required=False,
                            help='Temperature to use in distillation softmax')
        parser.add_argument('--lamb', type=float, default=1., required=False,
                            help="")
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        self.current_epoch = 1

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # Add exemplars before collecting new one (I'm still not sure about it)
        if t > 0:
            sel_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        else:
            sel_loader = trn_loader

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, sel_loader, val_loader.dataset.transform)

        # Get batch size for the exemplars loader
        assert trn_loader.batch_size % self.ratio == 0
        exemplars_bs = trn_loader.batch_size // self.ratio

        # Create exemplars loader after gathering new exemplars
        self.exemplars_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                            batch_size=exemplars_bs,
                                                            shuffle=True,
                                                            num_workers=trn_loader.num_workers,
                                                            pin_memory=trn_loader.pin_memory,
                                                            drop_last=True)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def _get_exemplars_batch(self):

        # ugly hack for getting exemplars
        try:
            ex_images, ex_targets = next(self.exemplars_iter)
        except StopIteration:
            self.exemplars_iter = self.exemplars_loader.__iter__()
            ex_images, ex_targets = next(self.exemplars_iter)

        ex_images = ex_images.to(self.device)
        ex_targets = ex_targets.to(self.device)

        return ex_images, ex_targets

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        if t > 0:
            self.exemplars_iter = self.exemplars_loader.__iter__()

        ex_outputs = None
        ex_targets = None
        outputs_old = None

        for images, targets in trn_loader:

            images = images.to(self.device)
            if t > 0:
                # Get fixed-size exemplars batch
                ex_images, ex_targets = self._get_exemplars_batch()
                # Forward exemplars on current model
                ex_outputs = self.model(ex_images)
                # Forward task data on old model
                task_outputs_old = self.model_old(images)
                # Forward exemplars on old model
                ex_outputs_old = self.model_old(ex_images)

                # Cat task data and exemplars head-wise (will help in TWD) + iCarl sigmoid
                outputs_old = [torch.nn.functional.log_softmax(torch.cat([task_head, ex_head], dim=0) / self.distill_temp , dim=1)
                               for task_head, ex_head in zip(task_outputs_old, ex_outputs_old)]

            # Forward task data on current model
            outputs = self.model(images)
            # Calculate loss
            loss, supp_losses = self.criterion(t,
                                  task_outputs=outputs,
                                  task_targets=targets.to(self.device),
                                  ex_outputs=ex_outputs,
                                  ex_targets=ex_targets,
                                  outputs_old=outputs_old)

            for key, val in supp_losses.items():
                self.additional_stats['train'][key].append(val.detach().cpu().item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        # log additional losses
        for key, val in self.additional_stats['train'].items():
            self.logger.log_scalar(task=t, iter=self.current_epoch, name=key, value=val, group='train')
            self.additional_stats['train'][key] = []

        self.current_epoch += 1

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            outputs_old = None

            for images, targets in val_loader:
                # Forward old model
                images = images.to(self.device)
                if t > 0:
                    outputs_old = self.model_old(images)

                # Forward current model
                outputs = self.model(images)

                # during training, the usual accuracy is computed on the outputs
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log

                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

                loss, supp_losses = self.criterion(t,
                                      task_outputs=outputs,
                                      task_targets=targets.to(self.device),
                                      ex_outputs=None,
                                      ex_targets=None,
                                      outputs_old=outputs_old)

                for key, val in supp_losses.items():
                    if val is not None:
                        self.additional_stats['eval'][key].append(val.cpu().item())

                total_loss += loss.item() * len(targets)

            # log additional losses
            for key, val in self.additional_stats['eval'].items():
                self.logger.log_scalar(task=t, iter=self.current_epoch - 1, name=key, value=val, group='eval')
                self.additional_stats['eval'][key] = []

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, task_outputs, task_targets, ex_outputs=None, ex_targets=None, outputs_old=None):
        """Returns the loss value"""

        # Separate classification loss for new classes
        current_task_loss = torch.nn.functional.cross_entropy(task_outputs[t], task_targets - self.model.task_offset[t])
        loss = current_task_loss

        if t > 0:
            if ex_outputs is not None and ex_targets is not None:
                # Separate classification loss for old classes
                ex_loss = torch.nn.functional.cross_entropy(torch.cat(ex_outputs[:t], dim=1), ex_targets)
                loss += ex_loss

                outputs = [torch.nn.functional.log_softmax(torch.cat([task_head, ex_head], dim=0) / self.distill_temp, dim=1)
                           for task_head, ex_head in zip(task_outputs, ex_outputs)]
            else:
                # This will run during eval when there's no exemplars - just TWD
                outputs = [torch.nn.functional.log_softmax(task_head / self.distill_temp , dim=1)
                           for task_head in task_outputs]
                outputs_old = [torch.nn.functional.log_softmax(old_head / self.distill_temp , dim=1)
                               for old_head in outputs_old]
                ex_loss = None

            # No distillation on current task's head
            g = outputs[:t]
            q = outputs_old[:t]

            # Task Wise Distillation loss for old classes
            kd_loss = self.lamb * sum(torch.nn.functional.kl_div(g_t, q_t, reduction='batchmean', log_target=True) for g_t, q_t in zip(g, q))
            loss += kd_loss

            return loss, {'L_curr': current_task_loss, 'L_e': ex_loss, 'L_kd': kd_loss}
        else:
            return loss, {'L_curr': loss}
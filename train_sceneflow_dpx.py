import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
# from util import InputPadder
from core.utils.utils import InputPadder
from models.models.stereoplus_dpx.stereoplus_use_aahead import StereoNet
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from models.loss.losses import calc_loss
from torch.utils.data import Subset

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"  # 彻底禁用 wandb


def gray_2_colormap_np(img, cmap='rainbow', max=None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img < 0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps[cmap]
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


def sequence_loss_ori(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)  # & ((disp_init_pred - disp_gt).abs() < quantile)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }
    return disp_loss, metrics


def sequence_loss(
        disp_preds,
        disp_gt,
        valid_mask,
        max_disp=192,
        loss_gamma=0.9,
        use_fixed_weights=True,
        loss_type='smooth_l1',
):
    # Handle input format
    if isinstance(disp_preds, dict):
        preds = [disp_preds['disp_low'], disp_preds['disp_1']]
    else:
        preds = disp_preds

    n_predictions = len(preds)
    assert n_predictions >= 1

    # Convert all inputs to 4D tensors
    disp_gt = disp_gt.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
    valid_mask = valid_mask.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]

    # Ensure predictions are 4D
    preds = [p.unsqueeze(1) if p.dim() == 3 else p for p in preds]

    # Compute valid mask
    disp_mag = disp_gt.norm(p=2, dim=1, keepdim=True)
    valid = (valid_mask >= 0.5) & (disp_mag < max_disp) & torch.isfinite(disp_gt)
    valid = valid.expand_as(disp_gt)  # Now safe

    # Rest of your original code...
    if use_fixed_weights:
        if n_predictions == 2:
            weights = [1 / 3, 2 / 3]
        else:
            raise ValueError("Fixed weight mode only supports 2-stage prediction.")
    else:
        adjusted_gamma = loss_gamma ** (15 / max(1, n_predictions - 1))
        weights = [adjusted_gamma ** (n_predictions - i - 1) for i in range(n_predictions)]

    loss = 0.0
    for i, (pred, weight) in enumerate(zip(preds, weights)):
        if loss_type == 'smooth_l1':
            error_map = F.smooth_l1_loss(pred, disp_gt, reduction='none')
        elif loss_type == 'l1':
            error_map = (pred - disp_gt).abs()
        valid_error = error_map[valid]
        if valid_error.numel() > 0:
            loss += weight * valid_error.mean()

    # EPE calculation
    final_pred = preds[-1]
    epe = (final_pred - disp_gt).norm(p=2, dim=1)  # shape: (B, H, W)
    epe_valid = epe[valid[:, 0, :, :]]  # remove channel dimension

    if epe_valid.numel() == 0:
        epe_valid = torch.tensor([0.0], device=disp_gt.device)

    metrics = {
        'train/epe': epe_valid.mean(),
        'train/1px': (epe_valid < 1).float().mean(),
        'train/3px': (epe_valid < 3).float().mean(),
        'train/5px': (epe_valid < 5).float().mean(),
    }

    return loss, metrics


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    # Check if model has 'feat_decoder' (for backward compatibility)
    if hasattr(model, "feat_decoder"):
        # Case 1: Model has feat_decoder (e.g., DPT-based models)
        DPT_params = list(map(id, model.feat_decoder.parameters()))
        rest_params = filter(
            lambda x: id(x) not in DPT_params and x.requires_grad,
            model.parameters()
        )
        params_dict = [
            {"params": model.feat_decoder.parameters(), "lr": args.lr / 2.0},
            {"params": rest_params, "lr": args.lr},
        ]
    else:
        # Case 2: Model does not have feat_decoder (e.g., StereoNet)
        params_dict = [
            {"params": model.parameters(), "lr": args.lr},
        ]

    optimizer = optim.AdamW(
        params_dict,
        lr=args.lr,
        weight_decay=args.wdecay,
        eps=1e-8,
    )
    max_lrs = (
        [args.lr / 2.0, args.lr]  # 如果有 2 个参数组（feat_decoder + rest）
        if hasattr(model, "feat_decoder")
        else args.lr  # 如果只有 1 个参数组
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lrs,
        args.total_step + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


@hydra.main(version_base=None, config_path='config', config_name='train_sceneflow')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='no', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb', kwargs_handlers=[
        kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs={
        'wandb': cfg.wandb})
    # 确保 wandb 已初始化
    if accelerator.is_main_process:
        wandb_run_name = wandb.run.name if wandb.run else "unnamed_run"
    else:
        wandb_run_name = "unnamed_run"  # 非主进程可能需要广播
    save_root = Path(cfg.save_path) / wandb_run_name
    save_root.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.fetch_dataloader(cfg)
    # Create a subset with first 1000 samples
    # train_dataset = Subset(train_dataset, indices=range(1000))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size // cfg.num_gpu,
                                               pin_memory=True, shuffle=True, num_workers=int(16), drop_last=True)

    aug_params = {}
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, pin_memory=True, shuffle=False, num_workers=8, drop_last=False)

    model = StereoNet(cfg)
    if not cfg.restore_ckpt.endswith("None"):
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]

        model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device)

    total_step = 0
    should_keep_training = True
    best_epe = float('inf')  # 最低的 epe
    while should_keep_training:
        active_train_loader = train_loader

        model.train()
        if hasattr(model, "module"):
            model.module.freeze_bn()
        else:
            model.freeze_bn()
        for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            _, left, right, disp_gt, valid = [x for x in data]
            with accelerator.autocast():
                disp_gt = disp_gt.squeeze(1)
                disp_preds = model(left, right, disp_gt)
            # loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=cfg.max_disp)
            loss, metrics = sequence_loss(disp_preds, disp_gt, valid)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metrics = accelerator.reduce(metrics, reduction='mean')
            accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics, total_step)

            ####visualize the depth_mono and disp_preds
            if total_step % 20 == 0 and accelerator.is_main_process:
                image1_np = left[0].squeeze().cpu().numpy()
                image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                image1_np = image1_np.astype(np.uint8)
                image1_np = np.transpose(image1_np, (1, 2, 0))

                image2_np = right[0].squeeze().cpu().numpy()
                image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                image2_np = image2_np.astype(np.uint8)
                image2_np = np.transpose(image2_np, (1, 2, 0))

                # depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                disp_preds_np = gray_2_colormap_np(disp_preds['disp_1'][0].squeeze())
                disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())

                accelerator.log({
                                    "disp_pred": wandb.Image(disp_preds_np, caption="step:{}".format(total_step))}, total_step)
                accelerator.log({"disp_gt": wandb.Image(disp_gt_np, caption="step:{}".format(total_step))}, total_step)
                # accelerator.log({"depth_mono": wandb.Image(depth_mono_np, caption="step:{}".format(total_step))}, total_step)

            # if (total_step > 0) and (total_step % cfg.save_frequency == 0):
            #     if accelerator.is_main_process:
            #         save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
            #         model_save = accelerator.unwrap_model(model)
            #         torch.save(model_save.state_dict(), save_path)
            #         del model_save

            if (total_step > 0) and (total_step % cfg.val_frequency == 0):

                model.eval()
                elem_num, total_epe, total_out = 0, 0, 0
                for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                    _, left, right, disp_gt, valid = [x for x in data]
                    padder = InputPadder(left.shape, divis_by=32)
                    left, right = padder.pad(left, right)
                    with torch.no_grad():
                        disp_pred = model(left, right).unsqueeze(1)
                    disp_pred = padder.unpad(disp_pred)
                    assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                    epe = torch.abs(disp_pred - disp_gt)
                    out = (epe > 1.0).float()
                    epe = torch.squeeze(epe, dim=1)
                    out = torch.squeeze(out, dim=1)
                    disp_gt = torch.squeeze(disp_gt, dim=1)
                    epe, out = accelerator.gather_for_metrics((epe[(valid >= 0.5) & (disp_gt.abs() < 192)].mean(),
                                                               out[(valid >= 0.5) & (disp_gt.abs() < 192)].mean()))
                    # Handle scalar values properly
                    if epe.dim() == 0:  # if it's a scalar
                        elem_num += 1
                        total_epe += epe.item()
                        total_out += out.item()
                    else:
                        elem_num += epe.shape[0]
                        for i in range(epe.shape[0]):
                            total_epe += epe[i].item()
                            total_out += out[i].item()
                    val_epe = total_epe / elem_num
                    val_d1 = 100 * total_out / elem_num
                    accelerator.log({'val/epe': val_epe, 'val/d1': val_d1}, total_step)

                    # 添加保存 best 模型
                    if accelerator.is_main_process and val_epe < best_epe:
                        best_epe = val_epe
                        best_model_path = save_root / 'best.pth'
                        model_save = accelerator.unwrap_model(model)
                        torch.save(model_save.state_dict(), best_model_path)
                        del model_save

                model.train()
                # model.module.freeze_bn()

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        save_path = save_root / 'last.pth'
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save

    accelerator.end_training()


if __name__ == '__main__':
    main()
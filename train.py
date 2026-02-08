import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import os
import importlib
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

# Project imports
from dataset import TrajectoryDataset
from model import TrajectoryModel
from utils import get_motion_modes


# ==========================================
# DataParallel Wrapper
# ==========================================
class DataParallelWrapper(nn.Module):
    """
    Wraps the model to handle constant arguments (motion_modes)
    that should not be split by DataParallel.
    """

    def __init__(self, model, motion_modes):
        super(DataParallelWrapper, self).__init__()
        self.model = model
        # Register as buffer so it moves to device but isn't a parameter
        self.register_buffer('fixed_motion_modes', motion_modes)

    def forward(self, ped_obs, neis_obs, dummy_motion_modes, mask, closest_mode_indices, test=False, num_k=20):
        # We ignore 'dummy_motion_modes' passed by DataParallel scatter
        # and use the full 'fixed_motion_modes' instead.
        return self.model(ped_obs, neis_obs, self.fixed_motion_modes, mask, closest_mode_indices, test, num_k)


# ==========================================
# Helper Functions
# ==========================================

def get_cls_label(gt, motion_modes, soft_label=True):
    """Calculates classification labels based on ground truth and modes."""
    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)
    motion_modes = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)
    distance = torch.norm(gt - motion_modes, dim=-1)

    # Soft labels for cross-entropy
    soft_label = F.softmax(-distance, dim=-1)
    # Hard labels for accuracy/analysis
    closest_mode_indices = torch.argmin(distance, dim=-1)
    return soft_label, closest_mode_indices


def train_epoch(model, reg_criterion, cls_criterion, optimizer, train_dataloader, motion_modes, args, device):
    """Runs one training epoch."""
    model.train()
    total_loss = []

    for i, (ped, neis, mask) in enumerate(train_dataloader):
        ped = ped.to(device)
        neis = neis.to(device)
        mask = mask.to(device)

        # Scale data if using ETH dataset specific logic
        if args.dataset_name == 'eth':
            ped[:, :, 0] = ped[:, :, 0] * args.data_scaling[0]
            ped[:, :, 1] = ped[:, :, 1] * args.data_scaling[1]

        # Random scaling augmentation
        scale = torch.randn(ped.shape[0]) * 0.05 + 1
        scale = scale.to(device).reshape(ped.shape[0], 1, 1)
        ped = ped * scale
        scale = scale.reshape(ped.shape[0], 1, 1, 1)
        neis = neis * scale

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        # Generate target labels
        with torch.no_grad():
            soft_label, closest_mode_indices = get_cls_label(gt, motion_modes)

        optimizer.zero_grad()

        # Forward pass
        pred_traj, scores = model(ped_obs, neis_obs, motion_modes, mask, closest_mode_indices)

        # Loss calculation
        reg_label = gt.reshape(pred_traj.shape)
        reg_loss = reg_criterion(pred_traj, reg_label)
        clf_loss = cls_criterion(scores.squeeze(), soft_label)

        loss = reg_loss + clf_loss
        loss = loss.mean()  # Reduce to scalar for DataParallel

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return total_loss


def evaluate(model, test_dataloader, motion_modes, args, device):
    """Evaluates the model on test data."""
    model.eval()
    ade = 0
    fde = 0
    num_traj = 0

    for (ped, neis, mask) in test_dataloader:
        ped = ped.to(device)
        neis = neis.to(device)
        mask = mask.to(device)

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        with torch.no_grad():
            num_traj += ped_obs.shape[0]
            # Inference mode
            pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)

            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)
            gt_ = gt.unsqueeze(1)

            # Calculate metrics
            norm_ = torch.norm(pred_trajs - gt_, p=2, dim=-1)
            ade_ = torch.mean(norm_, dim=-1)
            fde_ = norm_[:, :, -1]

            # Select best mode
            min_ade, _ = torch.min(ade_, dim=-1)
            min_fde, _ = torch.min(fde_, dim=-1)

            ade += torch.sum(min_ade).item()
            fde += torch.sum(min_fde).item()

    ade = ade / num_traj
    fde = fde / num_traj
    return ade, fde, num_traj


# ==========================================
# Main Entry Point
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trajectory Prediction Training Script")
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--dataset_name', type=str, default='sdd')
    parser.add_argument("--hp_config", type=str, default=None, help='Path to hyper-parameter config file')
    parser.add_argument('--lr_scaling', action='store_true', default=False, help='Enable LR scheduler')
    parser.add_argument('--num_works', type=int, default=8)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')

    # Corrected argument parsing for list input
    parser.add_argument('--data_scaling', nargs='+', type=float, default=[1.9, 0.4], help='Scaling factors [x, y]')
    parser.add_argument('--dist_threshold', type=float, default=2)
    parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='Path to save models')

    args = parser.parse_args()

    # Set Reproducibility Seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"[*] Configuration: {args}")

    # Setup Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on {device}, GPU Count: {torch.cuda.device_count()}")

    # Load Hyper-parameters dynamically
    if not args.hp_config:
        raise ValueError("Please provide a hyper-parameter config file using --hp_config")

    spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
    hp_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp_config)

    # Dataset Initialization
    train_dataset = TrajectoryDataset(
        dataset_path=args.dataset_path, dataset_name=args.dataset_name,
        dataset_type='train', translation=True, rotation=True,
        scaling=True, obs_len=args.obs_len,
        dist_threshold=hp_config.dist_threshold, smooth=False
    )

    test_dataset = TrajectoryDataset(
        dataset_path=args.dataset_path, dataset_name=args.dataset_name,
        dataset_type='test', translation=True, rotation=True,
        scaling=False, obs_len=args.obs_len
    )

    # Load or Generate Motion Modes
    motion_modes_file = os.path.join(args.dataset_path, f"{args.dataset_name}_motion_modes.pkl")
    if not os.path.exists(motion_modes_file):
        print(f"[*] Generating motion modes for {args.dataset_name}...")
        motion_modes = get_motion_modes(
            train_dataset, args.obs_len, args.pred_len, hp_config.n_clusters,
            args.dataset_path, args.dataset_name,
            smooth_size=hp_config.smooth_size, random_rotation=hp_config.random_rotation,
            traj_seg=hp_config.traj_seg
        )
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).to(device)
    else:
        print(f"[*] Loading motion modes from {motion_modes_file}...")
        import pickle

        with open(motion_modes_file, 'rb') as f:
            motion_modes = pickle.load(f)
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).to(device)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, collate_fn=train_dataset.coll_fn,
        batch_size=hp_config.batch_size, shuffle=True, num_workers=args.num_works
    )
    test_loader = DataLoader(
        test_dataset, collate_fn=test_dataset.coll_fn,
        batch_size=hp_config.batch_size, shuffle=True, num_workers=args.num_works
    )

    # Model Initialization
    local_window_radius = getattr(hp_config, 'local_window_radius', 4)

    model = TrajectoryModel(
        in_size=2,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embed_size=hp_config.model_hidden_dim,
        enc_num_layers=hp_config.enc_num_layers,
        int_num_layers_list=[hp_config.int_num_layers, hp_config.int_num_layers],
        heads=hp_config.heads,
        forward_expansion=hp_config.forward_expansion,
        local_window_radius=local_window_radius
    )

    model = model.to(device)

    # Handle DataParallel
    if torch.cuda.device_count() > 1:
        print(f"[*] Using {torch.cuda.device_count()} GPUs for training.")
        model = DataParallelWrapper(model, motion_modes)
        model = torch.nn.DataParallel(model)
    else:
        print("[*] Using Single GPU.")

    # Optimizer & Loss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp_config.lr)
    reg_criterion = torch.nn.SmoothL1Loss().to(device)
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.lr_scaling:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[270, 400], gamma=0.5)

    # Training Loop
    min_ade = 99
    min_fde = 99
    min_fde_epoch = -1

    save_dir = os.path.join(args.checkpoint, args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[*] Start training on {args.dataset_name} ...")

    for ep in range(hp_config.epoch):
        total_loss = train_epoch(ep, model, reg_criterion, cls_criterion, optimizer, train_loader, motion_modes, args,
                                 device)
        ade, fde, num_traj = evaluate(model, test_loader, motion_modes, args, device)

        if args.lr_scaling:
            scheduler.step()

        train_loss = sum(total_loss) / len(total_loss)
        print(f"Epoch {ep:03d} | Loss: {train_loss:.4f} | ADE: {ade:.4f} | FDE: {fde:.4f}")

        # Save Best Model
        if (ade + fde) < (min_ade + min_fde):
            min_fde = fde
            min_ade = ade
            min_fde_epoch = ep

            save_path = os.path.join(save_dir, 'best.pth')

            # Unwrap model for saving (important for loading later without DataParallel)
            model_to_save = model
            if isinstance(model_to_save, torch.nn.DataParallel):
                model_to_save = model_to_save.module
            if isinstance(model_to_save, DataParallelWrapper):
                model_to_save = model_to_save.model

            torch.save(model_to_save.state_dict(), save_path)
            print(f"    >>> Best model saved at Epoch {ep} (ADE: {ade:.4f}, FDE: {fde:.4f})")

    print(f"[*] Training Finished. Best Epoch: {min_fde_epoch}, Min ADE: {min_ade:.4f}, Min FDE: {min_fde:.4f}")

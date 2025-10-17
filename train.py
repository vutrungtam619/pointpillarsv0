import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from configs import config
from utils import setup_seed, Loss
from datasets import get_train_dataloader, get_val_dataloader, Kitti
from models import PointPillars
from torch.utils.tensorboard import SummaryWriter

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)
        
def main(args):
    checkpoint_folder = Path(args.checkpoint_dir)
    checkpoint_folder.mkdir(exist_ok=True)
    log_folder = Path(args.log_dir)
    log_folder.mkdir(exist_ok=True)
    
    print("Loading dataset!..............................................")
    train_dataset = Kitti(data_root=args.data_root, split='train')
    val_dataset = Kitti(data_root=args.data_root, split='val')
    
    train_dataloader = get_train_dataloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = get_val_dataloader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)    
    print("Loading dataset succesfully!..................................")   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    print("Loading model!.....................")
    model = PointPillars().to(device)
    print("Finished loading model!............")
    
    loss_func = Loss()
    
    max_iters = len(train_dataloader) * args.epoch    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.init_lr, 
        betas=(0.95, 0.99), 
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.init_lr*10, 
        total_steps=max_iters, 
        pct_start=0.3,
        anneal_strategy='cos', 
        cycle_momentum=True, 
        base_momentum=0.85, 
        max_momentum=0.95, 
        div_factor=10,
    )    
    
    start_epoch = 0
    
    # Resume train if there is checkpoint exist
    checkpoint_exist = list(checkpoint_folder.glob("*.pth"))

    if checkpoint_exist:
        checkpoint_path = checkpoint_exist[0]
        checkpoint_dict = torch.load(checkpoint_path)
        start_epoch = int(checkpoint_dict['epoch']) 
        print(f"Found checkpoint! Continue training from epoch {start_epoch}.......")
        model.load_state_dict(checkpoint_dict['checkpoint'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        print("Checkpoint, optimizer, scheduler loaded successfully!")
    else:
        print("No checkpoint found! Starting training from scratch!")
    
    writer = SummaryWriter(args.log_dir)
    
    for epoch in range(start_epoch, args.epoch):
        print('='*20, f'Epoch {epoch}', '='*20)
        model.train()
        
        # ---- accumulate training loss ----
        train_loss_accum = {}
        train_batches = 0

        for i, data_dict in enumerate(tqdm(train_dataloader)):
            # Move tensors to device
            for key in data_dict:
                for j, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][j] = data_dict[key][j].to(device)

            optimizer.zero_grad()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']

            # Forward
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = model(
                batched_pts=batched_pts, mode='train',
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_labels,
            )   
            
            # Reshape
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.num_classes)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            
            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.num_classes)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            a = bbox_pred[:, -1].clone()
            b = batched_bbox_reg[:, -1].clone()
            bbox_pred[:, -1] = torch.sin(a) * torch.cos(b)
            batched_bbox_reg[:, -1] = torch.cos(a) * torch.sin(b)
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]
            
            num_cls_pos = (batched_bbox_labels < args.num_classes).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.num_classes
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]
            
            loss_dict = loss_func(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_labels=batched_bbox_labels, 
                num_cls_pos=num_cls_pos, 
                batched_bbox_reg=batched_bbox_reg, 
                batched_dir_labels=batched_dir_labels
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # accumulate loss
            for k, v in loss_dict.items():
                train_loss_accum[k] = train_loss_accum.get(k, 0.0) + v.item()
            train_batches += 1
        
        # ---- log average train loss for this epoch ----
        avg_train_loss = {k: v/train_batches for k, v in train_loss_accum.items()}
        save_summary(
            writer,
            avg_train_loss,
            epoch + 1,
            'train',
            lr=optimizer.param_groups[0]['lr'],
            momentum=optimizer.param_groups[0]['betas'][0]
        )

        # Save checkpoint
        if ((epoch + 1) % args.ckpt_freq == 0):
            save_dict = {
                "checkpoint": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1
            }
            new_ckpt_file = Path(checkpoint_folder) / f'epoch_{epoch+1}.pth'
            torch.save(save_dict, new_ckpt_file)
                
        # ---- Validation every 2 epochs ----
        if epoch % 2 == 0:
            continue
        
        model.eval()
        val_loss_accum = {}
        val_batches = 0
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].to(device)
                            
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                
                # Forward
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = model(
                    batched_pts=batched_pts, mode='val',
                    batched_gt_bboxes=batched_gt_bboxes, 
                    batched_gt_labels=batched_labels,
                )   
                
                # Reshape
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.num_classes)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.num_classes)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                a = bbox_pred[:, -1].clone()
                b = batched_bbox_reg[:, -1].clone()
                bbox_pred[:, -1] = torch.sin(a) * torch.cos(b)
                batched_bbox_reg[:, -1] = torch.cos(a) * torch.sin(b)
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]
                
                num_cls_pos = (batched_bbox_labels < args.num_classes).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.num_classes
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]
                
                loss_dict = loss_func(
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_pred=bbox_pred,
                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                    batched_labels=batched_bbox_labels, 
                    num_cls_pos=num_cls_pos, 
                    batched_bbox_reg=batched_bbox_reg, 
                    batched_dir_labels=batched_dir_labels
                )     

                # accumulate validation loss
                for k, v in loss_dict.items():
                    val_loss_accum[k] = val_loss_accum.get(k, 0.0) + v.item()
                val_batches += 1
        
        # ---- log average val loss for this epoch ----
        avg_val_loss = {k: v/val_batches for k, v in val_loss_accum.items()}
        save_summary(writer, avg_val_loss, epoch + 1, 'val')
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default=config['data_root'])
    parser.add_argument('--checkpoint_dir', default=config['checkpoint_dir'])
    parser.add_argument('--log_dir', default=config['log_dir'])
    parser.add_argument('--batch_size', default=config['batch_size_train'])
    parser.add_argument('--num_workers', default=config['num_workers'])
    parser.add_argument('--num_classes', default=config['num_classes'])
    parser.add_argument('--init_lr', default=config['init_lr'])
    parser.add_argument('--epoch', default=config['epoch'])
    parser.add_argument('--ckpt_freq', default=config['ckpt_freq'])
    args = parser.parse_args()
    
    main(args)      
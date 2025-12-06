import torch
import torch.nn as nn
import torch.optim as optim
import time
import os 
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import YOLOLoss
import config

class YOLOV3_trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = self.device
        
        anchors = [
            [(116, 90), (156, 198), (373, 326)],   # 13x13
            [(30, 61), (62, 45), (59, 119)],       # 26x26
            [(10, 13), (16, 30), (33, 23)]         # 52x52
        ]

        # 损失函数
        self.criterion = YOLOLoss(
            anchors = anchors,
            img_size = config.img_size
        )

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )

        # 训练状态
        self.cur_epoch = 0
        self.best_loss = float('inf')

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

    def train_epoch(self):
        """
            训练一个epoch的数据
        """
        self.model.train()
        total_loss = 0
        loss_dict = {
            'total_loss':0,
            'coord_loss':0,
            'obj_loss':0,
            'noobj_loss':0,
            'class_loss':0
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch{self.cur_epoch}')

        for batch_idx, (images, targets) in pbar:
            images = images.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # 计算损失
            loss, loss_item = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            total_loss += loss.item()
            for key in loss_dict:
                loss_dict[key] += loss_item[key]
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_dict:
            loss_dict[key] /= len(self.train_loader)
        return avg_loss, loss_dict
    
    def validation(self):
        """
            验证模型
        """

        self.model.eval()
        total_loss = 0
        loss_dict = {
            'total_loss':0,
            'coord_loss':0,
            'obj_loss':0,
            'noobj_loss':0,
            'class_loss':0
        }

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                outputs = self.model(images)
                loss, loss_item = self.criterion(outputs, targets)

                total_loss += loss.item()
                for key in loss_dict:
                    loss_dict[key] += loss_item[key]

        avg_loss = total_loss / len(self.val_loader)
        for key in loss_dict:
            loss_dict[key] /= len(self.val_loader)
        return avg_loss, loss_dict
    
    def save_checkpoint(self, is_best = False):
        checkpoint = {
            'epoch': self.cur_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }

        checkpoint_path = os.path.join(
            config.output_dir,
            f'checkpoint_epoch_{self.cur_epoch}.pth'
        )

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(config.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳的模型,loss:{self.best_loss:.4f}")

if __name__ == "__main__":
    print(config.learning_rate)

# train/train_segmoe.py
import sys
import os,re
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm  # 添加训练进度条
import torchvision.transforms import v2
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))  # 修正路径层级
import matplotlib.pyplot as plt
#导入数据,需要git-lfs支持

from modelscope.msdatasets import MsDataset

from models.segmoe import SegMoEPipeline


class GatedTrainer:
    def __init__(self, config_path: str):
        super().__init__()
        # 初始化配置
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5])
        ]).cuda()
        # 构建模型管道
        self.pipeline = SegMoEPipeline.from_pretrained(config_path)
        # self._freeze_experts()  # 冻结专家参数

        # 数据加载
        self.train_loader = self.load_dataset()

        # 训练配置
        self.num_epochs = 20
        self.grad_accum_steps = 4

        # 优化器配置
        self.optimizer = self._build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()
        self.ds =  MsDataset.load('wufaxianshi/wukong', subset_name='default', split='train')

    def _freeze_experts(self):
        """冻结所有专家参数，只保留门控可训练"""
        gate_pattern = re.compile(r'.*gate.*')
        for name, param in self.pipeline.pipe.unet.named_parameters():
            param.requires_grad = bool(gate_pattern.match(name))

    def _get_gate_params(self):
        """获取所有门控参数"""
        return [
            param
            for name, param in self.pipeline.pipe.unet.named_parameters()
            if "gate" in name and param.requires_grad 
        ]

    def _build_optimizer(self):
        """创建只针对门控参数的优化器"""
        gate_params = self._get_gate_params()
        return torch.optim.AdamW(
            gate_params, lr=3e-5, weight_decay=1e-4, betas=(0.9, 0.999)
        )

    def load_dataset(self) -> DataLoader:
        """实现你的数据加载逻辑（示例结构）"""
        # 这里需要替换为实际的数据加载代码
        raise NotImplementedError("请实现数据加载逻辑")
    def _compute_loss(self, outputs, targets):
        base_loss = F.mse_loss(outputs, targets)
        # 添加负载均衡损失（示例）
        load_imbalance_loss = self.pipeline.get_load_imbalance_loss()
        return base_loss + 0.1 * load_imbalance_loss

    def auto_grad_accum(self):
        """根据GPU显存使用率自动调整累积步数"""
        mem_used = torch.cuda.memory_allocated() / 1e9
        if mem_used > 12: 
            return max(1, self.grad_accum_steps + 1)
        else:
            return max(4, self.grad_accum_steps - 1)
    
    def _analyze_routing(self):
        stats = self.pipeline.pipe.unet.get_routing_stats()
        # 增加可视化输出
        plt.figure(figsize=(10,6))
        plt.bar(range(len(stats)), stats.values(), tick_label=stats.keys())
        plt.title('Expert Utilization Distribution')
        plt.savefig(f'routing_stats_epoch{self.current_epoch}.png')

    def _train_step(self, batch: dict) -> float:
        """单步训练流程"""
        self.pipeline.pipe.train()

        # 数据预处理（需要根据实际情况实现）
        inputs = self._tokenize(batch["prompt"])  # 文本编码
        targets = self._preprocess_images(batch["images"])  # 图像预处理

        # 混合精度训练上下文
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # 前向传播
            outputs = self.pipeline(
                input_ids=inputs, output_type="tensor", return_dict=True
            ).images

            # 损失计算
            loss = self._compute_loss(outputs, targets)
            loss = loss / self.grad_accum_steps  # 梯度累积归一化

        # 反向传播
        self.scaler.scale(loss).backward()
        return loss.item()

    def train(self):
        """增强版主训练循环"""
        # 初始化训练监控
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.train_loss_history = []
        self.val_metrics_history = []

        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=3e-4,
            total_steps=self.num_epochs * len(self.train_loader),
            pct_start=0.3
        )

        # 主训练循环
        with tqdm(total=self.num_epochs, desc="Training Progress", unit='epoch') as main_pbar:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                epoch_loss = 0.0
                self.optimizer.zero_grad()

                # 创建批次进度条
                batch_pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch}/{self.num_epochs}",
                    leave=False,
                    dynamic_ncols=True
                )

                try:
                    for step, batch in enumerate(batch_pbar):
                        # 动态调整梯度累积
                        self.grad_accum_steps = self.auto_grad_accum()

                        # 执行训练步骤
                        step_loss = self._train_step(batch)
                        epoch_loss += step_loss * self.grad_accum_steps

                        # 梯度更新条件判断
                        if (step + 1) % self.grad_accum_steps == 0:
                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self._get_gate_params(), 1.0)
                            
                            # 参数更新
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.scheduler.step()
                            self.optimizer.zero_grad()

                            # 更新批次进度条
                            batch_pbar.set_postfix({
                                'loss': f"{step_loss:.4f}",
                                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                                'grad_accum': self.grad_accum_steps
                            })

                    # 计算平均epoch损失
                    avg_epoch_loss = epoch_loss / len(self.train_loader)
                    self.train_loss_history.append(avg_epoch_loss)

                    # 验证与保存
                    if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                        val_metrics = self._validate()
                        self.val_metrics_history.append(val_metrics)
                        
                        # 自适应保存策略
                        if avg_epoch_loss < self.best_loss:
                            self.best_loss = avg_epoch_loss
                            self.save_checkpoint(is_best=True)
                        else:
                            self.save_checkpoint(is_best=False)

                    # 更新主进度条
                    main_pbar.update(1)
                    main_pbar.set_postfix({
                        'train_loss': f"{avg_epoch_loss:.4f}",
                        'best_loss': f"{self.best_loss:.4f}",
                        'val_divergence': f"{val_metrics.get('divergence', 0):.4f}"
                    })

                except torch.cuda.OutOfMemoryError:
                    print(f"检测到显存溢出，自动降低梯度累积步数到{self.grad_accum_steps}")
                    self.grad_accum_steps += 2
                    continue

                finally:
                    # 确保释放资源
                    batch_pbar.close()
                    torch.cuda.empty_cache()


    def _tokenize(self, text: list) -> torch.Tensor:
        """实现文本编码（示例）"""
        return self.pipeline.tokenizer(
            text,
            return_tensors="pt",
            padding=True
        ).input_ids.to(self.device)

    def _preprocess_images(self, images: list) -> torch.Tensor:
        """图像预处理（示例）"""
        images = torch.stack([torch.tensor(img) for img in images])
        images = images * 2 - 1
        return images

    def save_checkpoint(self):
        # 优化检查点存储
        state_dict = {
            'model': self.pipeline.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'routing_history': self.routing_buffer
        }
        # 使用异步保存
        torch.save(state_dict, 'checkpoint.pth', _use_new_zipfile_serialization=True)

    def _validate(self):
        with torch.no_grad():
            # 计算专家输出差异度
            divergence = self.pipeline.calculate_expert_divergence()
            print(f"Expert KL-Divergence: {divergence:.4f}")
            # 当差异度<0.1时触发专家重启
            if divergence < 0.1:
                self.pipeline.reinitialize_low_activity_experts()

    def _analyze_routing(self):
        """实现路由分析逻辑"""
        # 示例：统计各专家的使用频率
        print(self.pipeline.pipe.unet.get_routing_stats())


if __name__ == "__main__":
    # 使用示例
    trainer = GatedTrainer(config_path="path/to/config", data_path="path/to/dataset")
    trainer.train()

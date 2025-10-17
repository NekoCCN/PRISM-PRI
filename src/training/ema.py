# src/training/ema.py
"""
模型平滑技术：EMA 和 SWA
"""
import torch
from copy import deepcopy


class ModelEMA:
    """
    指数移动平均 (Exponential Moving Average)

    用法：
        ema = ModelEMA(model, decay=0.9999)

        # 训练循环中
        for batch in dataloader:
            loss = train_step(model, batch)
            optimizer.step()
            ema.update(model)  # 每步更新

        # 推理时使用EMA模型
        predictions = ema.ema(input)
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # 创建模型副本
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = decay

        # 冻结EMA模型参数
        for p in self.ema.parameters():
            p.requires_grad_(False)

        print(f"EMA INIT SUCCESSFUL (decay={decay})")

    def update(self, model):
        """在每个训练step后调用"""
        with torch.no_grad():
            self.updates += 1

            d = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000.0)))

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(self, model):
        for k in model.__dict__.keys():
            if not k.startswith('_') and k != 'ema':
                setattr(self.ema, k, getattr(model, k))

    def state_dict(self):
        return {
            'ema': self.ema.state_dict(),
            'updates': self.updates,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict['ema'])
        self.updates = state_dict['updates']
        self.decay = state_dict['decay']


class SWA:
    """
    随机权重平均 (Stochastic Weight Averaging)

    用法：
        swa = SWA(model)

        # 在训练后期（如最后25%的epochs）启用
        if epoch >= swa_start_epoch:
            swa.update(model)

        # 训练结束后使用SWA模型
        predictions = swa.model(input)
    """

    def __init__(self, model):
        self.model = deepcopy(model).eval()
        self.n_averaged = 0

        for p in self.model.parameters():
            p.requires_grad_(False)

        print(f"✅ SWA初始化完成")

    def update(self, model):
        """更新SWA模型（简单均值）"""
        with torch.no_grad():
            for p_swa, p_model in zip(self.model.parameters(), model.parameters()):
                if p_swa.dtype.is_floating_point:
                    p_swa.data = (p_swa.data * self.n_averaged + p_model.data) / (self.n_averaged + 1)

            self.n_averaged += 1

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'n_averaged': self.n_averaged
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.n_averaged = state_dict['n_averaged']
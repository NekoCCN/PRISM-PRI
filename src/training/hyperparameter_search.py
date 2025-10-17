"""
自动超参数搜索
使用Optuna找到最佳配置
"""
import optuna
from optuna.trial import Trial
import torch
from src.training.train_stage2_ultimate import *


def objective(trial: Trial):
    """
    Optuna目标函数

    搜索空间：
    - learning_rate
    - batch_size
    - ohem_ratio
    - focal_alpha/gamma
    - warmup_epochs
    """

    # 定义搜索空间
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
        'ohem_ratio': trial.suggest_float('ohem_ratio', 0.5, 0.9),
        'focal_alpha': trial.suggest_float('focal_alpha', 0.1, 0.5),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 3, 10),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
    }

    # 更新全局配置
    STAGE2_CONFIG.update(config)

    # 运行训练（缩短版）
    STAGE2_CONFIG['epochs'] = 20  # 快速试验

    try:
        # 执行训练
        best_loss = run_training_stage2_ultimate_for_optuna(trial, config)
        return best_loss

    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')


def run_training_stage2_ultimate_for_optuna(trial, config):
    """
    为Optuna定制的训练函数
    添加中间报告和剪枝
    """
    # ... 简化的训练代码 ...

    for epoch in range(20):
        # 训练一个epoch
        avg_loss = train_one_epoch()

        # 报告中间结果
        trial.report(avg_loss, epoch)

        # 剪枝：如果效果很差就提前停止
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_loss


def search_hyperparameters(n_trials=50):
    """
    执行超参数搜索

    Args:
        n_trials: 试验次数
    """
    print(f"🔍 开始超参数搜索 ({n_trials} trials)...")

    # 创建study
    study = optuna.create_study(
        study_name='prism-stage2-hp-search',
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler()
    )

    # 执行搜索
    study.optimize(objective, n_trials=n_trials, timeout=3600 * 24)  # 24小时

    # 打印结果
    print("\n" + "=" * 80)
    print("🎉 搜索完成！")
    print("=" * 80)
    print(f"\n最佳Trial: {study.best_trial.number}")
    print(f"最佳Loss: {study.best_value:.4f}")
    print(f"\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 可视化
    try:
        import matplotlib.pyplot as plt

        # 优化历史
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig('hp_search_history.png')

        # 参数重要性
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.savefig('hp_param_importance.png')

        print("\n可视化已保存")
    except:
        pass

    return study.best_params


# 使用
if __name__ == '__main__':
    best_params = search_hyperparameters(n_trials=50)

    # 保存最佳配置
    import json

    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
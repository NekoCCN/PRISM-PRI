import argparse


def main():
    parser = argparse.ArgumentParser(
        description="PRISM 级联检测系统命令行工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('mode',
                        choices=['train-stage1', 'gen-proposals', 'train-stage2', 'train-stage2-ultimate', 'serve'],
                        help="选择运行模式:\n"
                             "  train-stage1:         (第一步) 训练YOLO提议网络。\n"
                             "  gen-proposals:        (第二步) 为阶段二生成候选框。\n"
                             "  train-stage2:         (第三步) 训练ROI精炼网络（基础版）。\n"
                             "  train-stage2-ultimate:(第三步) 训练ROI精炼网络（终极优化版）🚀。\n"
                             "  serve:                (第四步) 启动API服务器进行部署。")

    args = parser.parse_args()

    if args.mode == 'train-stage1':
        from src.training.train_stage1 import run_training_stage1
        print("--- 进入 [阶段一：训练提议网络] 模式 ---")
        run_training_stage1()

    elif args.mode == 'gen-proposals':
        from src.utils.generate_proposals import run_proposal_generation
        print("--- 进入 [阶段二：生成候选框] 模式 ---")
        run_proposal_generation()

    elif args.mode == 'train-stage2':
        from src.training.train_stage2 import run_training_stage2
        print("--- 进入 [阶段三：训练精炼网络 - 基础版] 模式 ---")
        run_training_stage2()

    elif args.mode == 'train-stage2-ultimate':
        from src.training.train_stage2_ultimate import run_training_stage2_ultimate
        print("--- 进入 [阶段三：训练精炼网络 - 终极优化版] 模式 🚀 ---")
        run_training_stage2_ultimate()

    elif args.mode == 'serve':
        print("--- 进入 [阶段四：部署API服务] 模式 ---")
        import uvicorn
        from src.server import app
        from src.config import SERVER_CONFIG
        print(f"服务器将在 http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']} 上启动")
        uvicorn.run(app, host=SERVER_CONFIG['host'], port=SERVER_CONFIG['port'])


if __name__ == "__main__":
    main()
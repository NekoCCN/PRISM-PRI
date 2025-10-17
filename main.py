# main.py
"""
PRISM Cascade Detection System - Command Line Interface

Complete unified entry point for all PRISM operations including
training, data preparation, evaluation, analysis, and deployment.
"""
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_stage1_parser(subparsers):
    """Setup Stage 1 training parser."""
    parser = subparsers.add_parser(
        'train-stage1',
        help='Train Stage 1 YOLO proposal network'
    )
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    return parser


def setup_proposals_parser(subparsers):
    """Setup proposal generation parser."""
    parser = subparsers.add_parser(
        'gen-proposals',
        help='Generate region proposals for Stage 2 training'
    )
    parser.add_argument('--conf-thresh', type=float, default=0.05,
                        help='Confidence threshold for proposals')
    return parser


def setup_stage2_parser(subparsers):
    """Setup Stage 2 training parser."""
    parser = subparsers.add_parser(
        'train-stage2',
        help='Train Stage 2 ROI refinement network'
    )
    parser.add_argument('--basic', action='store_true',
                        help='Use basic training mode without advanced optimizations')
    parser.add_argument('--no-validation', action='store_true',
                        help='Disable validation during training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    return parser


def setup_stage2_ultimate_parser(subparsers):
    """Setup Stage 2 ultimate training parser (alias for advanced mode)."""
    parser = subparsers.add_parser(
        'train-stage2-ultimate',
        help='Train Stage 2 with all optimizations (same as train-stage2 without --basic)'
    )
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--no-validation', action='store_true',
                        help='Disable validation during training')
    return parser


def setup_eval_parser(subparsers):
    """Setup evaluation parser."""
    parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained model on test set'
    )
    parser.add_argument('--use-ema', action='store_true',
                        help='Use EMA model for evaluation')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    return parser


def setup_serve_parser(subparsers):
    """Setup deployment server parser."""
    parser = subparsers.add_parser(
        'serve',
        help='Start API server for deployment'
    )
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='Server port')
    parser.add_argument('--use-ema', action='store_true',
                        help='Use EMA model for inference')
    return parser


def setup_inference_parser(subparsers):
    """Setup local inference parser."""
    parser = subparsers.add_parser(
        'infer',
        help='Run local inference on images'
    )
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to directory containing images')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--use-ema', action='store_true',
                        help='Use EMA model for inference')
    parser.add_argument('--use-tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output JSON file for batch inference')
    return parser


def setup_gradio_parser(subparsers):
    """Setup Gradio web UI parser."""
    parser = subparsers.add_parser(
        'gradio',
        help='Launch Gradio web interface'
    )
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--port', type=int, default=7860,
                        help='Server port')
    parser.add_argument('--share', action='store_true',
                        help='Create public link')
    return parser


def setup_ensemble_parser(subparsers):
    """Setup ensemble inference parser."""
    parser = subparsers.add_parser(
        'ensemble',
        help='Run ensemble inference with multiple models'
    )
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to directory')
    parser.add_argument('--models', type=str, nargs='+',
                        help='List of model paths')
    parser.add_argument('--strategy', type=str, default='weighted_average',
                        choices=['weighted_average', 'voting', 'wbf'],
                        help='Ensemble strategy')
    parser.add_argument('--weights', type=float, nargs='+',
                        help='Model weights for ensemble')
    parser.add_argument('--output', type=str, default='ensemble_results.json',
                        help='Output file')
    return parser


def setup_data_check_parser(subparsers):
    """Setup data quality check parser."""
    parser = subparsers.add_parser(
        'data-check',
        help='Check dataset quality and statistics'
    )
    parser.add_argument('--data-yaml', type=str,
                        help='Path to data.yaml (default: from config)')
    parser.add_argument('--output-dir', type=str, default='data_quality_results',
                        help='Output directory')
    return parser


def setup_error_analysis_parser(subparsers):
    """Setup error analysis parser."""
    parser = subparsers.add_parser(
        'error-analysis',
        help='Analyze model errors and find hard cases'
    )
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--output-dir', type=str, default='error_analysis',
                        help='Output directory')
    return parser


def setup_gradcam_parser(subparsers):
    """Setup Grad-CAM visualization parser."""
    parser = subparsers.add_parser(
        'gradcam',
        help='Generate Grad-CAM visualizations'
    )
    parser.add_argument('--image', type=str,
                        help='Path to input image (for single image)')
    parser.add_argument('--dir', type=str,
                        help='Path to directory (for batch processing)')
    parser.add_argument('--output', type=str, default='gradcam_output.png',
                        help='Output path for single image')
    parser.add_argument('--output-dir', type=str, default='gradcam_results',
                        help='Output directory for batch processing')
    parser.add_argument('--target-layer', type=str,
                        help='Target layer name (auto-detect if not specified)')
    return parser


def setup_hp_search_parser(subparsers):
    """Setup hyperparameter search parser."""
    parser = subparsers.add_parser(
        'hp-search',
        help='Run hyperparameter search using Optuna'
    )
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials')
    parser.add_argument('--timeout', type=int, default=86400,
                        help='Timeout in seconds (default: 24h)')
    parser.add_argument('--study-name', type=str, default='prism-hp-search',
                        help='Study name')
    return parser


def setup_export_parser(subparsers):
    """Setup model export parser."""
    parser = subparsers.add_parser(
        'export',
        help='Export model to ONNX, TorchScript, or other formats'
    )
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'tensorrt'],
                        help='Export format')
    parser.add_argument('--model', type=str,
                        help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='exported_model',
                        help='Output path')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model')
    return parser


# Handler functions

def handle_train_stage1(args):
    """Handle Stage 1 training."""
    from src.training.train_stage1 import run_training_stage1
    from src.config import STAGE1_CONFIG

    logger.info("=" * 80)
    logger.info("Stage 1: Training YOLO Proposal Network")
    logger.info("=" * 80)

    if args.epochs:
        STAGE1_CONFIG['epochs'] = args.epochs
    if args.batch_size:
        STAGE1_CONFIG['batch_size'] = args.batch_size
    if hasattr(args, 'img_size') and args.img_size:
        STAGE1_CONFIG['img_size'] = args.img_size

    run_training_stage1()


def handle_gen_proposals(args):
    """Handle proposal generation."""
    from src.utils.generate_proposals import run_proposal_generation
    from src.config import STAGE1_CONFIG

    logger.info("=" * 80)
    logger.info("Stage 2 Preparation: Generating Region Proposals")
    logger.info("=" * 80)

    original_thresh = STAGE1_CONFIG.get('confidence_threshold', 0.1)
    STAGE1_CONFIG['confidence_threshold'] = args.conf_thresh

    try:
        run_proposal_generation()
    finally:
        STAGE1_CONFIG['confidence_threshold'] = original_thresh


def handle_train_stage2(args):
    """Handle Stage 2 training."""
    from src.training.train_stage2_unified import run_training_stage2_unified
    from src.config import STAGE2_CONFIG

    logger.info("=" * 80)
    logger.info("Stage 2: Training ROI Refinement Network")
    logger.info("=" * 80)

    if args.epochs:
        STAGE2_CONFIG['epochs'] = args.epochs
    if args.batch_size:
        STAGE2_CONFIG['batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr:
        STAGE2_CONFIG['learning_rate'] = args.lr

    mode = "Basic" if getattr(args, 'basic', False) else "Advanced"
    val_status = "Disabled" if args.no_validation else "Enabled"
    logger.info(f"Training mode: {mode}")
    logger.info(f"Validation: {val_status}")

    run_training_stage2_unified(
        use_advanced=not getattr(args, 'basic', False),
        use_validation=not args.no_validation
    )


def handle_train_stage2_ultimate(args):
    """Handle Stage 2 ultimate training (alias for advanced mode)."""
    # Convert to regular stage2 args with advanced mode
    args.basic = False
    handle_train_stage2(args)


def handle_evaluate(args):
    """Handle model evaluation."""
    from src.evaluation.evaluate_model import run_evaluation

    logger.info("=" * 80)
    logger.info("Model Evaluation on Test Set")
    logger.info("=" * 80)

    run_evaluation(use_ema=args.use_ema, output_dir=args.output_dir)


def handle_serve(args):
    """Handle API server deployment."""
    from src.config import SERVER_CONFIG
    import uvicorn
    from src.server import app

    logger.info("=" * 80)
    logger.info("Starting PRISM API Server")
    logger.info("=" * 80)

    host = args.host
    port = args.port

    logger.info(f"Server address: http://{host}:{port}")
    logger.info(f"EMA model: {'Enabled' if args.use_ema else 'Disabled'}")
    logger.info("Press CTRL+C to stop the server")

    uvicorn.run(app, host=host, port=port)


def handle_inference(args):
    """Handle local inference."""
    from src.inference.local_inference import LocalInference

    logger.info("=" * 80)
    logger.info("Local Inference")
    logger.info("=" * 80)

    if not args.image and not args.dir:
        logger.error("Error: Please specify either --image or --dir")
        sys.exit(1)

    inferencer = LocalInference(use_ema=args.use_ema, use_tta=args.use_tta)

    if args.image:
        logger.info(f"Processing single image: {args.image}")
        detections = inferencer.predict_single(args.image, conf_thresh=args.conf_thresh)

        import json
        logger.info("\nDetection results:")
        logger.info(json.dumps(detections, indent=2, ensure_ascii=False))

    elif args.dir:
        logger.info(f"Processing directory: {args.dir}")
        inferencer.predict_batch(args.dir, output_json=args.output, conf_thresh=args.conf_thresh)


def handle_gradio(args):
    """Handle Gradio web UI."""
    logger.info("=" * 80)
    logger.info("Starting Gradio Web Interface")
    logger.info("=" * 80)

    from src.inference.gradio_app import create_ui

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


def handle_ensemble(args):
    """Handle ensemble inference."""
    from src.inference.ensemble import EnsembleInference
    from pathlib import Path

    logger.info("=" * 80)
    logger.info("Ensemble Inference")
    logger.info("=" * 80)

    if not args.image and not args.dir:
        logger.error("Error: Please specify either --image or --dir")
        sys.exit(1)

    # Use default models if not specified
    if not args.models:
        from src.config import STAGE2_CONFIG
        default_models = [
            STAGE2_CONFIG['weights_path'],
            STAGE2_CONFIG['weights_path'].replace('.pth', '_ema.pth'),
            STAGE2_CONFIG['weights_path'].replace('.pth', '_swa.pth')
        ]
        args.models = [m for m in default_models if Path(m).exists()]

    logger.info(f"Using {len(args.models)} models for ensemble")

    ensemble = EnsembleInference(
        model_paths=args.models,
        weights=args.weights,
        strategy=args.strategy
    )

    # Run inference
    if args.image:
        logger.info(f"Processing image: {args.image}")
        # Implementation depends on your ensemble module
    elif args.dir:
        logger.info(f"Processing directory: {args.dir}")
        # Implementation depends on your ensemble module


def handle_data_check(args):
    """Handle data quality check."""
    from src.utils.data_quality_check import DataQualityChecker
    from src.config import DATA_YAML

    logger.info("=" * 80)
    logger.info("Data Quality Check")
    logger.info("=" * 80)

    data_yaml = args.data_yaml or DATA_YAML
    logger.info(f"Checking dataset: {data_yaml}")

    checker = DataQualityChecker(data_yaml)
    checker.run_checks()


def handle_error_analysis(args):
    """Handle error analysis."""
    import json
    from analysis.error_analysis import ErrorAnalyzer
    from src.config import DATA_YAML
    import yaml

    logger.info("=" * 80)
    logger.info("Error Analysis")
    logger.info("=" * 80)

    # Load class names
    with open(DATA_YAML) as f:
        class_names = yaml.safe_load(f)['names']

    # Load predictions
    with open(args.predictions) as f:
        predictions = json.load(f)

    analyzer = ErrorAnalyzer(class_names)

    # Process predictions
    # (Implementation depends on your prediction format)

    # Generate report
    analyzer.generate_report(output_dir=args.output_dir)


def handle_gradcam(args):
    """Handle Grad-CAM visualization."""
    from src.analysis.gradcam import generate_gradcam_for_image, batch_generate_gradcam
    from src.models.refiner import ROIRefinerModel
    from src.config import DEVICE, STAGE2_CONFIG
    import torch
    from pathlib import Path

    logger.info("=" * 80)
    logger.info("Grad-CAM Visualization")
    logger.info("=" * 80)

    if not args.image and not args.dir:
        logger.error("Error: Please specify either --image or --dir")
        sys.exit(1)

    # Load model
    logger.info("Loading model")
    model = ROIRefinerModel(device=DEVICE)

    # Load weights (prefer EMA)
    model_path = STAGE2_CONFIG['weights_path']
    ema_path = STAGE2_CONFIG['weights_path'].replace('.pth', '_ema.pth')
    if Path(ema_path).exists():
        model_path = ema_path
        logger.info("Using EMA model")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Model loaded from: {model_path}")

    # Find target layer if specified
    target_layer = None
    if args.target_layer:
        try:
            target_layer = dict(model.named_modules())[args.target_layer]
            logger.info(f"Using specified target layer: {args.target_layer}")
        except KeyError:
            logger.error(f"Layer '{args.target_layer}' not found in model")
            logger.info("Available convolutional layers:")
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    logger.info(f"  - {name}")
            sys.exit(1)

    # Generate Grad-CAM
    try:
        if args.image:
            # Single image mode
            logger.info(f"Processing single image: {args.image}")
            cam_image = generate_gradcam_for_image(
                model=model,
                image_path=args.image,
                output_path=args.output,
                target_layer=target_layer
            )
            logger.info(f"Output saved to: {args.output}")

        elif args.dir:
            # Batch mode
            logger.info(f"Processing directory: {args.dir}")
            batch_generate_gradcam(
                model=model,
                image_dir=args.dir,
                output_dir=args.output_dir,
                target_layer=target_layer
            )
            logger.info(f"Outputs saved to: {args.output_dir}")

        logger.info("=" * 80)
        logger.info("Grad-CAM Generation Complete")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        raise


def handle_hp_search(args):
    """Handle hyperparameter search."""
    from src.training.hyperparameter_search import search_hyperparameters

    logger.info("=" * 80)
    logger.info("Hyperparameter Search")
    logger.info("=" * 80)
    logger.info(f"Number of trials: {args.n_trials}")
    if args.timeout:
        logger.info(f"Timeout: {args.timeout} seconds ({args.timeout / 3600:.1f} hours)")
    logger.info(f"Study name: {args.study_name}")
    logger.info("")
    logger.info("This will run multiple short training trials to find optimal hyperparameters.")
    logger.info("Results will be saved to: hyperparameter_search_results.json")
    logger.info("")

    try:
        best_params = search_hyperparameters(
            n_trials=args.n_trials,
            timeout=args.timeout
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("Search Complete - Next Steps")
        logger.info("=" * 80)
        logger.info("1. Review results in: hyperparameter_search_results.json")
        logger.info("2. Check visualizations: hp_search_history.png, hp_param_importance.png")
        logger.info("3. Update config with best parameters")
        logger.info("4. Retrain with full epochs:")
        logger.info(
            f"   python main.py train-stage2 --lr {best_params.get('learning_rate', 'N/A'):.2e} --batch-size {best_params.get('batch_size', 'N/A')}")

    except KeyboardInterrupt:
        logger.info("\nSearch interrupted by user")
        logger.info("Partial results may be available in hyperparameter_search_results.json")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise


def handle_export(args):
    """Handle model export."""
    from src.models.refiner import ROIRefinerModel
    from src.config import DEVICE, STAGE2_CONFIG
    import torch
    from pathlib import Path

    logger.info("=" * 80)
    logger.info(f"Exporting Model to {args.format.upper()}")
    logger.info("=" * 80)

    # Load model
    model = ROIRefinerModel(device=DEVICE)
    model_path = args.model or STAGE2_CONFIG['weights_path']

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

    output_path = Path(args.output)

    if args.format == 'onnx':
        output_file = output_path.with_suffix('.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            output_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['class_logits', 'bbox_deltas'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )

        if args.simplify:
            try:
                import onnx
                from onnxsim import simplify
                model_onnx = onnx.load(output_file)
                model_simp, check = simplify(model_onnx)
                onnx.save(model_simp, output_file)
                logger.info("ONNX model simplified")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

        logger.info(f"Model exported to: {output_file}")

    elif args.format == 'torchscript':
        output_file = output_path.with_suffix('.pt')
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_file)
        logger.info(f"Model exported to: {output_file}")

    elif args.format == 'tensorrt':
        logger.error("TensorRT export not implemented yet")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PRISM Cascade Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Training:
    python main.py train-stage1
    python main.py gen-proposals
    python main.py train-stage2
    python main.py train-stage2-ultimate  # Same as train-stage2 (advanced mode)
    python main.py train-stage2 --basic   # Basic mode

  Evaluation:
    python main.py evaluate --use-ema
    python main.py error-analysis --predictions results.json

  Inference:
    python main.py infer --image test.jpg
    python main.py infer --dir images/ --use-tta
    python main.py ensemble --dir images/ --strategy voting

  Deployment:
    python main.py serve --port 8000
    python main.py gradio --share

  Analysis & Tools:
    python main.py data-check
    python main.py gradcam --image test.jpg
    python main.py hp-search --n-trials 100
    python main.py export --format onnx

For more information, visit: https://github.com/yourrepo/prism
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training commands
    setup_stage1_parser(subparsers)
    setup_proposals_parser(subparsers)
    setup_stage2_parser(subparsers)
    setup_stage2_ultimate_parser(subparsers)

    # Evaluation commands
    setup_eval_parser(subparsers)
    setup_error_analysis_parser(subparsers)

    # Inference commands
    setup_inference_parser(subparsers)
    setup_ensemble_parser(subparsers)

    # Deployment commands
    setup_serve_parser(subparsers)
    setup_gradio_parser(subparsers)

    # Analysis & visualization
    setup_gradcam_parser(subparsers)
    setup_data_check_parser(subparsers)

    # Advanced tools
    setup_hp_search_parser(subparsers)
    setup_export_parser(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate handler
    handlers = {
        'train-stage1': handle_train_stage1,
        'gen-proposals': handle_gen_proposals,
        'train-stage2': handle_train_stage2,
        'train-stage2-ultimate': handle_train_stage2_ultimate,
        'evaluate': handle_evaluate,
        'serve': handle_serve,
        'infer': handle_inference,
        'gradio': handle_gradio,
        'ensemble': handle_ensemble,
        'data-check': handle_data_check,
        'error-analysis': handle_error_analysis,
        'gradcam': handle_gradcam,
        'hp-search': handle_hp_search,
        'export': handle_export
    }

    try:
        handler = handlers[args.command]
        handler(args)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
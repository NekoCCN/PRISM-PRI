"""
Gradio Web UI for PRISM
"""
import gradio as gr
import torch
from PIL import Image
import yaml
from pathlib import Path

from src.config import DEVICE, DATA_YAML
from src.inference.local_inference import LocalInference

# 全局推理器
inferencer = None


def load_inferencer():
    """加载推理器"""
    global inferencer
    if inferencer is None:
        print("🚀 初始化Gradio UI...")
        inferencer = LocalInference(use_ema=True, use_tta=False)
    return inferencer


def predict_image(image, conf_threshold):
    """Gradio预测函数"""
    inf = load_inferencer()

    # 临时保存
    temp_path = Path("temp_gradio.jpg")
    image.save(temp_path)

    try:
        detections = inf.predict_single(
            temp_path,
            conf_thresh=conf_threshold,
            save_viz=True
        )

        # 加载可视化结果
        result_path = temp_path.parent / f"{temp_path.stem}_result.png"
        if result_path.exists():
            result_img = Image.open(result_path)
        else:
            result_img = image

        # 格式化输出
        if len(detections) == 0:
            text_output = "✅ 未检测到缺陷"
        else:
            text_output = f"🔍 检测到 {len(detections)} 个缺陷:\n\n"
            for i, det in enumerate(detections, 1):
                text_output += f"{i}. **{det['class']}** (置信度: {det['confidence']:.2%})\n"
                text_output += f"   位置: {[f'{x:.1f}' for x in det['bbox']]}\n\n"

        return result_img, text_output

    finally:
        # 清理
        if temp_path.exists():
            temp_path.unlink()


def create_ui():
    with gr.Blocks(title="PRISM 缺陷检测系统") as demo:
        gr.Markdown("""
        # 🔬 PRISM 缺陷检测系统
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="📤 上传图片")
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="🎯 置信度阈值"
                )
                detect_btn = gr.Button("🚀 开始检测", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="📊 检测结果")
                output_text = gr.Markdown(label="详细信息")

        detect_btn.click(
            fn=predict_image,
            inputs=[input_image, conf_slider],
            outputs=[output_image, output_text]
        )

        gr.Markdown("""
        ---
        ### 💡 使用提示
        1. 上传清晰的图片
        2. 调整置信度阈值（默认0.5）
        3. 点击"开始检测"
        4. 查看检测结果和可视化
        """)

    return demo


if __name__ == '__main__':
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
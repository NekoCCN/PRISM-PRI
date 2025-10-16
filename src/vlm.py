import google.generativeai as genai
import json
from PIL import Image
from src.config import VLM_CONFIG


class VisualLanguageModel:
    """
    使用 Google Gemini Pro Vision API 实现的视觉语言模型分析器。
    """

    def __init__(self):
        print("  - 初始化后端层: VLM 智能分析 (Gemini)...")
        self.api_key = VLM_CONFIG.get("gemini_api_key")
        if not self.api_key:
            raise ValueError("错误: 未在环境变量中找到 GEMINI_API_KEY。请在 .env 文件中设置它。")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(VLM_CONFIG['model_name'])
        print("    Gemini VLM 初始化成功。")

    def analyze(self, roi_image: Image.Image, defect_type: str, confidence: float, historical_data: str) -> dict:
        """
        对单个缺陷ROI进行分析，并返回结构化的JSON报告。

        Args:
            roi_image: 包含缺陷的PIL Image对象。
            defect_type: 阶段二检测出的缺陷类别。
            confidence: 阶段二的置信度分数。
            historical_data: 关于该设备的文本历史信息。

        Returns:
            一个包含详细分析的字典。
        """
        print(f"    VLM 正在分析缺陷: {defect_type}...")
        prompt_parts = [
            "You are a world-class expert in wind turbine blade maintenance and material science. Your task is to analyze an image of a potential blade defect and provide a structured JSON report. Do not include ```json markdown wrapper in your response.",
            "Input Image:",
            roi_image,
            "\nAnalysis Request:",
            f"""
            - **Detected Defect Type (from Stage 2 model):** {defect_type}
            - **Detection Confidence:** {confidence:.2f}
            - **Historical Context:** {historical_data}

            Based on the provided image and context, please perform the following analysis and respond ONLY with a single valid JSON object containing these fields:
            1.  **detailed_description**: A detailed visual description of the defect in the image (e.g., 'A transverse hairline crack, approximately 5cm long, located near the leading edge, showing slight signs of water ingress.').
            2.  **severity_assessment**: Assess the severity on a scale of 'Low', 'Medium', 'High', or 'Critical'.
            3.  **risk_analysis**: Briefly explain the potential risks if this defect is not addressed (e.g., 'Risk of crack propagation leading to structural failure, potential for moisture ingress to corrode internal components.').
            4.  **maintenance_recommendation**: Provide a clear, actionable maintenance recommendation (e.g., 'Immediate on-site inspection by a qualified technician is required. Recommend grinding and patching the affected area within the next 30 days.').
            """,
        ]

        try:
            response = self.model.generate_content(prompt_parts)
            # 移除 Gemini 可能添加的 markdown 格式
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            report = json.loads(cleaned_response)
            print(f"    VLM 分析成功。评估严重性为: {report.get('severity_assessment')}")
            return report
        except Exception as e:
            print(f"    VLM 分析失败: {e}")
            return {
                "error": str(e),
                "detailed_description": "Failed to get analysis from VLM.",
                "severity_assessment": "Unknown",
                "risk_analysis": "Unknown",
                "maintenance_recommendation": "Manual inspection required due to VLM analysis failure.",
            }

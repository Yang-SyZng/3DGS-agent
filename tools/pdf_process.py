import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoModel, AutoTokenizer
from config import setting
import logging
import tempfile, fitz

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

class PDFProcess:
    def __init__(self):
        logger.info("正在初始化PDF Process...")

    def extract_text(pdf_path: str | Path) -> Optional[str]:
        """
        适用于纯文本PDF

        Args:
            pdf_path: PDF文件路径

        Returns:
            提取的文本内容
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            logger.error(f"PDF文件不存在: {pdf_path}")
            return None

        logger.info(f"提取PDF文本: {pdf_path.name}")

        try:
            doc = fitz.open(pdf_path)
            full_text = ""

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                full_text += text

            doc.close()

            logger.info(f"成功提取文本，长度: {len(full_text)} 字符")
            return full_text

        except Exception as e:
            logger.error(f"PDF文本提取失败: {str(e)}")
            return None
        

class PDFProcess_OCR:
    """ OCR技术基于最新开源百度：https://github.com/baidu/Unlimited-OCR """

    def __init__(self):
        logger.info("正在初始化OCR...")
        model_name = setting.OCR_MODEL_ID

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.eval().cuda()

    def infer(self,
            prompt: str = '<image>Multi page parsing.', 
            image_files: str | list[str] = None, 
            output_path: str = '', 
            image_size: int = 1024, 
            save_results: bool = True, 
            max_length: int = 32768, 
            tps_interval: int = 0, 
            no_repeat_ngram_size: int = 35, 
            ngram_window: int = 1024, 
            temperature: int = 0.0,
            dpi: int | None = None
        ) -> str:
        """执行多图像OCR推理。

        支持三种输入形式:
        1. 单个图像路径；
        2. 图像路径列表；
        3. PDF 路径。

        如果输入为 PDF，会先调用 pdf_to_images() 将 PDF 转换为图像列表，再进行推理。
        当 dpi 为 None 时，使用 pdf_to_images() 的默认 DPI 值。

        参数:
            prompt: 包含一个 <image> 占位符的文本提示。
            image_files: 单张图像路径、图像路径列表，或 PDF 路径。
            output_path: 保存结果文件的目录，仅当 save_results=True 时生效。
            image_size: 输入图像缩放尺寸，用于视觉特征提取。
            save_results: 是否保存生成文本和可视化结果。
            max_length: 最大生成令牌长度。
            tps_interval: TPSTextStreamer 的每秒令牌日志间隔。
            no_repeat_ngram_size: 生成时避免重复的 n-gram 大小。
            ngram_window: no-repeat n-gram 处理的滑动窗口大小。
            temperature: 生成温度，0.0 表示贪心解码。
            dpi: 仅当输入为 PDF 时生效，指定 PDF 转换为图像的 DPI。

        返回:
            outputs: 解码后的文本结果。
        """
        if image_files is None:
            raise ValueError("image_files 不能为空，必须提供图像路径、图像路径列表或 PDF 路径。")

        pdf_dpi = dpi if dpi is not None else 300

        output, ouput_token = self.model.infer_multi(
            self.tokenizer,
            prompt=prompt,
            image_files=image_files,
            output_path=output_path,
            image_size=image_size,
            save_results=save_results,
            max_length=max_length,
            tps_interval=tps_interval,
            no_repeat_ngram_size=no_repeat_ngram_size, 
            ngram_window=ngram_window,
            temperature=temperature
        )

        logger.info(f"成功提取所有信息，长度 {len(output)} 个字符，共 {ouput_token} 个token")

        return output

def pdf_to_images(self, pdf_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix='pdf_ocr_')
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f'page_{i+1:04d}.png')
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths

        
pdf_process = PDFProcess()

_pdf_process = None


def get_PDFProcess() -> PDFProcess:
    global _pdf_process

    if _pdf_process is None:
        _pdf_process = PDFProcess()

    return _pdf_process


class LazyPDFProcess:
    def __getattr__(self, name):
        return getattr(get_PDFProcess(), name)


embedding = LazyPDFProcess()

# PDFProcessTools = [
#     StructuredTool.from_function(pdf_process.extract_pdf_stats),
#     StructuredTool.from_function(pdf_process.detect_layout_blocks)
# ]
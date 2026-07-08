from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import fitz
import torch
from transformers import AutoModel, AutoTokenizer

from config import setting

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

"""
                    PDF输入
                       |
                       ↓
            Document Type Detection
                       |
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
   Text PDF        Scanned PDF     Mixed PDF
       ↓               ↓               ↓
 Text Parser          OCR        Layout Parser
       ↓               ↓               ↓
 Section          Text Layer     Element Extraction
 Detection          Restore            |
       ↓               ↓               ↓
                next operation
"""


PDF_TYPE_NATIVE_TEXT = "native_text"
PDF_TYPE_SCANNED_IMAGE = "scanned_image"
PDF_TYPE_MIXED_LAYOUT = "mixed_layout"


def _ensure_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件不是PDF: {path}")
    return path


class PDFTypeDetector:
    """根据文本密度和图片占比判断 PDF 类型。"""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = _ensure_pdf_path(pdf_path)
        self.metadata: dict[str, Any] = {}

    def extract_features(self) -> dict[str, Any]:
        logger.info("正在提取PDF特征: %s", self.pdf_path)

        with fitz.open(self.pdf_path) as doc:
            total_chars = 0
            text_density = []
            image_counts = []
            image_area_ratios = []
            text_block_count = 0
            image_block_count = 0

            for page in doc:
                text = page.get_text()
                chars = len(text.strip())
                total_chars += chars

                page_area = page.rect.width * page.rect.height
                text_density.append(chars / page_area if page_area > 0 else 0)

                images = page.get_images(full=True)
                image_counts.append(len(images))

                image_area = 0
                for image in images:
                    xref = image[0]
                    for rect in page.get_image_rects(xref):
                        image_area += rect.width * rect.height

                image_area_ratios.append(image_area / page_area if page_area > 0 else 0)

                for block in page.get_text("dict").get("blocks", []):
                    if block.get("type") == 0:
                        text_block_count += 1
                    elif block.get("type") == 1:
                        image_block_count += 1

            pages = len(doc)
            self.metadata = {
                "pdf_path": str(self.pdf_path),
                "metadata": doc.metadata,
                "pages": pages,
                "total_chars": total_chars,
                "chars_per_page": total_chars / pages if pages else 0,
                "avg_text_density": sum(text_density) / pages if pages else 0,
                "avg_image_count": sum(image_counts) / pages if pages else 0,
                "avg_image_ratio": sum(image_area_ratios) / pages if pages else 0,
                "text_block_count": text_block_count,
                "image_block_count": image_block_count,
            }

        return self.metadata

    def classify(self) -> dict[str, Any]:
        features = self.extract_features()
        total_chars = features["total_chars"]
        avg_image_count = features["avg_image_count"]
        image_block_count = features["image_block_count"]

        if total_chars == 0 and image_block_count > 0:
            doc_type = PDF_TYPE_SCANNED_IMAGE
            confidence = 0.95
            reason = "没有可提取文本且存在图片块，归类为扫描件或纯图片PDF。"
        elif total_chars > 0 and image_block_count == 0 and avg_image_count == 0:
            doc_type = PDF_TYPE_NATIVE_TEXT
            confidence = 0.95
            reason = "存在可提取文本且未检测到图片，归类为纯文本PDF。"
        else:
            doc_type = PDF_TYPE_MIXED_LAYOUT
            confidence = 0.85
            reason = "不是纯图片也不是纯文字，归类为图文混排PDF。"

        result = {
            **features,
            "doc_type": doc_type,
            "confidence": confidence,
            "parse_strategy": {
                PDF_TYPE_NATIVE_TEXT: "direct_text_extraction",
                PDF_TYPE_SCANNED_IMAGE: "ocr_extraction",
                PDF_TYPE_MIXED_LAYOUT: "layout_text_extraction",
            }[doc_type],
            "reason": reason,
        }
        logger.info("PDF类型判断完成: %s", result["doc_type"])
        return result


class NativeTextPDFProcess:
    """原生文本 PDF：直接使用 PyMuPDF 提取文本。"""

    def extract_text(self, pdf_path: str | Path) -> str:
        pdf_path = _ensure_pdf_path(pdf_path)
        full_text = []

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    full_text.append(f"[Page {page_num}]\n{text}")

        output = "\n\n".join(full_text)
        logger.info("提取完成，长度: %s 字符", len(output))
        return output


class ScannedPDFProcess:
    """扫描件 PDF：将页面转为图片后调用 OCR。"""

    def __init__(self):
        self._ocr: PDFProcessOCR | None = None

    @property
    def ocr(self) -> "PDFProcessOCR":
        if self._ocr is None:
            self._ocr = PDFProcessOCR()
        return self._ocr

    def extract_text(self, pdf_path: str | Path, dpi: int = 300) -> str:
        pdf_path = _ensure_pdf_path(pdf_path)
        image_dir: str | None = None

        try:
            image_paths, image_dir = pdf_to_images(pdf_path, dpi=dpi)
            output_path = Path(setting.pdf_process_save_dir) / pdf_path.stem
            output_path.mkdir(parents=True, exist_ok=True)
            return self.ocr.infer(
                image_files=image_paths,
                output_path=str(output_path),
                save_results=True,
            )
        finally:
            if image_dir:
                shutil.rmtree(image_dir, ignore_errors=True)


class MixedLayoutPDFProcess:
    """图文混排 PDF：保留版面块顺序提取文本，必要时可回退 OCR。"""

    def __init__(self):
        self.native_parser = NativeTextPDFProcess()
        self.scanned_parser = ScannedPDFProcess()

    def detect_layout_blocks(
        self,
        pdf_path: str | Path,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        pdf_path = _ensure_pdf_path(pdf_path)
        blocks: list[dict[str, Any]] = []

        with fitz.open(pdf_path) as doc:
            page_count = len(doc) if max_pages is None else min(len(doc), max_pages)
            for page_index in range(page_count):
                page = doc[page_index]
                for block in page.get_text("dict").get("blocks", []):
                    block_type = "text" if block.get("type") == 0 else "image"
                    item: dict[str, Any] = {
                        "page": page_index + 1,
                        "type": block_type,
                        "bbox": block.get("bbox"),
                    }
                    if block_type == "text":
                        lines = []
                        for line in block.get("lines", []):
                            spans = line.get("spans", [])
                            lines.append("".join(span.get("text", "") for span in spans))
                        item["text"] = "\n".join(line for line in lines if line.strip())
                    blocks.append(item)

        return blocks

    def extract_text(
        self,
        pdf_path: str | Path,
        min_text_chars: int = 500,
        use_ocr: bool = False,
        use_ocr_fallback: bool = True,
    ) -> str:
        if use_ocr:
            logger.info("混排PDF使用OCR解析: %s", pdf_path)
            return self.scanned_parser.extract_text(pdf_path)

        blocks = self.detect_layout_blocks(pdf_path)
        text_parts = [
            f"[Page {block['page']}]\n{block['text']}"
            for block in blocks
            if block["type"] == "text" and block.get("text")
        ]
        output = "\n\n".join(text_parts).strip()

        if use_ocr_fallback and len(output) < min_text_chars:
            logger.info("混排PDF文本较少，回退OCR解析: %s", pdf_path)
            return self.scanned_parser.extract_text(pdf_path)

        logger.info("混排PDF版面文本提取完成，长度: %s 字符", len(output))
        return output


class PDFProcessOCR:
    """OCR 技术基于百度 Unlimited-OCR。"""

    def __init__(self):
        if not setting.OCR_MODEL_ID:
            raise ValueError("OCR_MODEL_ID 未配置，无法初始化OCR模型。")

        logger.info("正在初始化OCR模型: %s", setting.OCR_MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(
            setting.OCR_MODEL_ID,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            setting.OCR_MODEL_ID,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.eval().cuda()

    def infer(
        self,
        prompt: str = "<image>Multi page parsing.",
        image_files: str | list[str] | None = None,
        output_path: str = "",
        image_size: int = 1024,
        save_results: bool = True,
        max_length: int = 32768,
        tps_interval: int = 0,
        no_repeat_ngram_size: int = 35,
        ngram_window: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        if image_files is None:
            raise ValueError("image_files 不能为空，必须提供图像路径或图像路径列表。")

        output, output_token = self.model.infer_multi(
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
            temperature=temperature,
        )
        logger.info("OCR提取完成，长度 %s 字符，共 %s 个token", len(output), output_token)
        return output


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> tuple[list[str], str]:
    pdf_path = _ensure_pdf_path(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    image_paths = []
    matrix = fitz.Matrix(dpi / 72, dpi / 72)

    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            image_path = Path(tmp_dir) / f"page_{index:04d}.png"
            page.get_pixmap(matrix=matrix).save(image_path)
            image_paths.append(str(image_path))

    return image_paths, tmp_dir


class PDFProcess:
    """统一 PDF 处理入口"""

    def __init__(self):
        self.native_text_parser = NativeTextPDFProcess()
        self.scanned_image_parser = ScannedPDFProcess()
        self.mixed_layout_parser = MixedLayoutPDFProcess()

    def extract_pdf_stats(self, pdf_path: str) -> dict[str, Any]:
        """提取 PDF 文本、图片和版面统计信息。"""
        return PDFTypeDetector(pdf_path).extract_features()

    def detect_pdf_type(self, pdf_path: str) -> dict[str, Any]:
        """判断 PDF 类型: native_text、scanned_image 或 mixed_layout。"""
        return PDFTypeDetector(pdf_path).classify()

    def detect_layout_blocks(
        self,
        pdf_path: str,
        max_pages: int | None = 3,
    ) -> list[dict[str, Any]]:
        """提取 PDF 版面块，用于识别文本块和图片块。"""
        return self.mixed_layout_parser.detect_layout_blocks(pdf_path, max_pages=max_pages)

    def extract_text(
        self,
        pdf_path: str,
        pdf_type: str | None = None,
        use_ocr_for_mixed: bool = False,
    ) -> Optional[str]:
        """根据 PDF 类型自动选择解析器并返回文本。"""
        try:
            pdf_type = pdf_type or self.detect_pdf_type(pdf_path)["doc_type"]

            if pdf_type == PDF_TYPE_NATIVE_TEXT:
                return self.native_text_parser.extract_text(pdf_path)
            if pdf_type == PDF_TYPE_SCANNED_IMAGE:
                return self.scanned_image_parser.extract_text(pdf_path)
            if pdf_type == PDF_TYPE_MIXED_LAYOUT:
                return self.mixed_layout_parser.extract_text(
                    pdf_path,
                    use_ocr=use_ocr_for_mixed,
                )

            raise ValueError(f"未知PDF类型: {pdf_type}")
        except Exception as exc:
            logger.exception("PDF文本提取失败: %s", exc)
            return None

    def parse_pdf(self, pdf_path: str, use_ocr_for_mixed: bool = False) -> dict[str, Any]:
        """完整解析 PDF，返回类型、统计信息和提取文本。"""
        detection = self.detect_pdf_type(pdf_path)
        text = self.extract_text(
            pdf_path,
            pdf_type=detection["doc_type"],
            use_ocr_for_mixed=use_ocr_for_mixed,
        )
        return {
            "pdf_path": str(pdf_path),
            "doc_type": detection["doc_type"],
            "confidence": detection["confidence"],
            "parse_strategy": detection["parse_strategy"],
            "reason": detection["reason"],
            "stats": detection,
            "text": text,
        }


pdf_process = PDFProcess()

PDFProcessTools = [
    pdf_process.extract_pdf_stats,
    pdf_process.detect_pdf_type,
    pdf_process.detect_layout_blocks,
    pdf_process.extract_text,
    pdf_process.parse_pdf,
]

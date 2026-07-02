import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoModel, AutoTokenizer
from config import setting
import logging
import tempfile, fitz
logger = logging.getLogger(__name__)


class PDFProcess:
    def __init__(self):
        logger.info("正在初始化PDF处理器...")
        model_name = setting.OCR_MODEL_ID

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.eval().cuda()

    def pdf_to_images(self, pdf_path, dpi=300):
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

    def infer(self):
        self.model.infer_multi(
            self.tokenizer,
            prompt='<image>Multi page parsing.',
            image_files=self.pdf_to_images('database/Papers/2404.05220v3.pdf', dpi=300),
            output_path='database/output',
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35, ngram_window=1024,
            save_results=True,
        )


pdf_process = PDFProcess()

# pdf_process.infer()

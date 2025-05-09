import os
import re
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
import pytesseract
from PIL import Image
from typing import Optional
from pathlib import Path

class FileProcessor:
    def __init__(self):
        # Configure Tesseract path (update for your system)
        self.tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_ext == '.docx':
            return self._extract_from_docx(file_path)
        elif file_ext in ('.png', '.jpg', '.jpeg'):
            return self._extract_from_image(file_path)
        elif file_ext == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            text = pdf_extract_text(file_path)
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"PDF extraction failed: {str(e)}")

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            raise ValueError(f"DOCX extraction failed: {str(e)}")

    def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image files using OCR"""
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"Image OCR failed: {str(e)}")

    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Text file reading failed: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing excessive whitespace and special characters"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove form feed characters
        text = re.sub(r'\x0c', '', text)
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in {'\n', '\t', '\r'})
        return text.strip()
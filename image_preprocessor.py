"""
Image preprocessing pipeline for document OCR.
Handles deskewing, contrast enhancement, denoising, and sharpening
to improve extraction accuracy from scanned/photographed documents.

Uses OpenCV (Apache 2.0) — fully free for commercial use.
"""
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import io


class ImagePreprocessor:
    """Preprocesses document images to improve OCR accuracy."""

    def __init__(self, deskew: bool = True, enhance_contrast: bool = True,
                 denoise: bool = True, sharpen: bool = True):
        self.do_deskew = deskew
        self.do_enhance_contrast = enhance_contrast
        self.do_denoise = denoise
        self.do_sharpen = sharpen

    def preprocess_pil_image(self, pil_image: Image.Image) -> Image.Image:
        """Preprocess a PIL Image and return an enhanced PIL Image."""
        # Convert PIL -> OpenCV (BGR)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Run the preprocessing pipeline
        processed = self._pipeline(cv_img)

        # Convert back to PIL
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def preprocess_file(self, file_path: str) -> str:
        """Preprocess an image file on disk. Returns path to the processed file."""
        cv_img = cv2.imread(file_path)
        if cv_img is None:
            return file_path  # Can't read — return original

        processed = self._pipeline(cv_img)

        # Write to a temp file with the same extension
        ext = Path(file_path).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            cv2.imwrite(tmp.name, processed)
            return tmp.name

    def preprocess_bytes(self, img_bytes: bytes) -> bytes:
        """Preprocess image bytes and return enhanced image bytes (JPEG)."""
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            return img_bytes  # Can't decode — return original

        processed = self._pipeline(cv_img)

        success, encoded = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return encoded.tobytes() if success else img_bytes

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _pipeline(self, img: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline on a BGR image.
        Each step is wrapped in try/except so a single failure doesn't break the whole pipeline.
        """
        # Ensure image is 3-channel BGR uint8
        if img is None or img.size == 0:
            return img
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if self.do_deskew:
            try:
                img = self._deskew(img)
            except Exception:
                pass  # Skip deskew on failure

        if self.do_denoise:
            try:
                img = self._denoise(img)
            except Exception:
                pass  # Skip denoise on failure

        if self.do_enhance_contrast:
            try:
                img = self._enhance_contrast(img)
            except Exception:
                pass  # Skip contrast on failure

        if self.do_sharpen:
            try:
                img = self._sharpen(img)
            except Exception:
                pass  # Skip sharpen on failure

        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct document skew using minAreaRect on text contours."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize with Otsu's threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find all text-like contours
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            return img  # Not enough content to determine skew

        # Get the minimum area bounding rectangle
        angle = cv2.minAreaRect(coords)[-1]

        # Normalize the angle
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = -(angle - 90)
        else:
            angle = -angle

        # Only correct if skew is significant (> 0.5 degrees) but not too extreme
        if abs(angle) < 0.5 or abs(angle) > 45:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions to avoid cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(img, M, (new_w, new_h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Remove scanner noise while preserving text edges."""
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Color image — use colored denoising
            return cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
        else:
            # Grayscale fallback
            return cv2.fastNlMeansDenoising(img, None, 6, 7, 21)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE — great for faded text on ID cards."""
        # Convert to LAB color space for luminance-based enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        # Merge back
        merged = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Gentle sharpening to make text edges crisper."""
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ])
        return cv2.filter2D(img, -1, kernel)

    # ------------------------------------------------------------------
    # Utility: detect if document is rotated 90/180/270 degrees
    # ------------------------------------------------------------------

    def detect_orientation(self, img: np.ndarray) -> int:
        """
        Detect if the document is rotated by 0, 90, 180, or 270 degrees.
        Returns the angle to rotate back to upright.
        Uses text line detection via Hough transform.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                 minLineLength=100, maxLineGap=10)
        if lines is None:
            return 0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        if not angles:
            return 0

        median_angle = np.median(angles)

        # Classify into 0, 90, 180, 270
        if -15 < median_angle < 15:
            return 0
        elif 75 < median_angle < 105:
            return 270  # Rotate 270 to fix
        elif -105 < median_angle < -75:
            return 90  # Rotate 90 to fix
        elif median_angle > 165 or median_angle < -165:
            return 180
        return 0

    def fix_orientation(self, img: np.ndarray) -> np.ndarray:
        """Detect and fix 90/180/270 degree rotations."""
        angle = self.detect_orientation(img)
        if angle == 0:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        if angle in (90, 270):
            # Swap dimensions for 90/270 rotation
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            return cv2.warpAffine(img, M, (new_w, new_h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        else:
            # 180 degree — simple flip
            return cv2.rotate(img, cv2.ROTATE_180)

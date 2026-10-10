"""
preprocess.py
-------------
STEP 3: DATA CLEANING & PREPROCESSING PIPELINE

This module implements Step 3 of the 10-stage Machine Learning pipeline:
1. Validates and filters corrupted or unreadable images.
2. Validates image channels, color modes, and formats (converts cleanly to 3-channel RGB).
3. Validates caption strings, text bounds, and metadata integrity.
4. Detects and removes exact content/prompt duplicates or corrupted entries.
5. Resizes images to the model's target resolution (64x64) via bicubic anti-aliasing.
6. Normalizes pixel values from uint8 [0, 255] to float32 [-1.0, 1.0] for DDPM training.
7. Builds a deterministic vocabulary and encodes captions into fixed-length token ID sequences.
8. Saves the processed dataset separately in `data/processed/` alongside pre-computed PyTorch tensors.
9. Produces a detailed quantitative preprocessing report (`preprocessing_report.json`).
"""

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image

# Ensure project root is on sys.path for direct script execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.custom_model.config import (
    DEFAULT_PREPROCESSING_CONFIG,
    DEFAULT_PROBLEM_SPEC,
    PROCESSED_DATA_DIR,
    PROCESSED_IMAGES_DIR,
    PROCESSED_METADATA_PATH,
    PROCESSED_TENSORS_DIR,
    PREPROCESSING_REPORT_PATH,
    RAW_DATA_DIR,
    RAW_IMAGES_DIR,
    RAW_METADATA_PATH,
    VOCAB_PATH,
    DataPreprocessingConfig,
)


class TextTokenizer:
    """
    Learned deterministic word-level tokenizer with special tokens (<pad>, <start>, <end>, <unk>).
    """

    def __init__(self, special_tokens: Optional[Dict[str, int]] = None, max_length: int = 32):
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "<unk>": 3,
        }
        self.max_length = max_length
        self.word2id: Dict[str, int] = dict(self.special_tokens)
        self.id2word: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self._is_fitted = False

    @staticmethod
    def clean_text(text: str) -> List[str]:
        """Lowercases text, removes punctuation, and splits into whitespace tokens."""
        text = text.lower().strip()
        # Keep alphanumeric characters and spaces
        text = re.sub(r"[^\w\s-]", "", text)
        tokens = [t for t in text.split() if t]
        return tokens

    def build_vocab(self, texts: List[str]):
        """Builds vocabulary mapping from a list of caption strings."""
        unique_tokens: Set[str] = set()
        for text in texts:
            tokens = self.clean_text(text)
            unique_tokens.update(tokens)

        # Sort for deterministic vocabulary ordering across runs
        next_id = max(self.special_tokens.values()) + 1
        for token in sorted(unique_tokens):
            if token not in self.word2id:
                self.word2id[token] = next_id
                self.id2word[next_id] = token
                next_id += 1

        self._is_fitted = True

    def encode(self, text: str) -> List[int]:
        """Encodes text to a fixed-length list of token IDs with <start>, <end>, and <pad>."""
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted on vocabulary before encoding.")

        tokens = self.clean_text(text)
        token_ids = [self.special_tokens["<start>"]]

        # Truncate if tokens exceed max_length - 2 (leaving room for start & end tokens)
        usable_len = self.max_length - 2
        for tok in tokens[:usable_len]:
            token_ids.append(self.word2id.get(tok, self.special_tokens["<unk>"]))

        token_ids.append(self.special_tokens["<end>"])

        # Pad remaining sequence length with <pad>
        pad_id = self.special_tokens["<pad>"]
        while len(token_ids) < self.max_length:
            token_ids.append(pad_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to a readable text string."""
        words = []
        for tid in token_ids:
            if tid == self.special_tokens["<pad>"]:
                continue
            word = self.id2word.get(tid, "<unk>")
            if word in ("<start>", "<end>"):
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: Path):
        """Saves vocabulary to JSON file."""
        data = {
            "vocab_size": len(self.word2id),
            "max_length": self.max_length,
            "special_tokens": self.special_tokens,
            "word2id": self.word2id,
            "id2word": {str(k): v for k, v in self.id2word.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class DatasetPreprocessor:
    """
    Executes the full Step 3 data cleaning, validation, normalization, and tokenization pipeline.
    """

    def __init__(self, config: Optional[DataPreprocessingConfig] = None):
        self.config = config or DEFAULT_PREPROCESSING_CONFIG
        self.tokenizer = TextTokenizer(
            special_tokens=self.config.special_tokens,
            max_length=self.config.max_text_length,
        )
        self.stats = {
            "raw_samples_total": 0,
            "valid_samples_processed": 0,
            "removed_corrupted_images": 0,
            "removed_invalid_captions": 0,
            "removed_duplicates": 0,
            "removed_channel_issues": 0,
        }

    def validate_image(self, image_path: Path) -> Tuple[bool, Optional[Image.Image], str]:
        """
        Validates image file existence, decodability, header integrity, and channels.
        Returns (is_valid, PIL_image_rgb, reason).
        """
        if not image_path.exists():
            return False, None, "File does not exist"

        if image_path.stat().st_size == 0:
            return False, None, "Zero-byte file"

        ext = image_path.suffix.lower()
        if ext not in self.config.allowed_image_extensions:
            return False, None, f"Unsupported extension: {ext}"

        try:
            # 1. Integrity check with PIL verify
            with Image.open(image_path) as img:
                img.verify()

            # 2. Re-open and fully decode pixel data
            with Image.open(image_path) as img:
                img_loaded = img.copy()
                img_loaded.load()

                # Convert to RGB (handles RGBA, Grayscale, Palette cleanly)
                if img_loaded.mode != "RGB":
                    img_loaded = img_loaded.convert("RGB")

                # Verify non-zero dimensions
                if img_loaded.width <= 0 or img_loaded.height <= 0:
                    return False, None, f"Invalid dimensions: {img_loaded.size}"

                return True, img_loaded, "OK"

        except Exception as exc:
            return False, None, f"Corrupted image error: {exc}"

    def validate_caption(self, caption: Any) -> Tuple[bool, str, str]:
        """
        Validates prompt text structure, length bounds, and validity.
        Returns (is_valid, cleaned_caption, reason).
        """
        if not isinstance(caption, str):
            return False, "", "Caption is not a string"

        cleaned = caption.strip()
        if not cleaned:
            return False, "", "Empty caption"

        if len(cleaned) < self.config.min_caption_chars:
            return False, "", f"Caption too short (< {self.config.min_caption_chars} chars)"

        if len(cleaned) > self.config.max_caption_chars:
            return False, "", f"Caption too long (> {self.config.max_caption_chars} chars)"

        return True, cleaned, "OK"

    @staticmethod
    def compute_content_hash(img: Image.Image, prompt: str) -> str:
        """Computes deterministic MD5 hash of raw image pixel bytes and caption text."""
        hasher = hashlib.md5()
        hasher.update(img.tobytes())
        hasher.update(prompt.encode("utf-8"))
        return hasher.hexdigest()

    def process_image(self, img: Image.Image) -> Tuple[Image.Image, torch.FloatTensor]:
        """
        Resizes image to target resolution (64x64) and normalizes pixel values to [-1.0, 1.0].
        Returns (resized_pil_image, normalized_torch_tensor [3, H, W]).
        """
        target_w = self.config.target_image_width
        target_h = self.config.target_image_height

        # 1. High-quality Bicubic Resampling
        if img.size != (target_w, target_h):
            img_resized = img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
        else:
            img_resized = img

        # 2. Convert to numpy float array [H, W, C] in range [0.0, 255.0]
        np_arr = np.array(img_resized, dtype=np.float32)

        # 3. Normalize to [-1.0, 1.0]: x_norm = (x / 127.5) - 1.0
        normalized = (np_arr / 127.5) - 1.0

        # 4. Rearrange from [H, W, C] to PyTorch standard [C, H, W]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).contiguous().float()

        return img_resized, tensor

    def run(self) -> Dict[str, Any]:
        """Executes the complete data cleaning and preprocessing pipeline."""
        start_time = time.time()
        print("=" * 70)
        print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
        print("  STEP 3: DATA CLEANING & PREPROCESSING PIPELINE")
        print("=" * 70)

        # Ensure output directories exist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_TENSORS_DIR.mkdir(parents=True, exist_ok=True)

        # Check raw dataset availability
        if not RAW_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Raw dataset metadata not found at '{RAW_METADATA_PATH}'. "
                "Ensure Step 2: Data Collection has been run first."
            )

        print(f"\n[1/6] Reading raw dataset metadata from: {RAW_METADATA_PATH}")
        raw_entries: List[Dict[str, Any]] = []
        with open(RAW_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_entries.append(json.loads(line))

        self.stats["raw_samples_total"] = len(raw_entries)
        print(f"      Total raw samples found: {len(raw_entries)}")

        # Step 1: Validation, Deduplication, and Filtering
        print("\n[2/6] Validating images, captions, and removing duplicates...")
        valid_records: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()

        for record in raw_entries:
            img_rel_path = record.get("image_file")
            caption_raw = record.get("prompt")
            sample_id = record.get("id")

            img_path = RAW_IMAGES_DIR / img_rel_path if img_rel_path else None
            if not img_path:
                self.stats["removed_corrupted_images"] += 1
                continue

            # Validate Image
            img_valid, img_obj, img_err = self.validate_image(img_path)
            if not img_valid or img_obj is None:
                self.stats["removed_corrupted_images"] += 1
                continue

            # Validate Caption
            cap_valid, clean_caption, cap_err = self.validate_caption(caption_raw)
            if not cap_valid:
                self.stats["removed_invalid_captions"] += 1
                continue

            # Deduplication
            if self.config.deduplicate_by_content:
                content_hash = self.compute_content_hash(img_obj, clean_caption)
                if content_hash in seen_hashes:
                    self.stats["removed_duplicates"] += 1
                    continue
                seen_hashes.add(content_hash)

            valid_records.append({
                "id": sample_id,
                "raw_record": record,
                "image_obj": img_obj,
                "caption": clean_caption,
            })

        print(f"      Valid candidates passed checks: {len(valid_records)}")
        print(f"      Removed corrupted images: {self.stats['removed_corrupted_images']}")
        print(f"      Removed invalid captions: {self.stats['removed_invalid_captions']}")
        print(f"      Removed duplicates: {self.stats['removed_duplicates']}")

        # Step 2: Vocabulary Building
        print("\n[3/6] Building vocabulary from verified captions...")
        all_captions = [r["caption"] for r in valid_records]
        self.tokenizer.build_vocab(all_captions)
        vocab_size = len(self.tokenizer.word2id)
        print(f"      Unique vocabulary size: {vocab_size} tokens")
        print(f"      Special tokens: {self.config.special_tokens}")

        # Save vocab.json
        self.tokenizer.save(VOCAB_PATH)
        print(f"      Vocabulary saved to: {VOCAB_PATH}")

        # Step 3: Image Resizing, Normalization, Tokenization & Serialization
        print("\n[4/6] Resizing images to (64, 64), normalizing to [-1, 1], and encoding tokens...")
        processed_metadata_lines: List[str] = []
        tensor_min_vals: List[float] = []
        tensor_max_vals: List[float] = []
        tensor_means: List[float] = []
        tensor_stds: List[float] = []

        for idx, rec in enumerate(valid_records):
            img_obj: Image.Image = rec["image_obj"]
            caption: str = rec["caption"]
            sample_id: int = rec["id"]

            # Resize & Normalize
            resized_img, img_tensor = self.process_image(img_obj)

            # Tokenize caption
            token_ids = self.tokenizer.encode(caption)

            # Track numerical statistics
            tensor_min_vals.append(float(img_tensor.min()))
            tensor_max_vals.append(float(img_tensor.max()))
            tensor_means.append(float(img_tensor.mean()))
            tensor_stds.append(float(img_tensor.std()))

            # File naming
            img_filename = f"processed_{sample_id:05d}.png"
            tensor_filename = f"tensor_{sample_id:05d}.pt"

            out_img_path = PROCESSED_IMAGES_DIR / img_filename
            out_tensor_path = PROCESSED_TENSORS_DIR / tensor_filename

            # Save clean PNG image
            resized_img.save(out_img_path, format="PNG")

            # Save precomputed PyTorch tensor dictionary
            torch.save({
                "id": sample_id,
                "image": img_tensor,                       # Shape: [3, 64, 64], dtype: float32, range: [-1, 1]
                "token_ids": torch.tensor(token_ids, dtype=torch.long),  # Shape: [32], dtype: int64
                "prompt": caption,
            }, out_tensor_path)

            # Metadata entry
            processed_entry = {
                "id": sample_id,
                "image_file": img_filename,
                "tensor_file": tensor_filename,
                "prompt": caption,
                "token_ids": token_ids,
                "token_count": len([t for t in token_ids if t != self.config.special_tokens["<pad>"]]),
                "shape": rec["raw_record"].get("shape"),
                "color": rec["raw_record"].get("color"),
                "background": rec["raw_record"].get("background"),
                "width": self.config.target_image_width,
                "height": self.config.target_image_height,
                "channels": self.config.target_image_channels,
                "pixel_range": list(self.config.pixel_range),
            }
            processed_metadata_lines.append(json.dumps(processed_entry))

        self.stats["valid_samples_processed"] = len(valid_records)

        # Step 4: Write processed metadata
        print(f"\n[5/6] Writing processed metadata to: {PROCESSED_METADATA_PATH}")
        with open(PROCESSED_METADATA_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(processed_metadata_lines) + "\n")

        # Step 5: Compute dataset-wide statistics & write processing report
        elapsed = round(time.time() - start_time, 2)
        print("\n[6/6] Compiling quantitative preprocessing report...")

        report = {
            "step": "Step 3: Data Cleaning & Preprocessing",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "execution_seconds": elapsed,
            "counts": {
                "raw_samples_total": self.stats["raw_samples_total"],
                "valid_samples_processed": self.stats["valid_samples_processed"],
                "removed_corrupted_images": self.stats["removed_corrupted_images"],
                "removed_invalid_captions": self.stats["removed_invalid_captions"],
                "removed_duplicates": self.stats["removed_duplicates"],
                "rejection_rate_percent": round(
                    ((self.stats["raw_samples_total"] - self.stats["valid_samples_processed"]) / max(1, self.stats["raw_samples_total"])) * 100, 2
                ),
            },
            "tensor_specifications": {
                "image_shape": [self.config.target_image_channels, self.config.target_image_height, self.config.target_image_width],
                "image_dtype": "torch.float32",
                "pixel_min": round(float(np.min(tensor_min_vals)), 4),
                "pixel_max": round(float(np.max(tensor_max_vals)), 4),
                "pixel_mean": round(float(np.mean(tensor_means)), 4),
                "pixel_std": round(float(np.mean(tensor_stds)), 4),
                "text_sequence_length": self.config.max_text_length,
                "text_dtype": "torch.int64",
                "vocabulary_size": vocab_size,
            },
            "vocabulary_preview": {
                "special_tokens": self.config.special_tokens,
                "sample_words": list(self.tokenizer.word2id.keys())[:25],
            },
            "storage_locations": {
                "raw_dir": str(RAW_DATA_DIR),
                "processed_dir": str(PROCESSED_DATA_DIR),
                "processed_images_dir": str(PROCESSED_IMAGES_DIR),
                "processed_tensors_dir": str(PROCESSED_TENSORS_DIR),
                "processed_metadata": str(PROCESSED_METADATA_PATH),
                "vocab_file": str(VOCAB_PATH),
                "report_file": str(PREPROCESSING_REPORT_PATH),
            },
        }

        with open(PREPROCESSING_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"      Report saved to: {PREPROCESSING_REPORT_PATH}")
        print("=" * 70)
        print("  STEP 3 SUMMARY:")
        print(f"  • Raw Samples Analyzed:       {self.stats['raw_samples_total']}")
        print(f"  • Valid Processed Samples:    {self.stats['valid_samples_processed']}")
        print(f"  • Corrupted/Removed Samples:  {self.stats['raw_samples_total'] - self.stats['valid_samples_processed']}")
        print(f"  • Vocabulary Size:            {vocab_size} tokens")
        print(f"  • Image Tensor Shape:         [{self.config.target_image_channels}, {self.config.target_image_height}, {self.config.target_image_width}]")
        print(f"  • Pixel Value Range:          [{report['tensor_specifications']['pixel_min']}, {report['tensor_specifications']['pixel_max']}]")
        print(f"  • Text Sequence Length:       {self.config.max_text_length} tokens (with <start>/<end>/<pad>)")
        print(f"  • Elapsed Time:               {elapsed}s")
        print("=" * 70)

        return report


def main():
    preprocessor = DatasetPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()

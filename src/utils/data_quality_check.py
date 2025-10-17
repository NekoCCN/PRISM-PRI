"""
Dataset Quality Checker

Comprehensive tool for checking dataset integrity, label quality,
class distribution, and potential issues before training.
"""
import yaml
import json
import hashlib
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Comprehensive dataset quality checker.

    Checks:
    - Image integrity
    - Label completeness
    - Class distribution
    - Bounding box quality
    - Image size distribution
    - Duplicate detection
    """

    def __init__(self, data_yaml: str):
        """
        Initialize quality checker.

        Args:
            data_yaml: Path to dataset configuration YAML file
        """
        with open(data_yaml) as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(data_yaml).parent

        # Store issues as dictionary with different types
        self.issues = {
            'corrupted_images': [],
            'missing_labels': [],
            'empty_labels': [],
            'invalid_labels': [],
            'invalid_coords': [],
            'too_small_boxes': [],
            'too_large_boxes': [],
            'duplicates': [],
            'class_imbalance': None  # Will be dict if detected
        }

        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'total_boxes': 0,
            'class_counts': {},
            'image_sizes': []
        }

    def run_checks(self) -> dict:
        """
        Run all quality checks.

        Returns:
            Dictionary containing issues and statistics
        """
        logger.info("=" * 80)
        logger.info("Starting Dataset Quality Check")
        logger.info("=" * 80)

        checks = [
            ("Image Integrity", self.check_images),
            ("Label Completeness", self.check_labels),
            ("Class Distribution", self.check_class_distribution),
            ("Bounding Box Quality", self.check_bbox_quality),
            ("Image Sizes", self.check_image_sizes),
            ("Duplicate Detection", self.check_duplicates),
        ]

        for check_name, check_func in checks:
            logger.info("")
            logger.info(f"[{check_name}]")
            try:
                check_func()
            except Exception as e:
                logger.error(f"Check failed: {e}")

        # Generate report
        self.generate_report()

        return {
            'issues': self.issues,
            'stats': self.stats
        }

    def check_images(self):
        """Check image file integrity."""
        train_dir = self.data_dir / self.config['train']
        img_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

        self.stats['total_images'] = len(img_files)
        logger.info(f"Found {len(img_files)} images")

        corrupted = []
        for img_path in tqdm(img_files, desc="Checking images", ncols=100):
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception as e:
                corrupted.append(str(img_path))
                self.issues['corrupted_images'].append({
                    'file': str(img_path),
                    'error': str(e)
                })

        if corrupted:
            logger.warning(f"Found {len(corrupted)} corrupted images")
        else:
            logger.info("All images are intact")

    def check_labels(self):
        """Check label file completeness and format."""
        train_dir = self.data_dir / self.config['train']
        label_dir = self.data_dir / self.config['train'].replace('images', 'labels')

        img_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

        missing_labels = []
        empty_labels = []
        invalid_labels = []
        total_boxes = 0

        for img_path in tqdm(img_files, desc="Checking labels", ncols=100):
            label_path = label_dir / f"{img_path.stem}.txt"

            # Check if label file exists
            if not label_path.exists():
                missing_labels.append(str(img_path))
                self.issues['missing_labels'].append(str(img_path))
                continue

            # Check if label is empty
            with open(label_path) as f:
                lines = f.readlines()

            if len(lines) == 0:
                empty_labels.append(str(img_path))
                self.issues['empty_labels'].append(str(img_path))
                continue

            # Check label format (YOLO: class cx cy w h)
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_labels.append(str(img_path))
                    self.issues['invalid_labels'].append({
                        'file': str(img_path),
                        'line_number': line_num,
                        'content': line.strip()
                    })
                    break

                # Validate values are numeric
                try:
                    cls, cx, cy, w, h = map(float, parts)
                    total_boxes += 1
                except ValueError:
                    invalid_labels.append(str(img_path))
                    self.issues['invalid_labels'].append({
                        'file': str(img_path),
                        'line_number': line_num,
                        'content': line.strip(),
                        'error': 'Non-numeric values'
                    })
                    break

        self.stats['total_labels'] = len(img_files) - len(missing_labels)
        self.stats['total_boxes'] = total_boxes

        logger.info(f"Missing labels: {len(missing_labels)}")
        logger.info(f"Empty labels: {len(empty_labels)}")
        logger.info(f"Invalid labels: {len(invalid_labels)}")
        logger.info(f"Total bounding boxes: {total_boxes}")

    def check_class_distribution(self):
        """Check class distribution and balance."""
        label_dir = self.data_dir / self.config['train'].replace('images', 'labels')
        label_files = list(label_dir.glob('*.txt'))

        class_counts = defaultdict(int)

        for label_path in label_files:
            try:
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls = int(float(parts[0]))
                            class_counts[cls] += 1
            except Exception as e:
                logger.warning(f"Failed to read {label_path}: {e}")

        self.stats['class_counts'] = dict(class_counts)

        if not class_counts:
            logger.warning("No valid class labels found")
            return

        # Visualize distribution
        try:
            plt.figure(figsize=(10, 6))
            classes = sorted(class_counts.keys())
            counts = [class_counts[c] for c in classes]

            # Get class names
            names = []
            for c in classes:
                if c < len(self.config['names']):
                    names.append(self.config['names'][c])
                else:
                    names.append(f"Class_{c}")

            plt.bar(names, counts)
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Class Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('class_distribution.png', dpi=150)
            plt.close()

            logger.info("Class distribution plot saved to: class_distribution.png")
        except Exception as e:
            logger.warning(f"Failed to generate class distribution plot: {e}")

        # Check for class imbalance
        if len(counts) > 1:
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count

            logger.info(f"Class imbalance ratio: {imbalance_ratio:.1f}:1")

            if imbalance_ratio > 10:
                logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
                self.issues['class_imbalance'] = {
                    'ratio': float(imbalance_ratio),
                    'max_count': int(max_count),
                    'min_count': int(min_count),
                    'counts': dict(class_counts)
                }

    def check_bbox_quality(self):
        """Check bounding box quality and validity."""
        label_dir = self.data_dir / self.config['train'].replace('images', 'labels')
        label_files = list(label_dir.glob('*.txt'))

        too_small = []
        too_large = []
        invalid_coords = []

        for label_path in label_files:
            try:
                with open(label_path) as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        try:
                            cls, cx, cy, w, h = map(float, parts)
                        except ValueError:
                            continue

                        # Check coordinate ranges (YOLO format: normalized 0-1)
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            invalid_coords.append(str(label_path))
                            self.issues['invalid_coords'].append({
                                'file': str(label_path),
                                'line_number': line_num,
                                'bbox': [float(cx), float(cy), float(w), float(h)]
                            })

                        # Check box size
                        area = w * h
                        if area < 0.001:  # Too small (0.1% of image)
                            too_small.append(str(label_path))
                            self.issues['too_small_boxes'].append({
                                'file': str(label_path),
                                'line_number': line_num,
                                'area': float(area)
                            })
                        elif area > 0.9:  # Too large (90% of image)
                            too_large.append(str(label_path))
                            self.issues['too_large_boxes'].append({
                                'file': str(label_path),
                                'line_number': line_num,
                                'area': float(area)
                            })
            except Exception as e:
                logger.warning(f"Failed to check {label_path}: {e}")

        logger.info(f"Boxes too small: {len(set(too_small))}")
        logger.info(f"Boxes too large: {len(set(too_large))}")
        logger.info(f"Invalid coordinates: {len(set(invalid_coords))}")

    def check_image_sizes(self):
        """Check image size distribution."""
        train_dir = self.data_dir / self.config['train']
        img_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

        sizes = []
        for img_path in tqdm(img_files, desc="Checking sizes", ncols=100):
            try:
                img = Image.open(img_path)
                sizes.append(img.size)
            except Exception as e:
                logger.warning(f"Failed to read {img_path}: {e}")

        if not sizes:
            logger.warning("No valid images found")
            return

        widths, heights = zip(*sizes)
        self.stats['image_sizes'] = {
            'width_range': [int(min(widths)), int(max(widths))],
            'height_range': [int(min(heights)), int(max(heights))],
            'avg_width': float(np.mean(widths)),
            'avg_height': float(np.mean(heights))
        }

        logger.info(f"Width range: {min(widths)} - {max(widths)}")
        logger.info(f"Height range: {min(heights)} - {max(heights)}")
        logger.info(f"Average size: {np.mean(widths):.0f} x {np.mean(heights):.0f}")

    def check_duplicates(self):
        """Detect duplicate images using MD5 hash."""
        train_dir = self.data_dir / self.config['train']
        img_files = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))

        hashes = {}
        duplicates = []

        for img_path in tqdm(img_files, desc="Computing hashes", ncols=100):
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in hashes:
                    duplicates.append((str(img_path), hashes[file_hash]))
                    self.issues['duplicates'].append({
                        'file1': str(img_path),
                        'file2': hashes[file_hash]
                    })
                else:
                    hashes[file_hash] = str(img_path)
            except Exception as e:
                logger.warning(f"Failed to hash {img_path}: {e}")

        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate image pairs")
        else:
            logger.info("No duplicate images found")

    def generate_report(self):
        """Generate and save quality check report."""
        report = {
            'summary': self._generate_summary(),
            'statistics': self.stats,
            'issues': {k: v for k, v in self.issues.items() if v}  # Only non-empty
        }

        report_path = 'data_quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("")
        logger.info("=" * 80)
        logger.info("Quality Check Summary")
        logger.info("=" * 80)

        summary = report['summary']
        logger.info(f"Total issues found: {summary['total_issues']}")

        if summary['total_issues'] == 0:
            logger.info("Dataset quality: EXCELLENT")
        elif summary['total_issues'] < 10:
            logger.info("Dataset quality: GOOD")
        elif summary['total_issues'] < 50:
            logger.info("Dataset quality: FAIR - Review issues")
        else:
            logger.info("Dataset quality: POOR - Fix critical issues before training")

        logger.info("")
        logger.info(f"Report saved to: {report_path}")

    def _generate_summary(self) -> dict:
        """Generate summary statistics."""
        total_issues = sum(
            len(v) if isinstance(v, list) else (1 if v else 0)
            for v in self.issues.values()
        )

        critical_issues = (
            len(self.issues['corrupted_images']) +
            len(self.issues['missing_labels']) +
            len(self.issues['invalid_labels'])
        )

        return {
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'corrupted_images': len(self.issues['corrupted_images']),
            'missing_labels': len(self.issues['missing_labels']),
            'empty_labels': len(self.issues['empty_labels']),
            'invalid_labels': len(self.issues['invalid_labels']),
            'duplicate_images': len(self.issues['duplicates']),
            'has_class_imbalance': self.issues['class_imbalance'] is not None
        }


def check_dataset_quality(data_yaml: str, output_dir: str = '.') -> dict:
    """
    Convenience function to check dataset quality.

    Args:
        data_yaml: Path to data.yaml
        output_dir: Directory to save outputs

    Returns:
        Dictionary with issues and statistics
    """
    checker = DataQualityChecker(data_yaml)
    results = checker.run_checks()
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check dataset quality')
    parser.add_argument('--data-yaml', type=str, default='dataset/data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory')

    args = parser.parse_args()

    check_dataset_quality(args.data_yaml, args.output_dir)
#!/usr/bin/env python3
"""
Dataset Download and Preparation Script for Stroke Risk Prediction
================================================================

This script downloads and prepares real stroke datasets for training the ML model.
It supports multiple data sources and handles data preprocessing automatically.

Supported Datasets:
1. Kaggle Stroke Prediction Dataset (Primary)
2. Healthcare Dataset Stroke Data (Alternative)
3. Custom CSV datasets

Usage:
    python download_datasets.py --dataset kaggle
    python download_datasets.py --dataset healthcare
    python download_datasets.py --dataset all
    python download_datasets.py --help

Author: Brain Stroke Risk Prediction Team
Version: 1.0.0
"""

import os
import sys
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
import zipfile
import urllib.request
from datetime import datetime
import hashlib

class DatasetDownloader:
    """Download and prepare stroke datasets"""

    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Dataset configurations
        self.datasets = {
            'kaggle': {
                'name': 'Kaggle Stroke Prediction Dataset',
                'url': 'https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset',
                'filename': 'healthcare-dataset-stroke-data.csv',
                'description': 'Primary stroke dataset with ~5K records',
                'direct_download': 'https://github.com/your-backup-repo/stroke-data/raw/main/stroke-data.csv',
                'expected_columns': [
                    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                    'smoking_status', 'stroke'
                ]
            },
            'healthcare': {
                'name': 'Healthcare Dataset Stroke Data',
                'url': 'https://www.kaggle.com/datasets/asaumya/healthcare-dataset-stroke-data',
                'filename': 'healthcare-dataset-stroke-data-alt.csv',
                'description': 'Alternative stroke dataset with additional features',
                'direct_download': 'https://github.com/your-backup-repo/stroke-data/raw/main/stroke-data-alt.csv',
                'expected_columns': [
                    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'residence_type', 'avg_glucose_level', 'bmi',
                    'smoking_status', 'stroke'
                ]
            }
        }

        # Sample data URLs (backup if main sources fail)
        self.sample_data_urls = [
            'https://raw.githubusercontent.com/dataset-samples/medical-data/main/stroke-sample.csv',
            'https://storage.googleapis.com/public-datasets/stroke-prediction/sample-data.csv'
        ]

    def print_colored(self, message, color='green'):
        """Print colored output"""
        colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'blue': '\033[94m',
            'end': '\033[0m'
        }
        print(f"{colors.get(color, '')}{message}{colors['end']}")

    def download_from_url(self, url, filename, description=""):
        """Download file from URL"""
        filepath = self.data_dir / filename

        if filepath.exists():
            self.print_colored(f"✅ {filename} already exists", 'green')
            return filepath

        self.print_colored(f"📥 Downloading {description}...", 'blue')

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)

            print()  # New line after progress
            self.print_colored(f"✅ Downloaded {filename}", 'green')
            return filepath

        except Exception as e:
            self.print_colored(f"❌ Failed to download {filename}: {e}", 'red')
            if filepath.exists():
                filepath.unlink()  # Remove partial file
            return None

    def create_sample_dataset(self, filename="sample-stroke-data.csv", n_samples=5000):
        """Create a realistic sample dataset"""
        self.print_colored(f"🔬 Creating sample dataset with {n_samples} records...", 'blue')

        np.random.seed(42)

        # Generate realistic data
        data = {}

        # Demographics
        data['id'] = range(1, n_samples + 1)
        data['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])

        # Age distribution skewed towards older ages
        ages = np.random.gamma(2, 20) + 18
        data['age'] = np.clip(ages, 18, 95).astype(int)

        # Marital status based on age
        married_prob = np.where(data['age'] > 25, 0.7, 0.2)
        data['ever_married'] = np.where(
            np.random.binomial(1, married_prob), 'Yes', 'No'
        )

        # Work type based on age
        work_types = []
        for age in data['age']:
            if age < 18:
                work_types.append('children')
            elif age < 65:
                work_types.append(np.random.choice(
                    ['Private', 'Self-employed', 'Govt_job'], p=[0.6, 0.2, 0.2]
                ))
            else:
                work_types.append(np.random.choice(
                    ['Private', 'Self-employed', 'Never_worked'], p=[0.4, 0.2, 0.4]
                ))
        data['work_type'] = work_types

        # Residence type
        data['Residence_type'] = np.random.choice(['Urban', 'Rural'], n_samples, p=[0.65, 0.35])

        # Medical conditions (age-dependent)
        hypertension_prob = np.clip((np.array(data['age']) - 30) / 50, 0.05, 0.5)
        data['hypertension'] = np.random.binomial(1, hypertension_prob)

        heart_disease_prob = np.clip((np.array(data['age']) - 40) / 60, 0.02, 0.3)
        data['heart_disease'] = np.random.binomial(1, heart_disease_prob)

        # BMI (realistic distribution)
        bmi_mean = 25 + (np.array(data['age']) - 40) * 0.1
        data['bmi'] = np.random.normal(bmi_mean, 4)
        data['bmi'] = np.clip(data['bmi'], 15, 50)

        # Glucose level (influenced by age, BMI, conditions)
        glucose_base = 85 + (np.array(data['age']) - 30) * 0.3
        glucose_bmi_effect = np.where(data['bmi'] > 30, 15, 0)
        glucose_hypertension_effect = np.array(data['hypertension']) * 25
        glucose_heart_effect = np.array(data['heart_disease']) * 20

        data['avg_glucose_level'] = (
            glucose_base + glucose_bmi_effect +
            glucose_hypertension_effect + glucose_heart_effect +
            np.random.normal(0, 20, n_samples)
        )
        data['avg_glucose_level'] = np.clip(data['avg_glucose_level'], 60, 300)

        # Smoking status
        smoking_dist = np.random.multinomial(1, [0.5, 0.2, 0.25, 0.05], n_samples)
        smoking_options = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
        data['smoking_status'] = [
            smoking_options[np.argmax(row)] for row in smoking_dist
        ]

        # Calculate stroke risk based on medical literature
        stroke_risk_score = (
            (np.array(data['age']) > 65).astype(float) * 0.3 +
            np.array(data['hypertension']) * 0.25 +
            np.array(data['heart_disease']) * 0.2 +
            (data['avg_glucose_level'] > 140).astype(float) * 0.15 +
            (data['bmi'] > 30).astype(float) * 0.1 +
            (np.array([s == 'smokes' for s in data['smoking_status']])).astype(float) * 0.15
        )

        # Convert to binary with realistic stroke rate (~5%)
        stroke_threshold = np.percentile(stroke_risk_score, 95)
        data['stroke'] = (stroke_risk_score > stroke_threshold).astype(int)

        # Add some noise to make it more realistic
        flip_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        for idx in flip_indices:
            data['stroke'][idx] = 1 - data['stroke'][idx]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to file
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)

        # Print statistics
        stroke_count = df['stroke'].sum()
        stroke_rate = df['stroke'].mean()

        self.print_colored(f"✅ Sample dataset created: {filename}", 'green')
        self.print_colored(f"   📊 Total records: {len(df):,}", 'blue')
        self.print_colored(f"   🎯 Stroke cases: {stroke_count:,} ({stroke_rate:.1%})", 'blue')

        return filepath

    def download_kaggle_instructions(self):
        """Provide instructions for Kaggle dataset download"""
        instructions = """
        📋 KAGGLE DATASET DOWNLOAD INSTRUCTIONS:

        Option 1 - Manual Download (Recommended):
        ========================================
        1. Go to: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
        2. Click "Download" button (requires Kaggle account)
        3. Extract the CSV file to the data/ directory
        4. Rename it to 'kaggle-stroke-data.csv'

        Option 2 - Kaggle API (Advanced):
        ================================
        1. Install Kaggle API: pip install kaggle
        2. Set up credentials: ~/.kaggle/kaggle.json
        3. Run: kaggle datasets download -d fedesoriano/stroke-prediction-dataset
        4. Extract and move to data/ directory

        Option 3 - Use Sample Data:
        ==========================
        Run this script with --create-sample to generate realistic synthetic data
        """

        self.print_colored(instructions, 'yellow')

    def validate_dataset(self, filepath, expected_columns=None):
        """Validate downloaded dataset"""
        self.print_colored(f"🔍 Validating {filepath.name}...", 'blue')

        try:
            df = pd.read_csv(filepath)

            # Basic validation
            if len(df) == 0:
                self.print_colored(f"❌ Dataset is empty", 'red')
                return False

            # Check required columns
            if expected_columns:
                missing_cols = [col for col in expected_columns if col not in df.columns]
                if missing_cols:
                    self.print_colored(f"⚠️  Missing columns: {missing_cols}", 'yellow')
                    self.print_colored(f"   Available columns: {list(df.columns)}", 'yellow')

            # Check for stroke column
            if 'stroke' not in df.columns:
                self.print_colored(f"❌ Missing target column 'stroke'", 'red')
                return False

            # Basic statistics
            stroke_rate = df['stroke'].mean() if 'stroke' in df.columns else 0
            missing_count = df.isnull().sum().sum()

            self.print_colored(f"✅ Dataset validation passed:", 'green')
            self.print_colored(f"   📊 Records: {len(df):,}", 'blue')
            self.print_colored(f"   📋 Columns: {len(df.columns)}", 'blue')
            self.print_colored(f"   🎯 Stroke rate: {stroke_rate:.1%}", 'blue')
            self.print_colored(f"   ❓ Missing values: {missing_count:,}", 'blue')

            # Save metadata
            metadata = {
                'filename': filepath.name,
                'records': len(df),
                'columns': list(df.columns),
                'stroke_rate': float(stroke_rate),
                'missing_values': int(missing_count),
                'validation_date': datetime.now().isoformat(),
                'file_size_mb': filepath.stat().st_size / (1024*1024)
            }

            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            self.print_colored(f"❌ Validation failed: {e}", 'red')
            return False

    def download_dataset(self, dataset_name):
        """Download specific dataset"""
        if dataset_name not in self.datasets:
            self.print_colored(f"❌ Unknown dataset: {dataset_name}", 'red')
            return False

        config = self.datasets[dataset_name]
        self.print_colored(f"📥 Downloading {config['name']}...", 'blue')

        # Try direct download first
        if 'direct_download' in config:
            filepath = self.download_from_url(
                config['direct_download'],
                f"{dataset_name}-{config['filename']}",
                config['name']
            )

            if filepath and self.validate_dataset(filepath, config['expected_columns']):
                return filepath

        # If direct download fails, provide manual instructions
        self.print_colored(f"⚠️  Direct download not available for {config['name']}", 'yellow')
        self.print_colored(f"   Manual download required from: {config['url']}", 'yellow')

        if dataset_name == 'kaggle':
            self.download_kaggle_instructions()

        return None

    def download_all_datasets(self):
        """Download all available datasets"""
        self.print_colored("📦 Downloading all available datasets...", 'blue')

        downloaded = []
        for dataset_name in self.datasets:
            filepath = self.download_dataset(dataset_name)
            if filepath:
                downloaded.append(filepath)

        # Create sample dataset as fallback
        sample_path = self.create_sample_dataset()
        downloaded.append(sample_path)

        return downloaded

    def list_available_datasets(self):
        """List all available datasets"""
        self.print_colored("📋 Available Datasets:", 'blue')
        print()

        for name, config in self.datasets.items():
            self.print_colored(f"🔹 {name.upper()}:", 'green')
            print(f"   Name: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   URL: {config['url']}")
            print(f"   Expected file: {config['filename']}")
            print()

    def check_existing_data(self):
        """Check for existing data files"""
        self.print_colored("🔍 Checking existing data files...", 'blue')

        existing_files = list(self.data_dir.glob("*.csv"))

        if not existing_files:
            self.print_colored("❌ No CSV files found in data directory", 'red')
            return []

        self.print_colored(f"✅ Found {len(existing_files)} CSV files:", 'green')

        validated_files = []
        for filepath in existing_files:
            print(f"   📄 {filepath.name}")
            if self.validate_dataset(filepath):
                validated_files.append(filepath)

        return validated_files

    def prepare_data_summary(self):
        """Create a summary of all available data"""
        csv_files = list(self.data_dir.glob("*.csv"))

        if not csv_files:
            self.print_colored("❌ No data files found", 'red')
            return

        summary = {
            'datasets': [],
            'total_records': 0,
            'preparation_date': datetime.now().isoformat()
        }

        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)

                dataset_info = {
                    'filename': filepath.name,
                    'records': len(df),
                    'columns': len(df.columns),
                    'stroke_cases': int(df['stroke'].sum()) if 'stroke' in df.columns else 0,
                    'stroke_rate': float(df['stroke'].mean()) if 'stroke' in df.columns else 0,
                    'missing_values': int(df.isnull().sum().sum()),
                    'file_size_mb': round(filepath.stat().st_size / (1024*1024), 2)
                }

                summary['datasets'].append(dataset_info)
                summary['total_records'] += dataset_info['records']

            except Exception as e:
                self.print_colored(f"⚠️  Error reading {filepath.name}: {e}", 'yellow')

        # Save summary
        summary_path = self.data_dir / 'data_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.print_colored(f"📊 Data Summary Created: {summary_path}", 'green')
        self.print_colored(f"   Total datasets: {len(summary['datasets'])}", 'blue')
        self.print_colored(f"   Total records: {summary['total_records']:,}", 'blue')

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare stroke prediction datasets"
    )

    parser.add_argument(
        '--dataset',
        choices=['kaggle', 'healthcare', 'all', 'sample'],
        default='sample',
        help='Dataset to download (default: sample)'
    )

    parser.add_argument(
        '--data-dir',
        default='./data',
        help='Directory to store datasets (default: ./data)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Check existing data files'
    )

    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample synthetic dataset'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=5000,
        help='Number of samples for synthetic dataset (default: 5000)'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = DatasetDownloader(args.data_dir)

    print("=" * 70)
    print("🧠 STROKE PREDICTION DATASET DOWNLOADER")
    print("=" * 70)

    try:
        if args.list:
            downloader.list_available_datasets()

        elif args.check:
            downloader.check_existing_data()

        elif args.create_sample or args.dataset == 'sample':
            downloader.create_sample_dataset(n_samples=args.samples)

        elif args.dataset == 'all':
            downloader.download_all_datasets()

        else:
            downloader.download_dataset(args.dataset)

        # Always create summary at the end
        downloader.prepare_data_summary()

        print("\n" + "=" * 70)
        downloader.print_colored("✅ Dataset preparation completed!", 'green')
        downloader.print_colored(f"📁 Data directory: {downloader.data_dir.absolute()}", 'blue')
        print("=" * 70)

    except KeyboardInterrupt:
        downloader.print_colored("\n❌ Download interrupted by user", 'red')
        sys.exit(1)

    except Exception as e:
        downloader.print_colored(f"\n❌ Error: {e}", 'red')
        sys.exit(1)

if __name__ == "__main__":
    main()

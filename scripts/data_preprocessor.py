"""
Data Preprocessing Module for Breast Cancer Chatbot
===================================================

This module handles all data preprocessing tasks including:
- Data cleaning and normalization
- Text preprocessing
- Dataset expansion using patterns
- Data validation and quality checks
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
import warnings

from config import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing class for breast cancer Q&A data
    """

    def __init__(self):
        """Initialize the preprocessor with required resources"""
        self.config = config.data
        self._setup_nltk()

    def _setup_nltk(self):
        """Download and setup required NLTK resources"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK resources loaded successfully")
        except Exception as e:
            logger.warning(f"Could not setup NLTK resources: {e}")
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data

        Args:
            text: Raw text to clean

        Returns:
            Cleaned and normalized text
        """
        if pd.isna(text) or text == "":
            return ""

        # Convert to string and strip whitespace
        text = str(text).strip()

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)

        # Keep only alphanumeric, basic punctuation, and common symbols
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\(\)\'\"]', ' ', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*', r'\1 ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def normalize_medical_terms(self, text: str) -> str:
        """
        Normalize common medical terms and abbreviations

        Args:
            text: Text containing medical terms

        Returns:
            Text with normalized medical terms
        """
        # Medical term mappings
        medical_mappings = {
            r'\bca\b': 'cancer',
            r'\btx\b': 'treatment',
            r'\bdx\b': 'diagnosis',
            r'\bpt\b': 'patient',
            r'\bher2\+': 'her2 positive',
            r'\bher2\-': 'her2 negative',
            r'\ber\+': 'estrogen receptor positive',
            r'\ber\-': 'estrogen receptor negative',
            r'\bpr\+': 'progesterone receptor positive',
            r'\bpr\-': 'progesterone receptor negative',
            r'\btn\b': 'triple negative',
            r'\bidc\b': 'invasive ductal carcinoma',
            r'\bilc\b': 'invasive lobular carcinoma',
            r'\bdcis\b': 'ductal carcinoma in situ',
            r'\blcis\b': 'lobular carcinoma in situ',
        }

        text_lower = text.lower()
        for pattern, replacement in medical_mappings.items():
            text_lower = re.sub(pattern, replacement, text_lower)

        return text_lower

    def validate_qa_pair(self, question: str, answer: str) -> bool:
        """
        Validate a question-answer pair for quality

        Args:
            question: Question text
            answer: Answer text

        Returns:
            True if pair is valid, False otherwise
        """
        # Check minimum lengths
        if len(question) < self.config.min_question_length:
            return False
        if len(answer) < self.config.min_answer_length:
            return False

        # Check maximum lengths
        if len(question) > self.config.max_question_length:
            return False
        if len(answer) > self.config.max_answer_length:
            return False

        # Check for meaningful content
        question_words = len(question.split())
        answer_words = len(answer.split())

        if question_words < 3 or answer_words < 5:
            return False

        # Check for breast cancer relevance
        if not self._is_breast_cancer_related(question + " " + answer):
            return False

        return True

    def _is_breast_cancer_related(self, text: str) -> bool:
        """
        Check if text is related to breast cancer domain

        Args:
            text: Text to check

        Returns:
            True if text is domain-related, False otherwise
        """
        text_lower = text.lower()

        # Check for domain-specific keywords
        domain_keywords = config.inference.domain_keywords
        return any(keyword in text_lower for keyword in domain_keywords)

    def expand_dataset_with_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand dataset using pattern variations

        Args:
            df: DataFrame with Questions, Answers, and optionally Patterns columns

        Returns:
            Expanded DataFrame
        """
        if not self.config.use_patterns or 'Patterns' not in df.columns:
            return df

        expanded_data = []

        for _, row in df.iterrows():
            # Add original question-answer pair
            expanded_data.append({
                'Questions': row['Questions'],
                'Answers': row['Answers'],
                'Tags': row.get('Tags', ''),
                'Source': row.get('Source', 'original')
            })

            # Process patterns if available
            if pd.notna(row['Patterns']) and row['Patterns'].strip():
                patterns = self._parse_patterns(row['Patterns'])

                for pattern in patterns:
                    if self.validate_qa_pair(pattern, row['Answers']):
                        expanded_data.append({
                            'Questions': pattern,
                            'Answers': row['Answers'],
                            'Tags': row.get('Tags', ''),
                            'Source': row.get('Source', 'pattern')
                        })

        expanded_df = pd.DataFrame(expanded_data)
        logger.info(f"Dataset expanded from {len(df)} to {len(expanded_df)} records")

        return expanded_df

    def _parse_patterns(self, patterns_text: str) -> List[str]:
        """
        Parse pattern text into individual patterns

        Args:
            patterns_text: Raw patterns text

        Returns:
            List of individual patterns
        """
        # Split by common delimiters
        patterns = re.split(r'[.;|]', patterns_text)

        # Clean and filter patterns
        cleaned_patterns = []
        for pattern in patterns:
            pattern = self.clean_text(pattern)
            if pattern and len(pattern) > 10:  # Filter very short patterns
                cleaned_patterns.append(pattern)

        return cleaned_patterns

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive preprocessing of the dataset

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Starting preprocessing of {len(df)} records")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Handle missing values
        df = df.dropna(subset=['Questions', 'Answers'])

        # Clean text columns
        logger.info("Cleaning text data...")
        df['Questions'] = df['Questions'].apply(self.clean_text)
        df['Answers'] = df['Answers'].apply(self.clean_text)

        # Normalize medical terms
        df['Questions'] = df['Questions'].apply(self.normalize_medical_terms)
        df['Answers'] = df['Answers'].apply(self.normalize_medical_terms)

        # Process patterns column if exists
        if 'Patterns' in df.columns:
            df['Patterns'] = df['Patterns'].fillna('').apply(self.clean_text)

        # Validate Q&A pairs
        logger.info("Validating Q&A pairs...")
        valid_mask = df.apply(
            lambda row: self.validate_qa_pair(row['Questions'], row['Answers']),
            axis=1
        )
        df = df[valid_mask]

        logger.info(f"Kept {len(df)} valid records after validation")

        # Expand dataset with patterns
        if self.config.use_patterns:
            df = self.expand_dataset_with_patterns(df)

        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Questions', 'Answers'])
        logger.info(f"Removed {initial_count - len(df)} duplicate records")

        # Add metadata
        df['processed_at'] = pd.Timestamp.now()
        df['word_count_q'] = df['Questions'].apply(lambda x: len(x.split()))
        df['word_count_a'] = df['Answers'].apply(lambda x: len(x.split()))

        logger.info(f"Preprocessing completed. Final dataset size: {len(df)} records")

        return df

    def create_train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df.get('Tags', None) if 'Tags' in df.columns else None
        )

        # Second split: train vs val
        val_size = self.config.validation_size / (1 - self.config.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=self.config.random_state,
            stratify=train_val_df.get('Tags', None) if 'Tags' in train_val_df.columns else None
        )

        logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def generate_data_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive data quality report

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing data statistics
        """
        report = {
            'total_records': len(df),
            'unique_questions': df['Questions'].nunique(),
            'unique_answers': df['Answers'].nunique(),
            'avg_question_length': df['Questions'].str.len().mean(),
            'avg_answer_length': df['Answers'].str.len().mean(),
            'avg_question_words': df['word_count_q'].mean() if 'word_count_q' in df.columns else None,
            'avg_answer_words': df['word_count_a'].mean() if 'word_count_a' in df.columns else None,
        }

        # Add column-specific statistics
        if 'Tags' in df.columns:
            report['unique_tags'] = df['Tags'].nunique()
            report['most_common_tags'] = df['Tags'].value_counts().head(5).to_dict()

        if 'Source' in df.columns:
            report['source_distribution'] = df['Source'].value_counts().to_dict()

        # Add quality metrics
        report['question_length_distribution'] = {
            'min': df['Questions'].str.len().min(),
            'max': df['Questions'].str.len().max(),
            'median': df['Questions'].str.len().median(),
            'std': df['Questions'].str.len().std()
        }

        report['answer_length_distribution'] = {
            'min': df['Answers'].str.len().min(),
            'max': df['Answers'].str.len().max(),
            'median': df['Answers'].str.len().median(),
            'std': df['Answers'].str.len().std()
        }

        return report

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = None
    ):
        """
        Save processed datasets to files

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Output directory path
        """
        if output_dir is None:
            output_dir = config.paths.data_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save datasets
        train_df.to_csv(output_path / "train_data.csv", index=False)
        val_df.to_csv(output_path / "val_data.csv", index=False)
        test_df.to_csv(output_path / "test_data.csv", index=False)

        # Generate and save data reports
        train_report = self.generate_data_report(train_df)
        val_report = self.generate_data_report(val_df)
        test_report = self.generate_data_report(test_df)

        import json
        with open(output_path / "data_reports.json", 'w') as f:
            json.dump({
                'train': train_report,
                'validation': val_report,
                'test': test_report
            }, f, indent=2)

        logger.info(f"Processed data saved to {output_dir}")

def main():
    """
    Main function to run data preprocessing
    """
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess breast cancer chatbot data")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default=config.paths.data_dir, help="Output directory")

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)

    # Preprocess data
    processed_df = preprocessor.preprocess_dataset(df)

    # Split data
    train_df, val_df, test_df = preprocessor.create_train_val_test_split(processed_df)

    # Save processed data
    preprocessor.save_processed_data(train_df, val_df, test_df, args.output)

    # Print summary
    print("\nData Preprocessing Summary:")
    print("=" * 40)
    print(f"Original records: {len(df)}")
    print(f"Processed records: {len(processed_df)}")
    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(val_df)}")
    print(f"Test records: {len(test_df)}")

if __name__ == "__main__":
    main()

"""
Dataset Module for Breast Cancer Chatbot
========================================

This module contains PyTorch Dataset classes for handling breast cancer Q&A data
for training transformer models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from transformers import AutoTokenizer
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)


class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset class for breast cancer Q&A data

    This dataset handles the preprocessing and tokenization of question-answer pairs
    for fine-tuning transformer models.
    """

    def __init__(self,
        questions: List[str],
        answers: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = None,
        conversation_format: str = "qa"):
        """
        Initialize the dataset

        Args:
            questions: List of input questions
            answers: List of corresponding answers
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization
            conversation_format: Format for conversation ("qa", "chat", "instruction")
        """
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length or config.model.max_sequence_length
        self.conversation_format = conversation_format

        # Validate input
        assert len(questions) == len(
            answers), "Questions and answers must have same length"

        logger.info(
            f"Created dataset with {len(questions)} samples, max_length={self.max_length}")

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset

        Args:
            idx: Index of the item to retrieve

        Returns:
            Dictionary containing tokenized input and labels
        """
        question = str(self.questions[idx])
        answer = str(self.answers[idx])

        # Format conversation based on specified format
        if self.conversation_format == "qa":
            input_text = f"Question: {question}\nAnswer: {answer}"
        elif self.conversation_format == "chat":
            input_text = f"User: {question}\nAssistant: {answer}"
        elif self.conversation_format == "instruction":
            input_text = f"### Instruction:\n{question}\n\n### Response:\n{answer}"
        else:
            input_text = f"{question} {answer}"

        # Tokenize the input
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

    def get_sample_text(self, idx: int) -> str:
        """Get the formatted text for a sample (useful for debugging)"""
        question = str(self.questions[idx])
        answer = str(self.answers[idx])

        if self.conversation_format == "qa":
            return f"Question: {question}\nAnswer: {answer}"
        elif self.conversation_format == "chat":
            return f"User: {question}\nAssistant: {answer}"
        elif self.conversation_format == "instruction":
            return f"### Instruction:\n{question}\n\n### Response:\n{answer}"
        else:
            return f"{question} {answer}"


class BreastCancerInferenceDataset(Dataset):
    """
    Dataset class for inference/evaluation tasks

    This dataset is used for generating responses to questions during evaluation.
    """

    def __init__(
        self,
        questions: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = None
    ):
        """
        Initialize the inference dataset

        Args:
            questions: List of questions for inference
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length or config.model.max_sequence_length

        logger.info(
            f"Created inference dataset with {len(questions)} questions")

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single question for inference

        Args:
            idx: Index of the question

        Returns:
            Dictionary containing tokenized question
        """
        question = str(self.questions[idx])
        input_text = f"Question: {question}\nAnswer:"

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'original_question': question
        }


class DataCollatorForCausalLM:
    """
    Custom data collator for causal language modeling

    This collator handles batching and padding for training.
    """

    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = None):
        """
        Initialize the data collator

        Args:
            tokenizer: Hugging Face tokenizer
            pad_to_multiple_of: Pad to multiple of this value
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch

        Args:
            examples: List of examples from dataset

        Returns:
            Batched data
        """
        batch = {}

        # Stack tensors
        for key in examples[0].keys():
            if isinstance(examples[0][key], torch.Tensor):
                batch[key] = torch.stack([example[key]
                                         for example in examples])
            else:
                batch[key] = [example[key] for example in examples]

        return batch


class DatasetFactory:
    """
    Factory class for creating different types of datasets
    """

    @staticmethod
    def create_training_datasets(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        tokenizer: AutoTokenizer
    ) -> Tuple[BreastCancerDataset, BreastCancerDataset]:
        """
        Create training and validation datasets

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            tokenizer: Tokenizer to use

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_dataset = BreastCancerDataset(
            questions=train_df['Questions'].tolist(),
            answers=train_df['Answers'].tolist(),
            tokenizer=tokenizer,
            conversation_format="qa"
        )

        val_dataset = BreastCancerDataset(
            questions=val_df['Questions'].tolist(),
            answers=val_df['Answers'].tolist(),
            tokenizer=tokenizer,
            conversation_format="qa"
        )

        return train_dataset, val_dataset

    @staticmethod
    def create_inference_dataset(questions: List[str],
                                 tokenizer: AutoTokenizer) -> BreastCancerInferenceDataset:
        """
        Create dataset for inference

        Args:
            questions: List of questions
            tokenizer: Tokenizer to use

        Returns:
            Inference dataset
        """
        return BreastCancerInferenceDataset(
            questions=questions,
            tokenizer=tokenizer
        )

    @staticmethod
    def create_dataloaders(
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = None,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            num_workers: Number of worker processes

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        if batch_size is None:
            batch_size = config.training.batch_size

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return train_dataloader, val_dataloader


def load_datasets_from_files(data_dir: str, tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load datasets from preprocessed CSV files

    Args:
        data_dir: Directory containing the CSV files
        tokenizer: Tokenizer to use

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_dir)

    # Load DataFrames
    train_df = pd.read_csv(data_path / "train_data.csv")
    val_df = pd.read_csv(data_path / "val_data.csv")
    test_df = pd.read_csv(data_path / "test_data.csv")

    # Create datasets
    train_dataset = BreastCancerDataset(
        questions=train_df['Questions'].tolist(),
        answers=train_df['Answers'].tolist(),
        tokenizer=tokenizer
    )

    val_dataset = BreastCancerDataset(
        questions=val_df['Questions'].tolist(),
        answers=val_df['Answers'].tolist(),
        tokenizer=tokenizer
    )

    test_dataset = BreastCancerDataset(
        questions=test_df['Questions'].tolist(),
        answers=test_df['Answers'].tolist(),
        tokenizer=tokenizer
    )

    logger.info(
        f"Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def analyze_dataset_statistics(dataset: BreastCancerDataset) -> Dict:
    """
    Analyze statistics of a dataset

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary containing statistics
    """
    stats = {
        'total_samples': len(dataset),
        'question_lengths': [],
        'answer_lengths': [],
        'total_lengths': []
    }

    for i in range(len(dataset)):
        sample_text = dataset.get_sample_text(i)
        question = dataset.questions[i]
        answer = dataset.answers[i]

        stats['question_lengths'].append(len(question.split()))
        stats['answer_lengths'].append(len(answer.split()))
        stats['total_lengths'].append(len(sample_text.split()))

    # Calculate summary statistics
    for key in ['question_lengths', 'answer_lengths', 'total_lengths']:
        values = stats[key]
        stats[f'{key}_mean'] = np.mean(values)
        stats[f'{key}_std'] = np.std(values)
        stats[f'{key}_min'] = np.min(values)
        stats[f'{key}_max'] = np.max(values)
        stats[f'{key}_median'] = np.median(values)

    return stats


def main():
    """
    Main function for testing dataset functionality
    """
    # Load tokenizer for testing
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create sample data
    questions = [
        "What is breast cancer?",
        "What are the symptoms of breast cancer?",
        "How is breast cancer diagnosed?",
        "What are the treatment options?",
        "How can I prevent breast cancer?"
    ]

    answers = [
        "Breast cancer is a type of cancer that forms in breast tissue cells.",
        "Common symptoms include lumps, breast changes, and nipple discharge.",
        "Diagnosis involves mammograms, biopsies, and imaging tests.",
        "Treatment options include surgery, chemotherapy, and radiation therapy.",
        "Prevention includes regular screening, healthy lifestyle, and genetic counseling."
    ]

    # Create dataset
    dataset = BreastCancerDataset(questions, answers, tokenizer)

    # Test dataset functionality
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item keys: {list(dataset[0].keys())}")
    print(f"Sample text: {dataset.get_sample_text(0)}")

    # Analyze statistics
    stats = analyze_dataset_statistics(dataset)
    print(f"\nDataset Statistics:")
    print(
        f"Average question length: {stats['question_lengths_mean']:.2f} words")
    print(f"Average answer length: {stats['answer_lengths_mean']:.2f} words")
    print(f"Average total length: {stats['total_lengths_mean']:.2f} words")

    # Create dataloader
    train_dataset, val_dataset = DatasetFactory.create_training_datasets(
        pd.DataFrame({'Questions': questions[:3], 'Answers': answers[:3]}),
        pd.DataFrame({'Questions': questions[3:], 'Answers': answers[3:]}),
        tokenizer
    )

    train_loader, val_loader = DatasetFactory.create_dataloaders(
        train_dataset, val_dataset, batch_size=2)

    print(f"\nDataLoader test:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")


if __name__ == "__main__":
    main()

"""
Model Module for Breast Cancer Chatbot
======================================

This module contains the main chatbot class that handles model loading,
fine-tuning, and inference for breast cancer Q&A.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import re
import os
from datetime import datetime

from config import config
from data_preprocessor import DataPreprocessor
from dataset import DatasetFactory, load_datasets_from_files

logger = logging.getLogger(__name__)

class BreastCancerChatbot:
    """
    Main chatbot class that handles model loading, training, and inference
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the chatbot

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name or config.model.base_model_name
        self.tokenizer = None
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.training_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seed for reproducibility
        set_seed(config.seed)

        logger.info(f"Initialized chatbot with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

    def load_model(self, model_path: str = None):
        """
        Load the pre-trained model and tokenizer

        Args:
            model_path: Path to model directory (if None, loads from model_name)
        """
        try:
            model_path = model_path or self.model_name
            logger.info(f"Loading model from: {model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Added padding token")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            # Move model to device if not using device_map
            if not torch.cuda.is_available():
                self.model.to(self.device)

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def prepare_training_data(self, data_path: str) -> Tuple:
        """
        Prepare training data from CSV file

        Args:
            data_path: Path to the CSV file or directory with processed data

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info(f"Preparing training data from: {data_path}")

        if os.path.isdir(data_path):
            # Load from preprocessed files
            train_dataset, val_dataset, test_dataset = load_datasets_from_files(
                data_path, self.tokenizer
            )
        else:
            # Load and preprocess CSV file
            df = pd.read_csv(data_path)
            processed_df = self.preprocessor.preprocess_dataset(df)

            # Split data
            train_df, val_df, test_df = self.preprocessor.create_train_val_test_split(processed_df)

            # Create datasets
            train_dataset, val_dataset = DatasetFactory.create_training_datasets(
                train_df, val_df, self.tokenizer
            )
            test_dataset, _ = DatasetFactory.create_training_datasets(
                test_df, val_df.iloc[:0], self.tokenizer  # Empty val set for test
            )

        logger.info(f"Training data prepared - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def create_trainer(self, train_dataset, val_dataset, output_dir: str) -> Trainer:
        """
        Create a Hugging Face Trainer instance

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for model checkpoints

        Returns:
            Configured Trainer instance
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.training.num_epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_steps=config.training.warmup_steps,
            max_grad_norm=config.training.max_grad_norm,

            # Logging and evaluation
            logging_dir=f"{output_dir}/logs",
            logging_steps=config.training.logging_steps,
            evaluation_strategy=config.training.eval_strategy,
            eval_steps=config.training.logging_steps,
            save_strategy=config.training.save_strategy,
            save_steps=config.training.logging_steps,

            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=config.training.save_total_limit,

            # Performance optimizations
            dataloader_pin_memory=True,
            dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues

            # Disable external logging
            report_to=["wandb"] if config.use_wandb else None,

            # Mixed precision training
            fp16=torch.cuda.is_available(),

            # Seed for reproducibility
            seed=config.seed,
        )

        # Callbacks
        callbacks = []
        if config.training.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.training.early_stopping_patience,
                    early_stopping_threshold=config.training.early_stopping_threshold
                )
            )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )

        return trainer

    def fine_tune(self, data_path: str, output_dir: str = None) -> Dict:
        """
        Fine-tune the model on the breast cancer dataset

        Args:
            data_path: Path to the training data
            output_dir: Directory to save the fine-tuned model

        Returns:
            Training metrics and history
        """
        if output_dir is None:
            output_dir = config.paths.fine_tuned_model_dir

        logger.info("Starting fine-tuning process")

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Prepare data
        train_dataset, val_dataset, test_dataset = self.prepare_training_data(data_path)

        # Create trainer
        trainer = self.create_trainer(train_dataset, val_dataset, output_dir)

        # Train the model
        logger.info("Training started...")
        start_time = datetime.now()

        try:
            train_result = trainer.train()

            # Calculate training time
            training_time = datetime.now() - start_time
            logger.info(f"Training completed in {training_time}")

            # Save the model and tokenizer
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

            # Save training metrics
            metrics = {
                'training_time': str(training_time),
                'final_train_loss': train_result.training_loss,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'model_name': self.model_name,
                'config': config.to_dict()
            }

            # Save evaluation metrics
            eval_results = trainer.evaluate()
            metrics.update({f'final_{k}': v for k, v in eval_results.items()})

            # Save metrics to file
            with open(f"{output_dir}/training_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Model fine-tuning completed and saved to {output_dir}")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            logger.info(f"Final validation loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def generate_response(self, question: str, max_length: int = None, **generation_kwargs) -> str:
        """
        Generate a response to a user question

        Args:
            question: User's question
            max_length: Maximum length of the generated response
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please load or train the model first."

        # Use config defaults if not specified
        if max_length is None:
            max_length = config.inference.max_new_tokens

        # Clean the input question
        question = self.preprocessor.clean_text(question)

        # Check if question is breast cancer related
        if not self._is_breast_cancer_related(question):
            return ("I'm sorry, I can only answer questions related to breast cancer. "
                   "Please ask about breast cancer symptoms, treatment, diagnosis, or prevention.")

        # Prepare input
        input_text = f"Question: {question}\nAnswer:"

        try:
            # Tokenize input
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')
            inputs = inputs.to(self.device)

            # Set generation parameters
            generation_config = {
                'max_new_tokens': max_length,
                'temperature': config.inference.temperature,
                'top_p': config.inference.top_p,
                'top_k': config.inference.top_k,
                'do_sample': config.inference.do_sample,
                'repetition_penalty': config.inference.repetition_penalty,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            generation_config.update(generation_kwargs)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_config)

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part
            response = full_response.replace(input_text, "").strip()

            # Clean up the response
            response = self._clean_response(response)

            return response if response else "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, there was an error generating a response. Please try again."

    def _is_breast_cancer_related(self, question: str) -> bool:
        """
        Check if a question is related to breast cancer

        Args:
            question: User's question

        Returns:
            True if question is breast cancer related, False otherwise
        """
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in config.inference.domain_keywords)

    def _clean_response(self, response: str) -> str:
        """
        Clean the generated response

        Args:
            response: Raw generated response

        Returns:
            Cleaned response
        """
        # Remove repeated phrases
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()

        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line)

        response = ' '.join(cleaned_lines)

        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'

        # Remove very short responses
        if len(response.split()) < 3:
            return ""

        # Ensure response is within length limits
        if len(response) > config.inference.max_response_length:
            # Truncate at sentence boundary
            sentences = response.split('.')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '.') <= config.inference.max_response_length:
                    truncated += sentence + '.'
                else:
                    break
            response = truncated

        return response.strip()

    def batch_generate(self, questions: List[str], batch_size: int = 4, **generation_kwargs) -> List[str]:
        """
        Generate responses for multiple questions in batches

        Args:
            questions: List of questions
            batch_size: Number of questions to process at once
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generated responses
        """
        responses = []

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_responses = []

            for question in batch_questions:
                response = self.generate_response(question, **generation_kwargs)
                batch_responses.append(response)

            responses.extend(batch_responses)
            logger.info(f"Processed {min(i + batch_size, len(questions))}/{len(questions)} questions")

        return responses

    def save_model(self, output_dir: str):
        """
        Save the current model and tokenizer

        Args:
            output_dir: Directory to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded to save")

        os.makedirs(output_dir, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'saved_at': datetime.now().isoformat(),
            'config': config.to_dict()
        }

        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {output_dir}")

    def load_fine_tuned_model(self, model_dir: str):
        """
        Load a fine-tuned model

        Args:
            model_dir: Directory containing the fine-tuned model
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Check if required files exist
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]

        if missing_files:
            logger.warning(f"Some model files may be missing: {missing_files}")

        try:
            logger.info(f"Loading fine-tuned model from: {model_dir}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            # Move model to device if not using device_map
            if not torch.cuda.is_available():
                self.model.to(self.device)

            # Load metadata if available
            metadata_path = os.path.join(model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"Model metadata: {metadata}")

            logger.info("Fine-tuned model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            raise

    def evaluate_model(self, test_dataset, output_dir: str = None) -> Dict:
        """
        Evaluate the model on test dataset

        Args:
            test_dataset: Test dataset
            output_dir: Directory to save evaluation results

        Returns:
            Evaluation metrics
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded for evaluation")

        logger.info("Starting model evaluation")

        # Create evaluation trainer
        eval_args = TrainingArguments(
            output_dir=output_dir or "./eval_results",
            per_device_eval_batch_size=config.training.batch_size,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
        )

        # Run evaluation
        eval_results = trainer.evaluate()

        # Calculate additional metrics
        eval_metrics = {
            'eval_loss': eval_results['eval_loss'],
            'eval_perplexity': np.exp(eval_results['eval_loss']),
            'test_samples': len(test_dataset),
            'evaluation_time': datetime.now().isoformat()
        }

        # Save evaluation results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/evaluation_results.json", 'w') as f:
                json.dump(eval_metrics, f, indent=2)

        logger.info(f"Evaluation completed - Loss: {eval_metrics['eval_loss']:.4f}, "
                   f"Perplexity: {eval_metrics['eval_perplexity']:.4f}")

        return eval_metrics

    def get_model_info(self) -> Dict:
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        model_info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming 4 bytes per parameter
            'vocab_size': len(self.tokenizer) if self.tokenizer else None,
            'max_position_embeddings': getattr(self.model.config, 'max_position_embeddings', None),
            'hidden_size': getattr(self.model.config, 'hidden_size', None),
            'num_attention_heads': getattr(self.model.config, 'num_attention_heads', None),
            'num_hidden_layers': getattr(self.model.config, 'num_hidden_layers', None),
        }

        return model_info

    def benchmark_inference(self, sample_questions: List[str], num_runs: int = 3) -> Dict:
        """
        Benchmark inference performance

        Args:
            sample_questions: List of sample questions for benchmarking
            num_runs: Number of runs for averaging

        Returns:
            Benchmark results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded for benchmarking")

        logger.info(f"Running inference benchmark with {len(sample_questions)} questions, {num_runs} runs")

        times = []
        responses = []

        for run in range(num_runs):
            run_times = []
            run_responses = []

            for question in sample_questions:
                start_time = datetime.now()
                response = self.generate_response(question)
                end_time = datetime.now()

                inference_time = (end_time - start_time).total_seconds()
                run_times.append(inference_time)
                run_responses.append(response)

            times.append(run_times)
            responses.extend(run_responses)

        # Calculate statistics
        all_times = [time for run_times in times for time in run_times]

        benchmark_results = {
            'num_questions': len(sample_questions),
            'num_runs': num_runs,
            'total_inferences': len(all_times),
            'avg_inference_time': np.mean(all_times),
            'min_inference_time': np.min(all_times),
            'max_inference_time': np.max(all_times),
            'std_inference_time': np.std(all_times),
            'total_time': sum(all_times),
            'questions_per_second': len(all_times) / sum(all_times),
            'benchmark_timestamp': datetime.now().isoformat()
        }

        logger.info(
            f"Benchmark completed - Avg time: {benchmark_results['avg_inference_time']:.3f}s, "
            f"QPS: {benchmark_results['questions_per_second']:.2f}"
        )

        return benchmark_results

    def __del__(self):
        """
        Cleanup when object is destroyed
        """
        # Clear GPU memory if model was loaded
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer

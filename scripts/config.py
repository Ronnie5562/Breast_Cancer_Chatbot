"""
Configuration file for the Breast Cancer Chatbot
================================================

This file contains all configuration parameters for the chatbot system.
Modify these values to customize the behavior of your chatbot.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    # Base model name from Hugging Face
    base_model_name: str = "microsoft/DialoGPT-small"


    # Tokenization settings
    max_sequence_length: int = 512
    padding_strategy: str = "max_length"
    truncation: bool = True

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100

    # Gradient and optimization settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Model saving and evaluation
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Data file paths
    train_data_path: str = "data/breast_cancer_data.csv"
    test_data_path: str = "data/test_data.csv"

    # Data preprocessing
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42

    # Text cleaning parameters
    min_question_length: int = 5
    max_question_length: int = 500
    min_answer_length: int = 10
    max_answer_length: int = 1000

    # Data augmentation
    use_patterns: bool = True
    expand_with_paraphrases: bool = False

@dataclass
class InferenceConfig:
    """Configuration for inference parameters"""
    # Generation settings
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1

    # Response filtering
    min_response_length: int = 10
    max_response_length: int = 300

    # Domain filtering
    domain_keywords: List[str] = None

    def __post_init__(self):
        if self.domain_keywords is None:
            self.domain_keywords = [
                'breast cancer', 'breast tumor', 'breast carcinoma', 'mammogram',
                'mastectomy', 'lumpectomy', 'chemotherapy', 'radiation therapy',
                'hormone therapy', 'her2', 'estrogen receptor', 'progesterone receptor',
                'metastasis', 'lymph nodes', 'oncology', 'biopsy', 'breast screening',
                'brca', 'genetic testing', 'breast self-exam', 'ductal carcinoma',
                'lobular carcinoma', 'inflammatory breast cancer', 'triple negative',
                'tumor', 'cancer', 'malignant', 'benign', 'diagnosis', 'treatment',
                'symptoms', 'prevention', 'risk factors', 'screening', 'therapy'
            ]

@dataclass
class UIConfig:
    """Configuration for user interface"""
    # Gradio settings
    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860
    gradio_share: bool = True

    # Interface customization
    title: str = "🎗️ Breast Cancer Information Chatbot"
    description: str = "Ask me any questions about breast cancer symptoms, treatment, diagnosis, or prevention."

    # Example questions
    example_questions: List[str] = None

    def __post_init__(self):
        if self.example_questions is None:
            self.example_questions = [
                "What is breast cancer?",
                "What are the symptoms of breast cancer?",
                "How is breast cancer diagnosed?",
                "What are the treatment options for breast cancer?",
                "How can I prevent breast cancer?",
                "What is a mammogram?",
                "What is chemotherapy?",
                "What are the risk factors for breast cancer?",
                "What is the difference between ductal and lobular carcinoma?",
                "What does triple negative breast cancer mean?"
            ]

@dataclass
class PathConfig:
    """Configuration for file paths"""
    # Model directories
    base_model_dir: str = "models/base"
    fine_tuned_model_dir: str = "models/fine_tuned"
    experiment_dir: str = "experiments"

    # Data directories
    data_dir: str = "data"
    logs_dir: str = "logs"
    output_dir: str = "outputs"

    # Specific file paths
    training_log_file: str = "outputs/training.log"
    evaluation_results_file: str = "outputs/evaluation_results.json"
    hyperparameter_results_file: str = "outputs/hyperparameter_results.csv"

    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.base_model_dir, exist_ok=True)
        os.makedirs(self.fine_tuned_model_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

@dataclass
class ExperimentConfig:
    """Configuration for hyperparameter experiments"""
    # Hyperparameter ranges to test
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    num_epochs_options: List[int] = None

    # Experiment settings
    num_trials: int = 10
    optimization_metric: str = "bleu_score"  # or "eval_loss", "response_relevance"

    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4]

        if self.batch_sizes is None:
            self.batch_sizes = [2, 4, 8]

        if self.num_epochs_options is None:
            self.num_epochs_options = [2, 3, 4]

class Config:
    """Main configuration class that combines all configurations"""

    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.inference = InferenceConfig()
        self.ui = UIConfig()
        self.paths = PathConfig()
        self.experiment = ExperimentConfig()

        # Environment variables
        self.device = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
        self.seed = int(os.getenv("RANDOM_SEED", "42"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.wandb_project = os.getenv("WANDB_PROJECT", "breast-cancer-chatbot")
        self.use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "inference": self.inference.__dict__,
            "ui": self.ui.__dict__,
            "paths": self.paths.__dict__,
            "experiment": self.experiment.__dict__,
            "device": self.device,
            "seed": self.seed,
            "debug": self.debug,
            "log_level": self.log_level,
            "wandb_project": self.wandb_project,
            "use_wandb": self.use_wandb
        }

    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file"""
        import json

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        config = cls()

        # Update configuration with loaded values
        for section, values in config_dict.items():
            if hasattr(config, section) and isinstance(getattr(config, section), object):
                for key, value in values.items():
                    if hasattr(getattr(config, section), key):
                        setattr(getattr(config, section), key, value)
            elif hasattr(config, section):
                setattr(config, section, values)

        return config

# Global configuration instance
config = Config()

# Example usage:
if __name__ == "__main__":
    # Print current configuration
    print("Current Configuration:")
    print("=" * 50)

    for section_name in ['model', 'training', 'data', 'inference', 'ui', 'paths', 'experiment']:
        section = getattr(config, section_name)
        print(f"\n{section_name.upper()}:")
        for key, value in section.__dict__.items():
            print(f"  {key}: {value}")

    print(f"\nEnvironment:")
    print(f"  device: {config.device}")
    print(f"  seed: {config.seed}")
    print(f"  debug: {config.debug}")

    # Save configuration
    config.save_config("config.json")
    print("\nConfiguration saved to config.json")

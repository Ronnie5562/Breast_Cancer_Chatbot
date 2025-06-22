#!/usr/bin/env python3
"""
Training Script for Breast Cancer Chatbot
==========================================

This script demonstrates how to start training your breast cancer chatbot model.
"""

import os
import logging
from pathlib import Path
from model import BreastCancerChatbot

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)


def main():
    """Main training function"""

    # Step 1: Initialize the chatbot
    print("🚀 Initializing Breast Cancer Chatbot...")
    chatbot = BreastCancerChatbot()

    # Step 2: Load the base model
    print("📥 Loading base model...")
    try:
        chatbot.load_model()
        print("✅ Base model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Step 3: Prepare your data path
    # Replace this with the path to your actual data
    data_path = "path/to/your/breast_cancer_data.csv"

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please provide the correct path to your breast cancer Q&A dataset")
        print("Expected format: CSV with 'question' and 'answer' columns")
        return

    # Step 4: Set output directory for fine-tuned model
    output_dir = "./fine_tuned_breast_cancer_model"
    os.makedirs(output_dir, exist_ok=True)

    # Step 5: Start fine-tuning
    print("🎯 Starting fine-tuning process...")
    print(f"📂 Data source: {data_path}")
    print(f"💾 Output directory: {output_dir}")

    try:
        # This will automatically:
        # - Load and preprocess your data
        # - Split into train/validation/test sets
        # - Create datasets
        # - Configure training parameters
        # - Start training with early stopping
        # - Save the fine-tuned model
        training_metrics = chatbot.fine_tune(
            data_path=data_path,
            output_dir=output_dir
        )

        print("🎉 Training completed successfully!")
        print("📊 Training Summary:")
        print(
            f"   • Final training loss: {training_metrics.get('final_train_loss', 'N/A')}")
        print(
            f"   • Final validation loss: {training_metrics.get('final_eval_loss', 'N/A')}")
        print(
            f"   • Training time: {training_metrics.get('training_time', 'N/A')}")
        print(f"   • Model saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        logging.error(f"Training error: {e}", exc_info=True)
        return

    # Step 6: Test the fine-tuned model
    print("\n🧪 Testing the fine-tuned model...")
    test_questions = [
        "What are the early signs of breast cancer?",
        "How is breast cancer diagnosed?",
        "What are the treatment options for breast cancer?"
    ]

    try:
        # Load the fine-tuned model
        chatbot.load_fine_tuned_model(output_dir)

        print("Sample responses from fine-tuned model:")
        for i, question in enumerate(test_questions, 1):
            response = chatbot.generate_response(question)
            print(f"\n{i}. Q: {question}")
            print(f"   A: {response[:200]}...")  # Show first 200 chars

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        logging.error(f"Testing error: {e}", exc_info=True)


def prepare_sample_data():
    """
    Helper function to create a sample dataset if you don't have one yet
    """
    import pandas as pd

    # Sample breast cancer Q&A data
    sample_data = [
        {
            "question": "What are the early signs of breast cancer?",
            "answer": "Early signs of breast cancer may include a lump in the breast or underarm, changes in breast size or shape, skin dimpling, nipple discharge, or changes in the skin texture of the breast."
        },
        {
            "question": "How is breast cancer diagnosed?",
            "answer": "Breast cancer is typically diagnosed through a combination of physical examination, imaging tests like mammograms or ultrasounds, and tissue biopsy to confirm the presence of cancer cells."
        },
        {
            "question": "What are the risk factors for breast cancer?",
            "answer": "Risk factors include age, family history, genetic mutations (BRCA1/BRCA2), personal history of breast cancer, radiation exposure, and certain lifestyle factors."
        },
        # Add more Q&A pairs here...
    ]

    df = pd.DataFrame(sample_data)
    df.to_csv("sample_breast_cancer_data.csv", index=False)
    print("📄 Sample dataset created: sample_breast_cancer_data.csv")
    return "sample_breast_cancer_data.csv"


if __name__ == "__main__":
    print("=" * 60)
    print("🏥 BREAST CANCER CHATBOT TRAINING")
    print("=" * 60)

    # Uncomment the line below if you need to create sample data first
    # prepare_sample_data()

    main()

    print("\n" + "=" * 60)
    print("Training process completed!")
    print("Check training.log for detailed logs")
    print("=" * 60)

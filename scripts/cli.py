#!/usr/bin/env python3
"""
Simple Diagnostic CLI for Breast Cancer Chatbot
==============================================

Debug version to help identify model issues.
"""

import argparse
import sys
import os
import torch

try:
    from model import BreastCancerChatbot
except ImportError as e:
    print(f"Error importing model: {e}")
    print("Please ensure model.py is in the same directory or in your Python path.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Breast Cancer Chatbot CLI - Debug Version")
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to your trained model directory'
    )
    parser.add_argument(
        '--base', '-b',
        action='store_true',
        help='Load base model instead of fine-tuned model (for comparison)'
    )

    args = parser.parse_args()

    # Check if model directory exists
    if not args.base and not os.path.exists(args.model):
        print(f"Error: Model directory '{args.model}' not found.")
        sys.exit(1)

    print("🔬 Breast Cancer Chatbot - Debug Version")
    print("=" * 50)

    # Load the chatbot
    try:
        chatbot = BreastCancerChatbot()

        if args.base:
            print("Loading BASE model for comparison...")
            chatbot.load_model()
        else:
            print("Loading your FINE-TUNED model...")
            chatbot.load_fine_tuned_model(args.model)

        print("✓ Model loaded successfully!")

        # Show model info
        print("\n📊 Model Information:")
        info = chatbot.get_model_info()
        print(f"Model: {info.get('model_name', 'Unknown')}")
        print(f"Device: {info.get('device', 'Unknown')}")
        print(f"Parameters: {info.get('total_parameters', 0):,}")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n🧪 Testing with {'BASE' if args.base else 'FINE-TUNED'} model")
    print("You can now ask questions about breast cancer.")
    print("Type 'quit' to exit, 'debug' for generation details.\n")

    # Interactive loop
    while True:
        try:
            # Get user question
            question = input("Question: ").strip()

            # Check for exit
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Debug mode
            if question.lower() == 'debug':
                print("\n🔧 Current generation settings:")
                try:
                    from config import config
                    print(f"Temperature: {config.inference.temperature}")
                    print(f"Top-p: {config.inference.top_p}")
                    print(f"Top-k: {config.inference.top_k}")
                    print(f"Max tokens: {config.inference.max_new_tokens}")
                    print(f"Do sample: {config.inference.do_sample}")
                    print(
                        f"Repetition penalty: {config.inference.repetition_penalty}")
                except:
                    print("Could not load config settings")
                print()
                continue

            # Skip empty questions
            if not question:
                continue

            # Generate response with conservative settings
            print("Answer: ", end="", flush=True)
            try:
                # Try with more conservative generation parameters
                response = chatbot.generate_response(
                    question,
                    temperature=0.7,      # Lower temperature
                    top_p=0.9,           # More focused sampling
                    top_k=50,            # Limit vocabulary
                    max_new_tokens=150,  # Shorter responses
                    do_sample=True,
                    repetition_penalty=1.1
                )
                print(response)
            except Exception as e:
                print(f"Generation error: {e}")
                # Try with even more conservative settings
                try:
                    print("\nTrying with very conservative settings...")
                    response = chatbot.generate_response(
                        question,
                        temperature=0.3,
                        top_p=0.8,
                        max_new_tokens=100,
                        do_sample=False  # Greedy decoding
                    )
                    print(response)
                except Exception as e2:
                    print(f"Still failed: {e2}")

            print()  # Empty line for readability

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

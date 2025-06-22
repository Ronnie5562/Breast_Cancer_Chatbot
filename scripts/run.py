"""
Gradio Interface for Breast Cancer Chatbot
==========================================

This script creates a web interface for the breast cancer chatbot using Gradio.
It provides an intuitive chat interface with additional features like model info,
settings, and conversation history.
"""

import gradio as gr
import torch
import logging
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import pandas as pd
from pathlib import Path

# Import your chatbot model
from model import BreastCancerChatbot
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioChatbotInterface:
    """
    Gradio interface wrapper for the breast cancer chatbot
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the Gradio interface

        Args:
            model_path: Path to the fine-tuned model (optional)
        """
        self.chatbot = BreastCancerChatbot()
        self.model_path = model_path
        self.conversation_history = []
        self.model_loaded = False
        self.model_info = {}

        # Load model on initialization
        self._load_model()

    def _load_model(self):
        """Load the chatbot model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading fine-tuned model from: {self.model_path}")
                self.chatbot.load_fine_tuned_model(self.model_path)
            else:
                logger.info("Loading base model")
                self.chatbot.load_model()

            self.model_loaded = True
            self.model_info = self.chatbot.get_model_info()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False

    def chat_response(self, message: str, history: List[List[str]],
                     temperature: float, max_tokens: int, top_p: float) -> Tuple[str, List[List[str]]]:
        """
        Generate chatbot response

        Args:
            message: User message
            history: Chat history
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter

        Returns:
            Tuple of (empty string, updated history)
        """
        if not self.model_loaded:
            error_msg = "❌ Model not loaded. Please check the model path and restart."
            history.append([message, error_msg])
            return "", history

        if not message.strip():
            return "", history

        try:
            # Generate response with custom parameters
            response = self.chatbot.generate_response(
                message,
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Add to conversation history
            conversation_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_message': message,
                'bot_response': response,
                'settings': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'top_p': top_p
                }
            }
            self.conversation_history.append(conversation_entry)

            # Update chat history
            history.append([message, response])

        except Exception as e:
            error_response = f"❌ Error generating response: {str(e)}"
            history.append([message, error_response])
            logger.error(f"Error in chat_response: {e}")

        return "", history

    def clear_conversation(self) -> Tuple[List, str]:
        """
        Clear the conversation history

        Returns:
            Tuple of (empty history, status message)
        """
        self.conversation_history = []
        return [], "✅ Conversation cleared"

    def get_model_status(self) -> str:
        """
        Get current model status

        Returns:
            Model status information
        """
        if not self.model_loaded:
            return "❌ **Model Status**: Not loaded"

        info = self.model_info
        status = f"""
        ✅ **Model Status**: Loaded and ready

        **Model Information:**
        - Model Name: {info.get('model_name', 'Unknown')}
        - Device: {info.get('device', 'Unknown')}
        - Total Parameters: {info.get('total_parameters', 0):,}
        - Trainable Parameters: {info.get('trainable_parameters', 0):,}
        - Model Size: {info.get('model_size_mb', 0):.1f} MB
        - Vocabulary Size: {info.get('vocab_size', 'Unknown')}
        - Max Sequence Length: {info.get('max_position_embeddings', 'Unknown')}
        """

        return status

    def export_conversation(self) -> Tuple[str, str]:
        """
        Export conversation history to JSON

        Returns:
            Tuple of (file_path, status_message)
        """
        if not self.conversation_history:
            return None, "❌ No conversation history to export"

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"breast_cancer_chat_history_{timestamp}.json"

            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'model_info': self.model_info,
                'total_conversations': len(self.conversation_history),
                'conversations': self.conversation_history
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return filename, f"✅ Conversation exported to {filename}"

        except Exception as e:
            return None, f"❌ Export failed: {str(e)}"

    def get_sample_questions(self) -> List[str]:
        """
        Get sample questions for users to try

        Returns:
            List of sample questions
        """
        return [
            "What are the early signs of breast cancer?",
            "How is breast cancer diagnosed?",
            "What are the different types of breast cancer treatment?",
            "Can breast cancer be prevented?",
            "What lifestyle changes can reduce breast cancer risk?",
            "How often should I get a mammogram?",
            "What is the difference between a lump and breast cancer?",
            "What should I do if I find a lump in my breast?",
            "Are there genetic factors that increase breast cancer risk?",
            "What is the survival rate for breast cancer?"
        ]

    def handle_sample_question(self, question: str) -> str:
        """
        Handle sample question selection

        Args:
            question: Selected sample question

        Returns:
            The question text for the input field
        """
        return question

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface

        Returns:
            Gradio Blocks interface
        """
        # Custom CSS for better styling
        custom_css = """
        .main-header {
            text-align: center;
            color: #2E86AB;
            margin-bottom: 20px;
        }
        .status-box {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        .sample-questions {
            background-color: #f0f8ff;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            color: #856404;
        }
        """

        with gr.Blocks(css=custom_css, title="Breast Cancer AI Assistant") as interface:

            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🎗️ Breast Cancer AI Assistant</h1>
                <p>An AI-powered chatbot to help answer questions about breast cancer</p>
                <div class="warning-box">
                    <strong styles="color:red;">⚠️ Medical Disclaimer:</strong> This AI assistant provides general information only
                    and should not replace professional medical advice. Always consult healthcare professionals
                    for medical concerns, diagnosis, or treatment decisions.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    # Main chat interface
                    chatbot_interface = gr.Chatbot(
                        label="Chat with Breast Cancer AI Assistant",
                        height=500,
                        show_label=True,
                        container=True
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask me anything about breast cancer...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("Clear Conversation", variant="secondary")
                        export_btn = gr.Button("Export Chat", variant="secondary")

                with gr.Column(scale=1):
                    # Settings panel
                    gr.Markdown("### ⚙️ Generation Settings")

                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=config.inference.temperature if hasattr(config.inference, 'temperature') else 0.7,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness (lower = more focused)"
                    )

                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=config.inference.max_new_tokens if hasattr(config.inference, 'max_new_tokens') else 200,
                        step=10,
                        label="Max Tokens",
                        info="Maximum response length"
                    )

                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=config.inference.top_p if hasattr(config.inference, 'top_p') else 0.9,
                        step=0.05,
                        label="Top-p",
                        info="Nucleus sampling parameter"
                    )

                    # Model status
                    gr.Markdown("### 📊 Model Status")
                    status_display = gr.Markdown(self.get_model_status())
                    refresh_btn = gr.Button("Refresh Status", variant="secondary", size="sm")

            # Sample questions section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 💡 Sample Questions")
                    sample_questions = self.get_sample_questions()

                    with gr.Row():
                        for i in range(0, len(sample_questions), 2):
                            with gr.Column():
                                if i < len(sample_questions):
                                    q1_btn = gr.Button(
                                        sample_questions[i][:50] + "..." if len(sample_questions[i]) > 50 else sample_questions[i],
                                        variant="outline",
                                        size="sm"
                                    )
                                    q1_btn.click(
                                        fn=lambda q=sample_questions[i]: q,
                                        outputs=msg_input
                                    )

                                if i + 1 < len(sample_questions):
                                    q2_btn = gr.Button(
                                        sample_questions[i + 1][:50] + "..." if len(sample_questions[i + 1]) > 50 else sample_questions[i + 1],
                                        variant="outline",
                                        size="sm"
                                    )
                                    q2_btn.click(
                                        fn=lambda q=sample_questions[i + 1]: q,
                                        outputs=msg_input
                                    )

            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                <p><strong>About this AI Assistant:</strong></p>
                <p>This chatbot is designed to provide general information about breast cancer.
                It uses advanced AI to answer questions about symptoms, diagnosis, treatment options,
                prevention, and general breast health topics.</p>
                <p><em>Remember: Always consult with healthcare professionals for personalized medical advice.</em></p>
            </div>
            """)

            # Event handlers
            def send_message(message, history, temp, max_tok, top_p_val):
                return self.chat_response(message, history, temp, max_tok, top_p_val)

            # Send button click
            send_btn.click(
                fn=send_message,
                inputs=[msg_input, chatbot_interface, temperature, max_tokens, top_p],
                outputs=[msg_input, chatbot_interface]
            )

            # Enter key press
            msg_input.submit(
                fn=send_message,
                inputs=[msg_input, chatbot_interface, temperature, max_tokens, top_p],
                outputs=[msg_input, chatbot_interface]
            )

            # Clear conversation
            clear_btn.click(
                fn=self.clear_conversation,
                outputs=[chatbot_interface, status_display]
            )

            # Export conversation
            export_btn.click(
                fn=self.export_conversation,
                outputs=[gr.File(visible=False), status_display]
            )

            # Refresh status
            refresh_btn.click(
                fn=self.get_model_status,
                outputs=status_display
            )

        return interface


def main():
    """
    Main function to launch the Gradio interface
    """
    import argparse

    parser = argparse.ArgumentParser(description="Launch Breast Cancer Chatbot Interface")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the interface on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create interface
        print("🚀 Initializing Breast Cancer AI Assistant...")
        chatbot_interface = GradioChatbotInterface(model_path=args.model_path)

        print("🌐 Creating Gradio interface...")
        interface = chatbot_interface.create_interface()

        print(f"✅ Launching interface on http://{args.host}:{args.port}")

        # Launch the interface
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
            # show_tips=True,
            inbrowser=True
        )

    except Exception as e:
        print(f"❌ Failed to launch interface: {e}")
        raise


if __name__ == "__main__":
    main()

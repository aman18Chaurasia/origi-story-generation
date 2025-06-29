#!/usr/bin/env python3
"""
LLM Application using Open Source Models
Supports Hugging Face models, LangChain, and local models like Llama
"""

import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import logging
from typing import Dict, Any, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMModelManager:
    """Manager class for handling different LLM models"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
    def load_huggingface_model(self, model_name: str, use_gpu: bool = True):
        """Load a Hugging Face model"""
        try:
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.pipelines[model_name] = pipe
            
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def get_pipeline(self, model_name: str):
        """Get the pipeline for a specific model"""
        return self.pipelines.get(model_name)

class LangChainManager:
    """Manager for LangChain operations"""
    
    def __init__(self, model_manager: LLMModelManager):
        self.model_manager = model_manager
        self.chains = {}
        self.memories = {}
    
    def create_conversation_chain(self, model_name: str, template: str = None):
        """Create a conversation chain with memory"""
        pipeline = self.model_manager.get_pipeline(model_name)
        if not pipeline:
            return None
        
        # Create Hugging Face LLM wrapper
        llm = HuggingFacePipeline(
            pipeline=pipeline,
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512,
                "do_sample": True,
                "pad_token_id": pipeline.tokenizer.eos_token_id
            }
        )
        
        # Create memory
        memory = ConversationBufferMemory()
        
        # Create custom prompt template if provided
        if template:
            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=template
            )
            chain = ConversationChain(
                llm=llm,
                memory=memory,
                prompt=prompt,
                verbose=True
            )
        else:
            chain = ConversationChain(
                llm=llm,
                memory=memory,
                verbose=True
            )
        
        self.chains[model_name] = chain
        self.memories[model_name] = memory
        
        return chain
    
    def get_chain(self, model_name: str):
        """Get conversation chain for a model"""
        return self.chains.get(model_name)
    
    def clear_memory(self, model_name: str):
        """Clear conversation memory for a model"""
        if model_name in self.memories:
            self.memories[model_name].clear()

class LLMApplication:
    """Main LLM Application class"""
    
    def __init__(self):
        self.model_manager = LLMModelManager()
        self.langchain_manager = LangChainManager(self.model_manager)
        
        # Available models
        self.available_models = {
            "microsoft/DialoGPT-medium": "DialoGPT Medium",
            "microsoft/DialoGPT-small": "DialoGPT Small",
            "distilgpt2": "DistilGPT2",
            "gpt2": "GPT2",
            "facebook/opt-350m": "OPT-350M",
            "EleutherAI/gpt-neo-125M": "GPT-Neo 125M"
        }
    
    def initialize_streamlit_app(self):
        """Initialize the Streamlit web interface"""
        st.set_page_config(
            page_title="LLM Application",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 Open Source LLM Application")
        st.markdown("---")
        
        # Sidebar for model selection and configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            selected_model = st.selectbox(
                "Select Model",
                options=list(self.available_models.keys()),
                format_func=lambda x: self.available_models[x]
            )
            
            # Load model button
            if st.button("Load Model"):
                with st.spinner(f"Loading {self.available_models[selected_model]}..."):
                    success = self.model_manager.load_huggingface_model(selected_model)
                    if success:
                        st.success("Model loaded successfully!")
                        # Create conversation chain
                        self.langchain_manager.create_conversation_chain(selected_model)
                    else:
                        st.error("Failed to load model")
            
            # Generation parameters
            st.subheader("Generation Parameters")
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
            max_length = st.slider("Max Length", 50, 1000, 200, 50)
            
            # Clear conversation
            if st.button("Clear Conversation"):
                self.langchain_manager.clear_memory(selected_model)
                st.session_state.messages = []
                st.success("Conversation cleared!")
        
        # Main chat interface
        st.header("💬 Chat Interface")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Enter your message..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = self.generate_response(selected_model, prompt, temperature, max_length)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Additional features
        st.markdown("---")
        
        # Model information
        with st.expander("📊 Model Information"):
            if selected_model in self.model_manager.models:
                model = self.model_manager.models[selected_model]
                st.write(f"**Model Name:** {selected_model}")
                st.write(f"**Model Type:** {type(model).__name__}")
                st.write(f"**Device:** {next(model.parameters()).device}")
                st.write(f"**Parameters:** {sum(p.numel() for p in model.parameters()):,}")
            else:
                st.write("No model loaded")
        
        # Batch processing
        with st.expander("📝 Batch Processing"):
            st.subheader("Process Multiple Prompts")
            batch_input = st.text_area("Enter prompts (one per line):")
            
            if st.button("Process Batch"):
                if batch_input and selected_model in self.model_manager.pipelines:
                    prompts = batch_input.strip().split('\n')
                    results = []
                    
                    progress_bar = st.progress(0)
                    for i, prompt in enumerate(prompts):
                        response = self.generate_response(selected_model, prompt.strip(), temperature, max_length)
                        results.append(f"**Prompt:** {prompt}\n**Response:** {response}\n")
                        progress_bar.progress((i + 1) / len(prompts))
                    
                    st.subheader("Results:")
                    for result in results:
                        st.markdown(result)
                        st.markdown("---")
    
    def generate_response(self, model_name: str, prompt: str, temperature: float = 0.7, max_length: int = 200) -> str:
        """Generate response using the selected model"""
        try:
            # Try using LangChain conversation chain first
            chain = self.langchain_manager.get_chain(model_name)
            if chain:
                response = chain.predict(input=prompt)
                return response.strip()
            
            # Fallback to direct pipeline usage
            pipeline = self.model_manager.get_pipeline(model_name)
            if pipeline:
                # Update pipeline parameters
                pipeline.model.config.temperature = temperature
                
                response = pipeline(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=pipeline.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                generated_text = response[0]['generated_text']
                # Remove the original prompt from the response
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                return generated_text
            
            return "Error: No model available for generation"
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

def main():
    """Main function to run the application"""
    app = LLMApplication()
    app.initialize_streamlit_app()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Clean Origin Stories Generator
Fixed version with proper error handling and API key validation
"""

import streamlit as st
import openai
import json
import requests
from PIL import Image
import io
from datetime import datetime
from typing import Dict, List, Optional

# Set page config first (only once)
st.set_page_config(
    page_title="Origin Stories Generator",
    page_icon="📚",
    layout="wide"
)

# Configuration class
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    MODEL_NAME = "gpt-3.5-turbo"
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000

class SimpleOriginStoryGenerator:
    """Simple origin story generator using OpenAI API directly"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = Config.MODEL_NAME
    
    def generate_story(self, topic: str, story_type: str) -> str:
        """Generate a single origin story"""
        
        prompts = {
            "mythological": f"""Create an enchanting mythological origin story for {topic}.
            
            Include:
            - Ancient gods, spirits, or magical beings
            - A cosmic or divine origin
            - Symbolic meaning and deeper wisdom
            - Beautiful, poetic language
            - A sense of wonder and mystery
            
            Make it feel like an ancient myth passed down through generations.""",
            
            "scientific": f"""Explain the scientific origin and development of {topic}.
            
            Provide:
            - Scientific principles involved
            - Natural processes or human innovation
            - Timeline of development
            - Key discoveries or breakthroughs
            - Current understanding
            
            Make it accessible and engaging for general audiences.""",
            
            "cultural": f"""Tell the cultural and social origin story of {topic}.
            
            Focus on:
            - Cultural significance and meaning
            - Social context and human needs
            - How it spread across cultures
            - Variations in different societies
            - Impact on human civilization""",
            
            "creative": f"""Create an imaginative, alternate origin story for {topic}.
            
            Imagine if {topic} had originated differently:
            - In a different time period
            - Through unexpected circumstances
            - With unusual characters or events
            - With surprising consequences
            
            Make it creative, engaging, and thought-provoking.""",
            
            "historical": f"""Research and provide historical facts about the origin of {topic}.
            
            Include:
            - Key historical dates and events
            - Important figures or inventors
            - Geographical origins
            - How it evolved over time
            - Major milestones in its development"""
        }
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a master storyteller and historian who creates engaging origin stories."},
                    {"role": "user", "content": prompts.get(story_type, prompts["mythological"])}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating {story_type} story: {str(e)}"
    
    def generate_all_stories(self, topic: str, selected_types: List[str]) -> Dict[str, str]:
        """Generate multiple types of origin stories"""
        stories = {}
        
        for story_type in selected_types:
            stories[story_type] = self.generate_story(topic, story_type)
        
        return stories

class SimpleImageGenerator:
    """Simple image generator using Hugging Face API"""
    
    def __init__(self, api_key: str):
        self.api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def generate_image(self, prompt: str) -> Optional[Image.Image]:
        """Generate an image based on the prompt"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": prompt},
                timeout=30
            )
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return image
            else:
                return None
                
        except Exception as e:
            return None

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #2E86AB;
    margin-bottom: 2rem;
}
.story-container {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 5px solid #2E86AB;
}
.story-type {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2E86AB;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📚 Origin Stories Generator</h1>', unsafe_allow_html=True)
st.markdown("*Discover how anything came to be through the power of AI storytelling*")

# Sidebar
with st.sidebar:
    st.header("🎛️ Settings")
    
    # API Keys with validation
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for story generation. Get from: https://platform.openai.com/api-keys",
        placeholder="sk-..."
    )
    
    hf_key = st.text_input(
        "Hugging Face API Key",
        type="password",
        help="Optional for image generation. Get from: https://huggingface.co/settings/tokens",
        placeholder="hf_..."
    )
    
    # Validate and set API keys
    api_key_valid = False
    if openai_key and len(openai_key) > 20 and openai_key.startswith('sk-'):
        Config.OPENAI_API_KEY = openai_key
        api_key_valid = True
        st.success("✅ OpenAI API Key set!")
    elif openai_key:
        st.error("❌ Invalid OpenAI API Key format (should start with 'sk-')")
    
    hf_key_valid = False
    if hf_key and len(hf_key) > 10:
        Config.HUGGINGFACE_API_KEY = hf_key
        hf_key_valid = True
        st.success("✅ Hugging Face API Key set!")
    
    st.divider()
    
    # Story type selection
    st.header("📖 Story Types")
    story_types = st.multiselect(
        "Select story types to generate:",
        ["mythological", "scientific", "cultural", "creative", "historical"],
        default=["mythological", "scientific"]
    )
    
    # Options
    st.header("⚙️ Options")
    generate_images = st.checkbox("Generate images", value=hf_key_valid)
    
    # Example topics
    st.header("💡 Example Topics")
    examples = ["Coffee", "Internet", "Democracy", "Music", "Pizza", "Money", "Writing", "Fire"]
    
    for example in examples:
        if st.button(f"📝 {example}", key=f"ex_{example}"):
            st.session_state.topic_input = example

# Main content
st.header("🔍 Enter Your Topic")

topic = st.text_input(
    "What origin story would you like to discover?",
    placeholder="Enter anything: pizza, internet, democracy, love, etc.",
    value=st.session_state.get('topic_input', ''),
    key="topic_main"
)

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button(
        "✨ Generate Origin Stories",
        type="primary",
        use_container_width=True
    )

# Main logic
if generate_button:
    if not topic:
        st.warning("⚠️ Please enter a topic!")
    elif not api_key_valid:
        st.error("❌ Please provide a valid OpenAI API key in the sidebar!")
        st.info("🔑 Get your API key from: https://platform.openai.com/api-keys")
    elif not story_types:
        st.warning("⚠️ Please select at least one story type!")
    else:
        try:
            # Initialize generator
            generator = SimpleOriginStoryGenerator(Config.OPENAI_API_KEY)
            
            # Display stories
            st.header(f"📖 Origin Stories: {topic.title()}")
            
            # Progress bar
            progress_bar = st.progress(0)
            total_stories = len(story_types)
            
            stories = {}
            for i, story_type in enumerate(story_types):
                progress_bar.progress((i + 1) / total_stories)
                
                with st.spinner(f"🔮 Generating {story_type} story..."):
                    story = generator.generate_story(topic, story_type)
                    stories[story_type] = story
                
                # Display story immediately
                st.markdown(f'<div class="story-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="story-type">{story_type.title()} Origin Story</div>', unsafe_allow_html=True)
                st.write(story)
                
                # Generate image if requested
                if generate_images and hf_key_valid:
                    with st.spinner(f"🎨 Generating {story_type} image..."):
                        image_generator = SimpleImageGenerator(Config.HUGGINGFACE_API_KEY)
                        image_prompt = f"{story_type} origin of {topic}, artistic, detailed, beautiful, fantasy art"
                        image = image_generator.generate_image(image_prompt)
                        
                        if image:
                            st.image(image, caption=f"{story_type.title()} visualization", use_column_width=True)
                        else:
                            st.info("Image generation unavailable at the moment")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.divider()
            
            progress_bar.progress(1.0)
            st.success("✅ All stories generated!")
            
            # Save to session state
            st.session_state.last_stories = {
                'topic': topic,
                'stories': stories,
                'timestamp': datetime.now()
            }
            
            # Export section
            st.header("💾 Export Stories")
            
            # Create downloadable content
            export_data = {
                'topic': topic,
                'generated_at': datetime.now().isoformat(),
                'stories': stories
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Download as JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"origin_stories_{topic.replace(' ', '_')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Create markdown version
                markdown_content = f"# Origin Stories: {topic}\n\n"
                markdown_content += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                
                for story_type, story_content in stories.items():
                    markdown_content += f"## {story_type.title()} Origin Story\n\n"
                    markdown_content += f"{story_content}\n\n"
                
                st.download_button(
                    "📝 Download as Markdown",
                    data=markdown_content,
                    file_name=f"origin_stories_{topic.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
            
        except Exception as e:
            st.error(f"❌ Error generating stories: {str(e)}")
            if "invalid_api_key" in str(e).lower():
                st.info("🔑 Please check that your OpenAI API key is correct")
            else:
                st.info("Please try again or check your internet connection")

# Instructions
with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
    ### Steps:
    1. **Get API Keys**: 
       - OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys) (Required)
       - Hugging Face token from [HF Settings](https://huggingface.co/settings/tokens) (Optional for images)
    
    2. **Enter API Keys** in the sidebar (they start with 'sk-' for OpenAI)
    
    3. **Choose a topic** - anything you're curious about!
    
    4. **Select story types** you want to generate
    
    5. **Click Generate** and discover amazing origin stories
    
    ### Story Types:
    - **Mythological**: Ancient myths and legends
    - **Scientific**: Factual, evidence-based explanations  
    - **Cultural**: Social and cultural perspectives
    - **Creative**: Imaginative alternative histories
    - **Historical**: Factual timeline and key events
    
    ### Features:
    - 🎨 **Image generation** (with Hugging Face API)
    - 💾 **Export stories** in JSON or Markdown
    - 📱 **Mobile-friendly** interface
    - 🔒 **Secure** - API keys are not stored
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and OpenAI API</p>
    <p>🔒 Your API keys are secure and not stored anywhere</p>
</div>
""", unsafe_allow_html=True)
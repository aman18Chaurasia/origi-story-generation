#!/usr/bin/env python3
"""
Simple test script for the LLM application
Save this as 'simple_test.py' in the same directory as your main app
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.getcwd())

def test_direct_pipeline():
    """Test direct pipeline usage without LangChain"""
    print("Testing direct pipeline usage...")
    
    try:
        from transformers import pipeline, AutoTokenizer
        
        # Create a simple text generation pipeline
        print("Loading model...")
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1  # Use CPU
        )
        
        # Test generation
        prompt = "Hello, how are you?"
        print(f"Prompt: {prompt}")
        
        response = generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        # Remove original prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        print(f"Response: {generated_text}")
        print("✓ Direct pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Direct pipeline test failed: {e}")
        return False

def test_app_import():
    """Test importing the main app"""
    print("\nTesting app import...")
    
    try:
        # Check if the app file exists
        if not os.path.exists('app.py'):
            print("✗ app.py not found in current directory")
            return False
        
        # Try to import
        from app import LLMApplication
        
        print("✓ App import successful!")
        
        # Test basic initialization
        app = LLMApplication()
        print("✓ App initialization successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ App import failed: {e}")
        return False

def main():
    print("=== LLM Application Test Suite ===")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Run tests
    test1_passed = test_direct_pipeline()
    test2_passed = test_app_import()
    
    print(f"\n=== Test Results ===")
    print(f"Direct Pipeline: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"App Import: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Your setup is working correctly.")
        
        # Interactive test
        print("\nStarting interactive test...")
        try:
            from app import LLMApplication
            app = LLMApplication()
            
            print("Loading a small model for testing...")
            success = app.model_manager.load_huggingface_model("distilgpt2")
            
            if success:
                print("Model loaded! Testing generation...")
                response = app.generate_response("distilgpt2", "Hello, how are you?")
                print(f"AI Response: {response}")
            else:
                print("Failed to load model")
                
        except Exception as e:
            print(f"Interactive test failed: {e}")
    
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
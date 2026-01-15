print("Hello, World!")
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Torch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# Test if we can import the model
print("\nTesting model import...")
try:
    from model import DiT, CFM
    print("✓ Model import successful")
except Exception as e:
    print(f"✗ Model import failed: {e}")

# Test if we can import other modules
print("\nTesting other imports...")
try:
    from infer.infer_utils import prepare_model
    print("✓ infer_utils import successful")
except Exception as e:
    print(f"✗ infer_utils import failed: {e}")

# Test if we can read config file
print("\nTesting config file...")
try:
    import json
    with open("./config/diffrhythm-1b.json") as f:
        model_config = json.load(f)
    print(f"✓ Config file read successful, model type: {model_config.get('model', {}).get('type', 'unknown')}")
except Exception as e:
    print(f"✗ Config file read failed: {e}")

# Test if we can access Hugging Face Hub
print("\nTesting Hugging Face Hub...")
try:
    from huggingface_hub import hf_hub_download
    print("✓ Hugging Face Hub import successful")
except Exception as e:
    print(f"✗ Hugging Face Hub import failed: {e}")

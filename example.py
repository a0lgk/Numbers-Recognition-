"""
Example script demonstrating how to use the digit recognition model.
"""

from train import create_model, load_and_preprocess_data
from predict import predict_digit, predict_from_mnist, load_model
import os


def run_example():
    """Run a complete example of training and prediction."""
    
    print("=" * 60)
    print("DIGIT RECOGNITION EXAMPLE")
    print("=" * 60)
    
    # Check if model exists
    model_path = 'models/digit_recognition_model.h5'
    
    if not os.path.exists(model_path):
        print("\n[Step 1] Model not found. Training new model...")
        print("-" * 60)
        
        # Import and run training
        from train import main as train_main
        model, history = train_main()
        
        print("\n✓ Training completed!")
    else:
        print("\n[Step 1] Loading existing model...")
        print("-" * 60)
        model = load_model(model_path)
        print("✓ Model loaded successfully!")
    
    print("\n[Step 2] Testing on random MNIST samples...")
    print("-" * 60)
    predict_from_mnist(model, num_samples=10)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Try: python predict.py (to test on random samples)")
    print("2. Try: python predict.py your_image.png (to predict custom images)")
    print("3. Check training_history.png for training visualization")


if __name__ == '__main__':
    run_example()

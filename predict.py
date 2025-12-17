"""
Predict handwritten digits using the trained model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys


def load_model(model_path='models/digit_recognition_model.h5'):
    """
    Load the trained model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        keras.Model: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run train.py first."
        )
    
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(image_path):
    """
    Preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert colors if necessary (MNIST has white digits on black background)
    # If the image has black digits on white background, invert it
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model input
    img_array = np.expand_dims(img_array, axis=(0, -1))
    
    return img_array, img


def predict_digit(model, image_path, show_plot=True):
    """
    Predict the digit in an image.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        show_plot: Whether to display the prediction plot
        
    Returns:
        tuple: (predicted_digit, confidence)
    """
    # Preprocess image
    img_array, original_img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit] * 100
    
    # Display results
    print(f"\nPrediction for {image_path}:")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show all probabilities
    print("\nProbabilities for each digit:")
    for digit in range(10):
        prob = predictions[0][digit] * 100
        bar = '█' * int(prob / 2)
        print(f"{digit}: {prob:5.2f}% {bar}")
    
    if show_plot:
        # Plot image and predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Show image
        ax1.imshow(original_img, cmap='gray')
        ax1.set_title(f'Input Image\nPredicted: {predicted_digit} ({confidence:.1f}%)')
        ax1.axis('off')
        
        # Show prediction probabilities
        ax2.bar(range(10), predictions[0] * 100)
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Probability (%)')
        ax2.set_title('Prediction Probabilities')
        ax2.set_xticks(range(10))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return predicted_digit, confidence


def predict_from_mnist(model, num_samples=10):
    """
    Test the model on random samples from the MNIST test set.
    
    Args:
        model: Trained Keras model
        num_samples: Number of samples to test
    """
    # Load MNIST test set
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Randomly select samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # Create plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    correct_predictions = 0
    
    for i, idx in enumerate(indices):
        # Preprocess image
        img = x_test[idx]
        img_input = img.astype('float32') / 255.0
        img_input = np.expand_dims(img_input, axis=(0, -1))
        
        # Predict
        predictions = model.predict(img_input, verbose=0)
        predicted = np.argmax(predictions[0])
        true_label = y_test[idx]
        confidence = predictions[0][predicted] * 100
        
        # Check if correct
        is_correct = predicted == true_label
        if is_correct:
            correct_predictions += 1
        
        # Plot
        axes[i].imshow(img, cmap='gray')
        color = 'green' if is_correct else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {predicted}\n({confidence:.1f}%)', 
                         color=color, fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle(f'Random MNIST Predictions (Accuracy: {correct_predictions}/{num_samples})')
    plt.tight_layout()
    plt.show()
    
    print(f"\nAccuracy on random samples: {correct_predictions}/{num_samples} ({correct_predictions/num_samples*100:.1f}%)")


def main():
    """Main prediction function."""
    print("Loading trained model...")
    try:
        model = load_model()
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Check if image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            predict_digit(model, image_path)
        else:
            print(f"Error: Image file '{image_path}' not found.")
    else:
        # Test on random MNIST samples
        print("\nNo image provided. Testing on random MNIST samples...\n")
        predict_from_mnist(model, num_samples=10)


if __name__ == '__main__':
    main()

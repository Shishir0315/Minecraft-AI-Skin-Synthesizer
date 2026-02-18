import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Import the model creation functions from dcgan.py if possible, 
# otherwise we'll try to load the full model.
try:
    from dcgan import make_generator_model, noise_dim
except ImportError:
    noise_dim = 100

def generate_images(model_path='dcgan_generator.h5', checkpoint_dir='./training_checkpoints', num_images=16):
    generator = None
    
    # Try loading from full saved model first
    if os.path.exists(model_path):
        print(f"Loading full model from {model_path}...")
        generator = tf.keras.models.load_model(model_path)
    # If not found, try loading from checkpoints
    elif os.path.exists(checkpoint_dir) and tf.train.latest_checkpoint(checkpoint_dir):
        print(f"Loading from latest checkpoint in {checkpoint_dir}...")
        generator = make_generator_model()
        checkpoint = tf.train.Checkpoint(generator=generator)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    else:
        print("No model or checkpoint found. Please train the model first.")
        return

    # Generate noise
    noise = tf.random.normal([num_images, noise_dim])
    
    # Generate images
    generated_images = generator(noise, training=False)
    
    # Plot results
    cols = 4
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(cols*2, rows*2))
    for i in range(generated_images.shape[0]):
        plt.subplot(rows, cols, i+1)
        # Rescale to [0, 255]
        img = (generated_images[i, :, :, :] * 127.5 + 127.5).numpy().astype(np.uint8)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i+1}")
    
    output_name = 'generated_output.png'
    plt.tight_layout()
    plt.savefig(output_name)
    print(f"Generated images saved to {output_name}")
    plt.show()

if __name__ == "__main__":
    num = 16
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    generate_images(num_images=num)

# Minecraft Skin Face DCGAN

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate Minecraft skin faces based on the provided dataset.

## Project Structure
- `mc_skin_faces/`: The source dataset containing 317 images of Minecraft skin faces.
- `dcgan.py`: The main training script. It defines the Generator and Discriminator architectures, handles data preprocessing, and runs the training loop.
- `predict.py`: A script to load the trained generator and produce new images.
- `generated_images/`: Directory containing sample images generated during training (every 10 epochs).
- `training_checkpoints/`: Directory where model weights are saved periodically.

## Model Architecture
### Generator
- **Input**: 100-dimensional noise vector.
- **Layers**:
  - Dense layer (4x4x512)
  - 4 Transposed Convolution layers with Batch Normalization and LeakyReLU activation.
  - Final Transposed Convolution layer with Tanh activation to produce a 64x64x3 image.

### Discriminator
- **Input**: 64x64x3 image.
- **Layers**:
  - 3 Convolutional layers with LeakyReLU and Dropout.
  - Flatten and Dense layer to output a single scalar (Real/Fake).

## How to Run
1. **Training**:
   ```bash
   python dcgan.py
   ```
   This will train the model for 200 epochs and save checkpoints and sample images.

3. **Web Application**:
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your browser to interact with the premium face generator.

## Requirements
- TensorFlow 2.x
- Flask
- Flask-CORS
- PIL (Pillow)
- Matplotlib
- NumPy

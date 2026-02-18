# PixelForge AI: Minecraft AI Skin Synthesizer 🧊🤖

**PixelForge AI** is a cutting-edge Generative AI suite designed to revolutionize procedural asset creation for Minecraft. Utilizing a customized Deep Convolutional Generative Adversarial Network (DCGAN), the system synthesizes unique 64x64 pixel-art textures that capture the authentic aesthetic of the Minecraft universe.

## ✨ Key Features

- **Anti-Checkerboard DCGAN Architecture**: Optimized with UpSampling2D + Conv2D layers to produce sharp, artifact-free textures.
- **AI Command Center Dashboard**: A premium, glassmorphism-inspired web interface for real-time model monitoring and generation.
- **3D Kinematic Preview**: Built-in 3D engine that live-wraps AI textures onto a rotating Minecraft head for spatial feedback.
- **Deterministic Seed Engine**: Generate, share, and revisit unique identities using integer-based seeds.
- **Java/Bedrock Skin Exporter**: One-click utility to export high-quality 64x64 RGBA skin templates ready for in-game use.
- **Real-Time Analytics**: Integrated Chart.js dashboards tracking Generator and Discriminator stability.
- **Evolution Gallery**: Chronological history of training snapshots documenting the AI's artistic growth.

## 🛠️ Technology Stack

- **Backend**: Python, TensorFlow, Flask, Keras
- **Frontend**: Vanilla JavaScript (ES6+), Chart.js, HTML5/CSS3 (Glassmorphism design)
- **Imaging**: Pillow, Matplotlib, NumPy
- **Deployment**: Background threaded model loading for high-availability.

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the AI Training**:
   ```bash
   python dcgan.py
   ```

3. **Launch the Command Center**:
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` to start synthesizing.

## 📈 Model Architecture

### Generator
- **Noise Dim**: 100
- **Layers**: Reshape(4,4,1024) -> UpSampling2D -> Conv2D(512) -> BN -> LeakyReLU -> ... -> Conv2D(3, Tanh)
- **Output**: 64x64x3 RGB Image

### Discriminator
- **Input**: 64x64x3 RGB Image
- **Structural Strategy**: Multiple Conv2D layers with Dropout(0.3) for robust feature extraction and binary classification.

---
Developed by **Antigravity** across the Metaverse.
© 2026 Deepmind Advanced Agentic Coding

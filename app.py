import tensorflow as tf
import numpy as np
from flask import Flask, render_template, jsonify, send_file, request
from flask_cors import CORS
import io
from PIL import Image
import os
import threading
import time
import json
from dcgan import make_generator_model, noise_dim

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'dcgan_generator.keras'
CHECKPOINT_DIR = './training_checkpoints'

# Global generator and a lock for thread safety
generator = None
gen_lock = threading.Lock()
last_loaded_time = 0

def load_generator_async():
    global generator, last_loaded_time
    while True:
        try:
            if os.path.exists(MODEL_PATH):
                mtime = os.path.getmtime(MODEL_PATH)
                if mtime > last_loaded_time:
                    print(f"New model detected, loading {MODEL_PATH}...")
                    new_gen = tf.keras.models.load_model(MODEL_PATH)
                    with gen_lock:
                        generator = new_gen
                        last_loaded_time = mtime
                    print("Model updated successfully.")
            elif generator is None and os.path.exists(CHECKPOINT_DIR) and tf.train.latest_checkpoint(CHECKPOINT_DIR):
                print("Loading from checkpoint...")
                new_gen = make_generator_model()
                checkpoint = tf.train.Checkpoint(generator=new_gen)
                checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()
                with gen_lock:
                    generator = new_gen
                print("Checkpoint loaded.")
            elif generator is None:
                print("Initializing fresh model...")
                with gen_lock:
                    generator = make_generator_model()
        except Exception as e:
            print(f"Background load error: {e}")
        
        time.sleep(30) # Check for updates every 30 seconds

# Start background loader
threading.Thread(target=load_generator_async, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    if os.path.exists('loss_history.json'):
        with open('loss_history.json', 'r') as f:
            return f.read()
    return jsonify([])

@app.route('/generate')
def generate():
    seed = request.args.get('seed', type=int)
    download = request.args.get('download', type=bool, default=False)
    
    with gen_lock:
        if generator is None:
            img = Image.new('RGB', (256, 256), color = (30, 30, 30))
        else:
            try:
                if seed is not None:
                    tf.random.set_seed(seed)
                
                noise = tf.random.normal([1, noise_dim])
                generated_image = generator(noise, training=False)
                
                if tf.reduce_any(tf.math.is_nan(generated_image)):
                    img = Image.new('RGB', (256, 256), color = (255, 0, 0))
                else:
                    img_array = generated_image[0].numpy()
                    img_array = (img_array * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    img = img.resize((512, 512), resample=Image.NEAREST)
            except Exception as e:
                print(f"DEBUG: Generation error: {e}")
                img = Image.new('RGB', (256, 256), color = (0, 0, 255))
    
    if not os.path.exists('static'):
        os.makedirs('static')
        
    img.save('static/latest.png')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    if download:
        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name=f'minecraft_face_{seed or "random"}.png')
    
    return send_file(img_byte_arr, mimetype='image/png')

@app.route('/status')
def status():
    # Return some info about training progress if available
    epochs = 0
    if os.path.exists('output.log'):
        with open('output.log', 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'Epoch' in line:
                    try:
                        epochs = line.split('/')[0].split(' ')[1]
                        break
                    except:
                        pass
    return jsonify({"status": "ready", "latest_epoch": epochs})

@app.route('/gallery')
def gallery():
    files = []
    if os.path.exists('static/gallery'):
        files = [f for f in os.listdir('static/gallery') if f.endswith('.png')]
        # Sort by epoch number in filename
        files.sort()
    return jsonify(files)

@app.route('/export')
def export_skin():
    seed = request.args.get('seed', type=int)
    
    with gen_lock:
        if generator is None:
            return "Generator not ready", 400
        
        # Generate the face at 64x64 first (model output)
        if seed is not None:
            tf.random.set_seed(seed)
        noise = tf.random.normal([1, noise_dim])
        generated_image = generator(noise, training=False)
        img_array = (generated_image[0].numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        face_img = Image.fromarray(img_array).resize((8, 8), resample=Image.NEAREST)

    # Create a 64x64 base skin template (Steve layout)
    # We use a neutral base color for the body
    skin = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    
    # Simple "Steve" colors for base
    body_color = (40, 40, 40, 255) # Dark shirt base
    skin_tone = (255, 200, 150, 255) # Skin base
    
    # Draw basic skin structure (Simplified Steve layout)
    # Head Base (8x8x8 area on actual skin)
    # The face goes at (8, 8) in the 64x64 layout for standard Java skins
    skin.paste(skin_tone, [8, 0, 24, 8]) # Head top/bottom/sides
    skin.paste(face_img, (8, 8)) # Paste the AI face
    
    # Body (20, 20) to (32, 32)
    skin.paste((0, 150, 255, 255), [16, 20, 32, 32]) # Blue shirt
    skin.paste((80, 80, 80, 255), [16, 32, 28, 48]) # Legs
    
    # Save to buffer
    img_byte_arr = io.BytesIO()
    skin.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name=f'mc_skin_{seed or "random"}.png')

if __name__ == '__main__':
    # Ensure templates folder exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, host='0.0.0.0', port=5000)

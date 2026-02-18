import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from PIL import Image

# Hyperparameters
BUFFER_SIZE = 317
BATCH_SIZE = 16
EPOCHS = 500
noise_dim = 100
IMG_SIZE = 64
DATASET_PATH = r'c:\Users\student\Desktop\GAN model\mc_skin_faces'

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

def get_dataset():
    all_image_paths = [os.path.join(DATASET_PATH, fname) for fname in os.listdir(DATASET_PATH) if fname.endswith('.png')]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    dataset = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Enhanced data augmentation
    def augment(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image

    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).take(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def make_generator_model():
    model = tf.keras.Sequential([
        # Foundation for 4x4 image
        layers.Dense(4*4*1024, use_bias=False, input_shape=(noise_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((4, 4, 1024)),

        # Upscale to 8x8: Use UpSampling2D + Conv2D to avoid checkerboard artifacts
        layers.UpSampling2D(),
        layers.Conv2D(512, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upscale to 16x16
        layers.UpSampling2D(),
        layers.Conv2D(256, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upscale to 32x32
        layers.UpSampling2D(),
        layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upscale to 64x64
        layers.UpSampling2D(),
        layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Final Refinement
        layers.Conv2D(3, (3, 3), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # Label smoothing for real images
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
generator = make_generator_model()
discriminator = make_discriminator_model()

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train(dataset, epochs):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    
    # Variable to track epoch
    current_epoch = tf.Variable(0)
    
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator,
                                     epoch=current_epoch)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Restored from {latest_checkpoint}")
    else:
        print("No checkpoint found, starting from scratch.")

    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')

    start_epoch = int(current_epoch.numpy())
    print(f"Starting training from epoch {start_epoch}...", flush=True)
    for epoch in range(start_epoch, epochs):
        current_epoch.assign(epoch)
        start = time.time()
        
        gen_losses = []
        disc_losses = []

        for i, image_batch in enumerate(dataset):
            # Skip if batch size is not exactly BATCH_SIZE (for train_step function consistency)
            if image_batch.shape[0] != BATCH_SIZE:
                continue
            gl, dl = train_step(image_batch)
            gen_losses.append(gl)
            disc_losses.append(dl)
            if i % 5 == 0:
                print(f'.', end='', flush=True)

        # Produce images for the GIF as you go
        if (epoch + 1) % 10 == 0 or epoch == 0:
            generate_and_save_images(generator, epoch + 1)

        import json
        
        # Track average losses
        avg_gen_loss = float(np.mean(gen_losses))
        avg_disc_loss = float(np.mean(disc_losses))
        
        # Save to history file for web dashboard charts
        history_file = 'loss_history.json'
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except:
                pass
        
        history.append({
            "epoch": epoch + 1,
            "gen_loss": avg_gen_loss,
            "disc_loss": avg_disc_loss,
            "time": time.time()
        })
        
        with open(history_file, 'w') as f:
            json.dump(history, f)

        # Save the model every epoch for visibility in the web app
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            generator.save('dcgan_generator.keras')

        print(f'Epoch {epoch + 1}/{epochs} | Time: {time.time()-start:.2f}s | Gen Loss: {avg_gen_loss:.4f} | Disc Loss: {avg_disc_loss:.4f}', flush=True)

    # Final generation
    generate_and_save_images(generator, epochs)

def generate_and_save_images(model, epoch):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  seed = tf.random.normal([16, noise_dim])
  predictions = model(seed, training=False)

  fig = plt.figure(figsize=(8, 8))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5).numpy().astype(np.uint8))
      plt.axis('off')

  plt.savefig(f'generated_images/image_at_epoch_{epoch:04d}.png')
  
  # Web Gallery: Save every 10 epochs or first epoch
  if (epoch % 10 == 0) or (epoch == 1):
      if not os.path.exists('static/gallery'):
          os.makedirs('static/gallery')
      plt.savefig(f'static/gallery/gallery_epoch_{epoch:04d}.png', bbox_inches='tight', pad_inches=0.1, dpi=150)
      
  plt.close()

if __name__ == "__main__":
    train_ds = get_dataset()
    train(train_ds, EPOCHS)
    generator.save('dcgan_generator.h5')

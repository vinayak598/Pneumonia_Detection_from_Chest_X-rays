import kagglehub
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Download dataset using kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("✅ Dataset downloaded to:", path)

# Step 2: Set correct dataset directory (KaggleHub extracts automatically)
base_dir = os.path.join(path, "chest_xray")
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Step 3: Verify directory structure
for label, d in [("Training", train_dir), ("Validation", val_dir), ("Test", test_dir)]:
    if not os.path.exists(d):
        print(f"❌ {label} directory not found at {d}")
    else:
        print(f"✅ {label} directory found at {d}")

# Step 4: Proceed only if all directories exist
if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
    val_gen = datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
    test_gen = datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

    # Step 5: Build CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Step 6: Train model
    print("🚀 Starting model training...")
    model.fit(train_gen, epochs=5, validation_data=val_gen)
    print("✅ Model training completed.")

    # Step 7: Save model
    save_folder = r'C:\Users\saiva\Desktop\New folder'
    os.makedirs(save_folder, exist_ok=True)

    model_save_path = os.path.join(save_folder, 'pneumonia_cnn_model.h5')
    model.save(model_save_path)
    print(f"💾 Model saved to: {model_save_path}")

else:
    print("❌ One or more required directories (train, val, test) are missing. Cannot proceed.")

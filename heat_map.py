import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "BreaKHis_v1/histology_slides/breast"  # Adjust to your base directory
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

file_paths = []
labels = []
patient_ids = []

# Walk through all image files inside nested folders
for filepath in glob.glob(os.path.join(DATA_DIR, "**/*.png"), recursive=True):
    parts = os.path.normpath(filepath).split(os.sep)
    try:
        sob_index = parts.index("SOB")
        class_label = parts[sob_index + 1]
        patient_id = parts[sob_index + 2]
        
        file_paths.append(filepath)
        labels.append(class_label)
        patient_ids.append(patient_id)
    except (ValueError, IndexError):
        continue

df = pd.DataFrame({
    'filepath': file_paths,
    'label': labels,
    'patient_id': patient_ids
})

print(f"Total images found: {len(df)}")
print(f"Classes found ({len(df['label'].unique())}): {df['label'].unique()}")

# Encode categorical labels to integers
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)

# ==========================================
# 2. PATIENT-LEVEL TRAIN / VALIDATION SPLIT
# ==========================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df['patient_id']))

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

# ==========================================
# 3. TF.DATA PIPELINE CREATION
# ==========================================
def parse_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

def build_dataset(dataframe, is_training=True):
    paths = dataframe['filepath'].values
    labels = dataframe['label_encoded'].values
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_ds = build_dataset(train_df, is_training=True)
val_ds = build_dataset(val_df, is_training=False)

# ==========================================
# 4. HEATMAP / SPATIAL ATTENTION MODEL
# ==========================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)

# Feature extraction backbone
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)

# Spatial Attention / Heatmap Subnetwork
# Generates a 1-channel spatial attention map (heatmap) with values in [0, 1]
heatmap = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='spatial_heatmap')(x)

# Weight feature maps using the learned heatmap
weighted_features = tf.keras.layers.Multiply()([x, heatmap])

# Global pooling on the attention-weighted features
x = tf.keras.layers.GlobalAveragePooling2D()(weighted_features)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 5. TRAIN MODEL
# ==========================================
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Save in modern .keras format
model.save("breakhis_heatmap_classifier.keras")
print("Heatmap model training complete and saved.")
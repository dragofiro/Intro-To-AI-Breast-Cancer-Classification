lazy import os
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
    # Target label is the subfolder right under SOB (e.g., 'adenosis', 'ductal_carcinoma')
    # Parts structure: [..., 'SOB', '<class_name>', '<patient_id>', '<magnification>', '<filename>']
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
    # EfficientNet handles internal scaling, but standard models use [0, 255] or [-1, 1]
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
# 4. TRANSFER LEARNING MODEL (EfficientNetB0)
# ==========================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False  # Freeze pretrained weights initially

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 5. TRAIN & FINE-TUNE
# ==========================================
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Optional: Fine-tuning step
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Low LR for fine-tuning
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

fine_tune_epochs = 5
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

# Save in modern .keras format
model.save("breakhis_classifier.keras")
print("Model training complete and saved.")
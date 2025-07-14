"""
Enhanced MobileNetV3Small Training for Mechanical Part Classification
Now with: Aug++, deeper fine-tune, label smoothing, LR decay, partial unfreeze
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ───── CONFIG ─────
DATA_DIR = Path("datasets/mech_parts")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 15
MODEL_DIR = Path("checkpoints/mobilenetv3_enhanced")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = sorted([p.name for p in (DATA_DIR / "train").iterdir()])
NUM_CLASSES = len(CLASS_NAMES)
print("🧾 Class Names:", CLASS_NAMES)

# ───── DATA LOAD + AUGMENTATION ─────
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# ⚡ Fancy Augmentation
aug_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="data_aug")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(aug_layer(x)), y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)

# ───── MODEL BUILD ─────
base_model = MobileNetV3Small(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
    dropout_rate=0.2
)
base_model.trainable = False  # Will unfreeze later

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()

# ───── OPTIMIZER + SCHEDULER + LOSS ─────
initial_lr = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=5000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# ───── CALLBACKS ─────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "head_only.keras"),  # ✅ FIXED
        save_best_only=True,
        monitor="val_accuracy",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
]

# ───── TRAIN: TOP LAYERS ONLY ─────
print("\n🎯 Training classifier head...")
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks
)

# ───── UNFREEZE 50% OF BASE MODEL ─────
print("\n🔓 Fine-tuning deeper layers...")
base_model.trainable = True
for layer in base_model.layers[:int(len(base_model.layers) * 0.5)]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

callbacks[0].filepath = str(MODEL_DIR / "finetuned.keras")  # ✅ FIXED

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

# ───── EVALUATE ─────
print("\n🧪 Evaluating on test set...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
).map(lambda x, y: (preprocess_input(x), y)).prefetch(AUTOTUNE)

loss, acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {acc:.2%}")

# ───── SAVE MODEL + LABELS ─────
model.save(MODEL_DIR / "mobilenetv3_enhanced.keras")
with open(MODEL_DIR / "labels.txt", "w") as f:
    f.write("\n".join(CLASS_NAMES))

print("🎉 Done! Enhanced model saved in:", MODEL_DIR)

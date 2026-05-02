import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import time

PLOTS_DIR = Path('plots')
CHECKPOINT_DIR = Path('checkpoints')
PLOTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
MODEL_PATH = 'food_classifier.h5'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 101


def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.keras.layers.RandomRotation(0.1)(image, training=True)
    image = tf.keras.layers.RandomZoom(0.1)(image, training=True)
    return image, label


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze all pretrained layers — transfer learning stage 1
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_latest_checkpoint():
    checkpoint_dir = Path('checkpoints')
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob('epoch_*.h5'))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    print(f'Resuming from checkpoint: {latest}')
    return tf.keras.models.load_model(latest)


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', color='steelblue')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='tomato')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='steelblue')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='tomato')
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '01_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {PLOTS_DIR / "01_training_curves.png"}')


def plot_sample_predictions(model, val_dataset, class_names):
    # Grab one batch and pick 9 samples
    images_batch, labels_batch = next(iter(val_dataset.take(1)))
    images_np = images_batch.numpy()[:9]
    labels_np = labels_batch.numpy()[:9]

    predictions = model.predict(images_np)
    pred_indices = predictions.argmax(axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images_np[i])
        true_label = class_names[labels_np[i]].replace('_', ' ').title()
        pred_label = class_names[pred_indices[i]].replace('_', ' ').title()
        confidence = predictions[i][pred_indices[i]] * 100
        color = 'green' if pred_indices[i] == labels_np[i] else 'red'
        ax.set_title(
            f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
            color=color, fontsize=9
        )
        ax.axis('off')

    plt.suptitle('Sample Predictions (green = correct, red = incorrect)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '02_sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {PLOTS_DIR / "02_sample_predictions.png"}')


def plot_confusion_matrix(model, val_dataset, class_names):
    all_true = []
    all_pred = []

    # Collect predictions across the full validation set
    for images_batch, labels_batch in val_dataset:
        preds = model.predict(images_batch, verbose=0)
        all_pred.extend(preds.argmax(axis=1))
        all_true.extend(labels_batch.numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    cm = confusion_matrix(all_true, all_pred)

    # Find the top 20 most confused classes (highest off-diagonal sum)
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    row_confusion = off_diag.sum(axis=1)
    top20_indices = row_confusion.argsort()[-20:][::-1]

    cm_subset = cm[np.ix_(top20_indices, top20_indices)]
    subset_labels = [class_names[i].replace('_', ' ').title() for i in top20_indices]

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        cm_subset, annot=True, fmt='d', cmap='Blues',
        xticklabels=subset_labels, yticklabels=subset_labels,
        linewidths=0.5
    )
    plt.title('Confusion Matrix — Top 20 Most Confused Classes', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '03_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {PLOTS_DIR / "03_confusion_matrix.png"}')


def main():
    start_time = time.time()
    print('Loading Food-101 dataset...')

    (train_ds_raw, val_ds_raw), info = tfds.load(
        'food101',
        split=['train', 'validation'],
        as_supervised=True,
        with_info=True
    )

    class_names = info.features['label'].names

    # Preprocess both splits
    train_dataset = (
        train_ds_raw
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .map(lambda x, y: (tf.clip_by_value(
            x + tf.random.uniform(tf.shape(x), -0.05, 0.05), 0.0, 1.0), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        val_ds_raw
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Resume from checkpoint if available, otherwise build fresh
    model = load_latest_checkpoint()
    if model is None:
        print('Building model from scratch...')
        model = build_model()

    initial_epoch = len(list(Path('checkpoints').glob('epoch_*.h5')))
    print(f'Starting training from epoch {initial_epoch + 1}')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/epoch_{epoch:02d}.h5',
        save_weights_only=False,
        save_best_only=False,
        verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    model.save(MODEL_PATH)
    print(f'\nModel saved to {MODEL_PATH}')

    print('\nGenerating visualisations...')
    plot_training_curves(history)
    plot_sample_predictions(model, val_dataset, class_names)
    plot_confusion_matrix(model, val_dataset, class_names)

    elapsed = time.time() - start_time
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print('\n' + '=' * 50)
    print('TRAINING SUMMARY')
    print('=' * 50)
    print(f'Final training accuracy : {final_train_acc:.4f} ({final_train_acc*100:.2f}%)')
    print(f'Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)')
    print(f'Model saved             : {MODEL_PATH}')
    print(f'Total training time     : {elapsed/60:.1f} minutes')
    print('=' * 50)


if __name__ == '__main__':
    main()

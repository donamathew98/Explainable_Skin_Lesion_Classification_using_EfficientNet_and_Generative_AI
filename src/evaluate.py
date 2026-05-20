# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt



# # Path
# MODEL_PATH = "models/model.keras"
# DATA_DIR = "data/images"

# IMG_SIZE = 224
# BATCH_SIZE = 16

# # Load model 
# model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# print("Model loaded successfully!")

# # Data generator
# val_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     validation_split=0.2
# )

# val_data = val_datagen.flow_from_directory(
#     DATA_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation',
#     shuffle=False   # IMPORTANT
# )

# print("Evaluating model...")

# # Predictions
# predictions = model.predict(val_data)
# y_pred = np.argmax(predictions, axis=1)

# # True labels
# y_true = val_data.classes

# # Class labels
# class_labels = list(val_data.class_indices.keys())

# # Classification report
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=class_labels))

# # Confusion matrix
# print("\nConfusion Matrix:")
# cm = confusion_matrix(y_true, y_pred)
# print(cm)

# cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# plt.figure(figsize=(8,6))
# plt.imshow(cm_norm, interpolation='nearest')
# plt.title("Normalized Confusion Matrix")
# plt.colorbar()

# tick_marks = np.arange(len(class_labels))

# plt.xticks(tick_marks, class_labels, rotation=45)
# plt.yticks(tick_marks, class_labels)

# for i in range(cm_norm.shape[0]):
#     for j in range(cm_norm.shape[1]):
#         plt.text(j, i, f"{cm_norm[i, j]:.2f}",
#                  ha="center")

# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.tight_layout()

# plt.savefig("confusion_matrix.png", dpi=300)
# plt.show()
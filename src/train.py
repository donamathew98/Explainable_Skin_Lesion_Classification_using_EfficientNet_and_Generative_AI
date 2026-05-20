# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras import layers, models
# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight

 
# # Paths
 
# DATA_DIR = "data/images"

 
# # Image settings
 
# IMG_SIZE = 224   
# BATCH_SIZE = 16

 
# # Data Generators
 
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     validation_split=0.2,
#     rotation_range=20,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1
# )



# val_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     validation_split=0.2
# )

# train_data = train_datagen.flow_from_directory(
#     DATA_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# val_data = val_datagen.flow_from_directory(
#     DATA_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )


# # Class Weights (for imbalance)
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_data.classes),
#     y=train_data.classes
# )

# class_weights_dict = dict(enumerate(class_weights))
# print("Class Weights:", class_weights_dict)

 
# # EfficientNet Model
# base_model = EfficientNetB0(
#     weights='imagenet',
#     include_top=False,
#     input_shape=(224, 224, 3) 
# )

# # Freeze base model (Phase 1)
# base_model.trainable = False

# # Custom classifier
# x = base_model.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dropout(0.5)(x)

# output = layers.Dense(7, activation='softmax')(x)

# model = models.Model(inputs=base_model.input, outputs=output)

 
# # Compile (Phase 1)
 
# model.compile(
#     optimizer=Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=2,
#     restore_best_weights=True
# )

# print(type(class_weights))
# print(type(class_weights_dict))

 
# # Phase 1 Training (Feature Extraction)
 
# print("\nStarting Phase 1 Training...\n")

# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=10,
#     class_weight=class_weights_dict,
#     callbacks=[early_stop]
# )

# # Fine-Tuning Phase

# print("\nStarting Fine-Tuning...")

# for layer in base_model.layers[:-50]:
#     layer.trainable = False

# for layer in base_model.layers[-50:]:
#     layer.trainable = True

# import tensorflow.keras.backend as K

# def focal_loss(gamma=2., alpha=0.25):
#     def loss(y_true, y_pred):
#         y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
#         cross_entropy = -y_true * K.log(y_pred)
#         weight = alpha * K.pow(1 - y_pred, gamma)
#         return K.sum(weight * cross_entropy, axis=1)
#     return loss

# # Recompile
# model.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss=focal_loss(),
#     metrics=['accuracy']
# )

# # Callbacks
# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=3,
#     restore_best_weights=True
# )

# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.3,
#     patience=2,
#     min_lr=1e-7
# )


# # Train (NO class_weight here)
# history_fine = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=10,
#     callbacks=[early_stop, reduce_lr]
# )
 
# # Save model
 
# model.save("models/model.keras")

# print("Model training complete and saved!")


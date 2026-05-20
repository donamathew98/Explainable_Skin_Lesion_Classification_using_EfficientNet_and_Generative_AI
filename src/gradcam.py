import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "models/model.keras"
IMG_SIZE = 224

def generate_gradcam(img_path: str, output_path: str):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded!")

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    last_conv_layer_name = "top_conv"
    print("Last Conv Layer:", last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) > 0:
        heatmap /= tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_img = heatmap_color * 0.4 + img_bgr
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap_color)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Grad-CAM")
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Grad-CAM to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/gradcam.py <input_image_path> <output_image_path>")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2]
    generate_gradcam(input_image, output_image)
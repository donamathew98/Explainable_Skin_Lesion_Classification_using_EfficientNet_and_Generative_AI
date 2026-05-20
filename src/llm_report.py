# import os
# import sys
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.efficientnet import preprocess_input

# MODEL_PATH = "models/model.keras"
# IMG_SIZE = 224
# class_labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# CLASS_INFO = {
#     "akiec": {
#         "name": "Actinic keratosis / intraepithelial carcinoma",
#         "summary": "This class represents potentially pre-cancerous skin lesions that may require medical review."
#     },
#     "bcc": {
#         "name": "Basal cell carcinoma",
#         "summary": "This is a common type of skin cancer that usually grows slowly but still requires treatment."
#     },
#     "bkl": {
#         "name": "Benign keratosis-like lesion",
#         "summary": "This class usually includes non-cancerous skin lesions, though clinical confirmation is still important."
#     },
#     "df": {
#         "name": "Dermatofibroma",
#         "summary": "This is generally a benign skin lesion, often appearing as a firm bump."
#     },
#     "mel": {
#         "name": "Melanoma",
#         "summary": "This is a serious type of skin cancer that can spread if not detected early."
#     },
#     "nv": {
#         "name": "Melanocytic nevus",
#         "summary": "This is commonly known as a mole and is usually benign."
#     },
#     "vasc": {
#         "name": "Vascular lesion",
#         "summary": "This class includes lesions related to blood vessels and is often benign."
#     }
# }

# def confidence_phrase(score: float) -> str:
#     pct = score * 100
#     if pct >= 85:
#         return "very high confidence"
#     elif pct >= 70:
#         return "high confidence"
#     elif pct >= 50:
#         return "moderate confidence"
#     else:
#         return "low confidence"

# def generate_report(pred_label: str, confidence: float, top3: list, gradcam_note: str) -> str:
#     label_info = CLASS_INFO[pred_label]
#     conf_text = confidence_phrase(confidence)
#     conf_pct = confidence * 100
#     alternatives = ", ".join([f"{label} ({score*100:.1f}%)" for label, score in top3[1:]])

#     return f"""AI-assisted lesion explanation

# Predicted class:
# {label_info['name']} ({pred_label})

# Confidence:
# The model predicts this class with {conf_text} ({conf_pct:.1f}%).

# Clinical meaning:
# {label_info['summary']}

# Visual explanation:
# {gradcam_note}

# Alternative model considerations:
# Other possible classes considered by the model were {alternatives}.

# Recommendation:
# This output should be treated as decision support only and not as a definitive diagnosis. Clinical review by a dermatologist is recommended, especially if the lesion is changing in size, color, or border characteristics.
# """.strip()

# def main(img_path: str, output_path: str):
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     print("Model loaded successfully!")

#     img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     predictions = model.predict(img_array)
#     pred_idx = int(np.argmax(predictions[0]))
#     confidence = float(np.max(predictions[0]))
#     pred_label = class_labels[pred_idx]

#     top3_idx = np.argsort(predictions[0])[::-1][:3]
#     top3 = [(class_labels[i], float(predictions[0][i])) for i in top3_idx]

#     gradcam_note = (
#         "The model focused mainly on the central lesion region, particularly darker pigmentation "
#         "and irregular internal structure."
#     )

#     report = generate_report(pred_label, confidence, top3, gradcam_note)

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(report)

#     print(report)
#     print(f"\nSaved report to: {output_path}")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python src/llm_report.py <input_image_path> <output_report_path>")
#         sys.exit(1)

#     input_image = sys.argv[1]
#     output_report = sys.argv[2]
#     main(input_image, output_report)
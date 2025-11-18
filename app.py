import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

# Load Model and Classes
model = load_model("brain_tumor_cnn_model.h5")
class_names = np.load("class_labels.npy")

IMG_SIZE = 224  # same as training

# ------------------------------------------------------
# GRAD-CAM FUNCTION
# ------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[:, pred_index]

    grads = tape.gradient(pred_output, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_output = conv_outputs[0]

    heatmap = np.zeros(conv_output.shape[:2])

    for i in range(pooled_grads.shape[-1]):
        heatmap += pooled_grads[i] * conv_output[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-9)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# STREAMLIT UI

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify the tumor type using CNN & Grad-CAM visualization.")

uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show Image
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Preprocessing
    img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_batch)[0]
    class_index = np.argmax(pred)
    prediction_name = class_names[class_index]
    confidence = pred[class_index]

    st.subheader("🔍 Prediction")
    st.write(f"**Class:** {prediction_name}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Grad-CAM
    st.subheader("🔥 Grad-CAM Heatmap")

    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer:
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer)
        superimposed_image = cv2.addWeighted(heatmap, 0.4, (img_array * 255).astype("uint8"), 0.6, 0)

        st.image(superimposed_image, caption="Grad-CAM", use_column_width=True)
    else:
        st.error("No Conv2D layers found in model for Grad-CAM.")
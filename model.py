import os, torch
import numpy as np
import onnxruntime as ort
import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_path = "models/mnist-8.onnx"


# load mnist-onnx model
@st.cache_resource
def get_onnx_model():
    if not os.path.exists(model_path):
        print(f"모델 파일이 없습니다: {model_path}")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"] # streamlit RAM(1~2GB)
    )
    return session


def predict_digit(image_28x28: np.ndarray):
    """
    image_28x28:
        shape = (28, 28)
        dtype = float32
        value range = [0, 1]
    """
    session = get_onnx_model()

    # ONNX 입력 형태: (batch, channel, height, width)
    input_tensor = image_28x28.astype(np.float32)  # (28, 28)
    input_tensor = input_tensor.reshape(1, 1, 28, 28)  # 1장, gray, -> (1, 1, 28, 28)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run(
        [output_name],
        {input_name: input_tensor}
    )[0]
    '''score = [
        array([[ -1.2, 3.4, 0.8, ... ]])  # logits 10개
        ]'''

    # softmax
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    predicted_label = int(np.argmax(probs)) # 큰 값의 인덱스
    probabilities = probs.flatten()

    return predicted_label, probabilities



# load vit model
@st.cache_resource
def get_vit_model():
    # Load model directly
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.eval()
    return processor, model


def predict_vit(img, processor, model):
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    id2label = model.config.id2label

    results = [
        {"label": id2label[i], "score": float(probs[i])}
        for i in range(len(probs))
    ]

    results.sort(key=lambda x: x["score"], reverse=True)

    return results
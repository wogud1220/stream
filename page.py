import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from model import predict_digit, predict_vit, get_vit_model

st.set_page_config(page_title="숫자, 그림 인식", layout="centered")
st.spinner()

with st.sidebar:
    model_type = st.selectbox("select model", options = ["Mnist-onnx", "hf-google/vit-base-patch16-224"])


STROKE_WIDTH = 18
STROKE_COLOR = "#FFFFFF"
BG_COLOR = "#000000"
CANVAS_SIZE = 280


def canvas_to_mnist(img_data: np.ndarray) -> np.ndarray:
    # RGBA → RGB
    img = Image.fromarray(img_data.astype("uint8"), "RGBA").convert("RGB")

    # Grayscale
    img = ImageOps.grayscale(img)

    # Resize to 28x28
    img_28 = img.resize((28, 28), Image.Resampling.BILINEAR)

    # Normalize [0,1]
    arr = np.array(img_28).astype(np.float32) / 255.0
    return arr  # (28, 28)



def call_mnist_ui():
    st.title("숫자를 아래에 그리세요")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("나의 입력")
        canvas_result = st_canvas(
            stroke_width=STROKE_WIDTH,
            stroke_color=STROKE_COLOR,
            background_color=BG_COLOR,
            update_streamlit=True,
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="digit_canvas",
        )

    predicted_label = None
    prob_dict = None

    with col2:
        st.subheader("모델 입력")

        if canvas_result.image_data is not None:
            mnist_img = canvas_to_mnist(canvas_result.image_data)

            st.image(mnist_img, width="stretch")
            st.caption(f"shape: {mnist_img.shape}")

            predicted_label, probabilities = predict_digit(mnist_img)
            prob_dict = {str(i): float(probabilities[i]) for i in range(10)}
        else:
            st.info("왼쪽에 숫자를 그려주세요 ✏️")

    if predicted_label is not None and prob_dict is not None:
        st.markdown("---")
        st.subheader("📊 예측 결과")

        st.metric("예측 숫자", predicted_label)

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            x=list(prob_dict.keys()),
            y=list(prob_dict.values()),
            ax=ax
        )

        # 막대 위에 퍼센트 표시
        for i, v in enumerate(prob_dict.values()):
            plt.text(
                i,  # x 위치 (막대 index)
                v + 0.01,  # y 위치 (막대 위 살짝)
                f"{v * 100:.1f}%",  # 표시할 텍스트
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.grid(axis="y") # grid
        st.pyplot(plt)


def set_source_upload():
    st.session_state["source"] = "upload"

def set_source_camera():
    st.session_state["source"] = "camera"


def call_vit_ui():
    col1, col2 = st.columns(2)
    with st.sidebar:
        top_k = st.slider(
            "Top-K 개수",
            min_value=1,
            max_value=10,
            value=5)
    with col1:
        st.file_uploader(
            "이미지를 업로드하세요.(png, jpg, jpeg)",
            type=["png", "jpg", "jpeg"],
            key = "uploaded_img",
            accept_multiple_files=True,
            on_change=set_source_upload
        ) #
    with col2:
        st.camera_input(label="camera input",key = "camera_img", on_change=set_source_camera)

    image_source = None

    if st.session_state.get("source") == "upload":
        image_source = st.session_state.get("uploaded_img")

    elif st.session_state.get("source") == "camera":
        image_source = st.session_state.get("camera_img")

    if image_source is not None:
        print_result(image_source, top_k)


def print_result(uploaded, top_k):
    if uploaded is None:
        st.info("이미지를 업로드해주세요 📤")
        return

    if isinstance(uploaded, list) and len(uploaded) == 0:
        return

    st.success("업로드 성공")
    proc, model = get_vit_model()

    if not isinstance(uploaded, list):
        uploaded = [uploaded]

    images = [Image.open(f).convert("RGB") for f in uploaded]

    batch_results = predict_vit(
        images=images,
        processor=proc,
        model=model,
        top_k=top_k
    )

    for img, results in zip(images, batch_results):
        top1 = results[0]
        label = top1["label"]
        score = top1["score"]

        st.markdown("---")
        st.image(img, width="stretch")

        st.subheader("🔍 분류 결과")
        st.success(f"🎯 예측 결과: **{label}**")
        st.metric("신뢰도", f"{score * 100:.1f}%")
        st.progress(score)

        if score >= 0.8:
            st.success("✅ 모델이 매우 확신합니다.")
        elif score >= 0.5:
            st.warning("🤔 어느 정도 유사하지만 확신은 낮습니다.")
        else:
            st.info("❓ 모델이 확신하지 못했습니다.")

        labels = [r["label"] for r in results]
        scores = [r["score"] * 100 for r in results]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels, scores)

        ax.invert_yaxis()
        ax.set_xlabel("Confidence (%)")
        ax.set_title(f"Top-{top_k} Prediction Results")

        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 10))

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center"
            )

        ax.grid(axis="x", linestyle="--", alpha=0.6)
        st.pyplot(fig)


if model_type == "Mnist-onnx":
    call_mnist_ui()

else:
    call_vit_ui()






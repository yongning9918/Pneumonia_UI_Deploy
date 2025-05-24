import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from torchvision import models

# 页面标题
st.title("Pneumonia Detection Image Classifier")
st.write("Upload a Chest X-ray image OR provide an image URL")

# 类别标签
class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# 加载模型
@st.cache_resource
def load_model():
    model = models.vgg16_bn(pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 198),
        torch.nn.ReLU(),
        torch.nn.Linear(198, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 4)
    )
    model.load_state_dict(torch.load("best_model_stage2.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

# 图像预处理
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

# 预测函数
def predict(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# 加载模型一次
model = load_model()

# 上传图片 or 输入 URL（两选一）
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or enter an image URL...")

image = None

# 处理本地上传图片
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")

# 或者处理 URL 图片
elif image_url:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(image_url, headers=headers)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Error fetching image: {e}")

# 如果成功读取图片，进行预测
if image is not None:
    st.image(image, caption="Chest X-ray", use_container_width=True)  # ✅ 修改此处消除警告
    label = predict(image)
    st.success(f"**Predicted Class:** {label}")

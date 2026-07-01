import base64
import io
import os

from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI


# Read OpenRouter API key from environment variable
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing environment variable OPENROUTER_API_KEY")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=120.0,
    max_retries=1,
)


app = FastAPI()


def image_to_base64_jpeg(image_bytes: bytes, max_size: int = 1024) -> str:
    """
    将上传的图片转成 JPEG base64，方便发送给 OpenRouter 多模态模型。
    """
    image = Image.open(io.BytesIO(image_bytes))

    # 处理透明通道图片，例如 PNG
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # 等比例缩放，避免图片过大导致请求慢
    image.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>多模态图片分析 Demo</title>
</head>
<body>
    <h2>多模态图片分析 Demo</h2>

    <input type="file" id="file" accept="image/*">

    <p>Prompt：</p>
    <textarea id="prompt" rows="4" cols="80">请使用简体中文回答。请详细描述这张图片中的主要内容、物体、颜色和场景。</textarea>

    <br><br>
    <button onclick="analyze()">分析图片</button>

    <h3>预览：</h3>
    <img id="preview" style="max-width: 600px; display: block;">

    <h3>模型输出：</h3>
    <pre id="result" style="white-space: pre-wrap; background: #f5f5f5; padding: 12px;"></pre>

    <script>
        async function analyze() {
            const fileInput = document.getElementById("file");
            const prompt = document.getElementById("prompt").value;
            const result = document.getElementById("result");
            const preview = document.getElementById("preview");

            if (!fileInput.files.length) {
                alert("请先选择图片");
                return;
            }

            const file = fileInput.files[0];
            preview.src = URL.createObjectURL(file);

            const formData = new FormData();
            formData.append("file", file);
            formData.append("prompt", prompt);

            result.innerText = "分析中，请稍等...";

            try {
                const response = await fetch("/analyze", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    result.innerText = "请求失败：" + JSON.stringify(data);
                    return;
                }

                result.innerText = data.result;
            } catch (error) {
                result.innerText = "请求异常：" + error;
            }
        }
    </script>
</body>
</html>
"""


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form("请使用简体中文回答。请详细描述这张图片。")
):
    image_bytes = await file.read()
    image_b64 = image_to_base64_jpeg(image_bytes)

    response = client.chat.completions.create(
        model="google/gemma-4-26b-a4b-it:free",
        messages=[
            {
                "role": "system",
                "content": "你是一个中文图像分析助手。无论用户输入什么语言，你都必须使用简体中文回答。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64
                        }
                    }
                ]
            }
        ],
        max_tokens=800
    )

    return {
        "result": response.choices[0].message.content
    }

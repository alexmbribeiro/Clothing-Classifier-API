---

# **Clothing Classifier API** 👕👗

A PyTorch-based clothing image classifier served via **FastAPI** and **Gradio**, capable of predicting Fashion-MNIST categories. This project demonstrates model training, deployment, and web-based interaction.

[![Model](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/alexrmb/fashion-classifier-model)
[![Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://huggingface.co/spaces/alexrmb/Clothing-Classifier-API)
---

## **🚀 Features**

* Fine-tuned **ResNet** for Fashion-MNIST clothing classification
* **FastAPI** endpoint for REST-based predictions (`/predict`)
* **Gradio** web interface for easy image uploads and predictions
* **Dockerized** for local or cloud deployment
* Model hosted via **Hugging Face Hub** for lightweight deployment

---

## **🛠 Installation / Local Setup**

1. Clone the repo:

```bash
git clone https://github.com/<username>/clothing-classifier-api.git
cd clothing-classifier-api
```

2. Create install dependencies:

```bash
pip install -r requirements.txt
```

3. Run locally:

```bash
python app/main.py
```

* **Gradio UI:** [http://127.0.0.1:7860](http://127.0.0.1:7860)
* **FastAPI docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## **🐳 Docker Deployment**

1. Build Docker image:

```bash
docker build -t clothing-classifier .
```

2. Run container:

```bash
docker run -p 8000:8000 clothing-classifier
```

3. Access the same FastAPI endpoints locally.

---

## **🖼 Usage**

* **Gradio Interface:** Upload any clothing image → instant prediction
* **FastAPI Endpoint:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path/to/image.jpg"
```

* Returns JSON with predicted class label:

```json
{
  "class": "T-shirt/top"
}
```

---

## **📦 Model**

* Pretrained on Fashion-MNIST
* Hosted on Hugging Face Hub: [![Model](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/alexrmb/fashion-classifier-model)
* Loaded dynamically at runtime, no large files stored in repo

---

## **💻 Technologies Used**

* **PyTorch** – deep learning model
* **FastAPI** – REST API backend
* **Gradio** – web interface for demos
* **Docker** – containerized deployment
* **Hugging Face Hub** – hosting pretrained model

---

## **📝 License**

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## **📌 Demo / Links**

* **Hugging Face Spaces Demo:** [![Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://huggingface.co/spaces/alexrmb/Clothing-Classifier-API)

---

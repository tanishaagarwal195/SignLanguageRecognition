## **Sign Language Recognition for Emergency Communication**  
🚀 **Real-time AI-powered sign language recognition system for emergency communication using deep learning.**  

---

### 📌 **Project Overview**  
This project aims to assist the **deaf and hard-of-hearing community** by recognizing **emergency sign language gestures** using **computer vision and deep learning**. The system captures hand gestures via a **webcam**, classifies them using a **CNN-based MobileNetV2 model**, and provides an **audio alert** for communication with emergency services.

### ✅ **Key Features**  
✔ **Real-time sign language recognition** using OpenCV & PyTorch  
✔ **Custom dataset** of 8 emergency signs  
✔ **Deep learning model (MobileNetV2)** trained with transfer learning  
✔ **Temporal analysis (majority voting over frames)** for better accuracy  
✔ **Text-to-speech (TTS)** for immediate audio alerts  
✔ **Future scope: Mobile app deployment with TensorFlow Lite**  

---

## 📂 **Dataset & Classes**  
This project uses **custom sign language videos**, converted into frames and labeled into **8 emergency-related classes**:  

🔴 **Danger**  
🆘 **Emergency**  
🔥 **Fire**  
🤝 **Help**  
🙏 **Please**  
🚔 **Police**  
✋ **Stop**  
💧 **Water**  

Each class contains **500 images**, balanced through **data augmentation**.

---

## 🛠 **Technologies Used**  
- **Python** (Core Programming)  
- **OpenCV** (Computer Vision & Webcam Integration)  
- **PyTorch & Torchvision** (Deep Learning)  
- **Mediapipe** (Hand Tracking & ROI Detection)  
- **Pyttsx3 & gTTS** (Text-to-Speech for Audio Alerts)  
- **Google Colab / Jupyter Notebook** (Model Training)  

---

## 🚀 **Installation & Setup**  
### 🔹 **1. Clone the Repository**  
```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```

### 🔹 **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 🔹 **3. Run the Webcam-Based Gesture Recognition**  
```bash
python sign_language_detection.py
```

### 🔹 **4. Train the Model (If Needed)**  
```bash
python train_model.py
```

---

## 🖼 **Preprocessing Pipeline**  
1️⃣ **Frame Extraction:** Convert videos into images at **30 FPS**.  
2️⃣ **Hand Detection:** Use **Mediapipe** to **crop the hand region**.  
3️⃣ **Preprocessing:** Resize (224×224), Normalize, Grayscale, Edge Detection.  
4️⃣ **Data Augmentation:** Flip, Rotate, Scale to **balance dataset**.  
5️⃣ **Train CNN Model (MobileNetV2)** using **Transfer Learning**.  
6️⃣ **Deploy Model in Real-Time** using **OpenCV & PyTorch**.  
7️⃣ **Use Majority Voting over 30 Frames** for **better accuracy**.  
8️⃣ **Convert Predictions to Speech** using **pyttsx3 / gTTS**.  

---

## 🎯 **Real-Time Gesture Detection & Audio Alerts**  
Once deployed, the system:  
✔ Captures **live video feed** from the **webcam**  
✔ Predicts sign gestures using the **trained MobileNetV2 model**  
✔ Uses **majority voting (over 30 frames)** to stabilize predictions  
✔ Converts sign language to **speech output** for emergency alerts  

💡 Example: If a user signs **“HELP”**, the system will:  
🖥 Display: `"Help detected!"`  
🔊 Speak: `"Help! Emergency detected!"`  

---

## 📊 **Model Training & Performance**  
✅ **Trained on a balanced dataset (500 images per class)**  
✅ **Achieved ~90% accuracy** on the test set  
✅ **Used Adam Optimizer, Cross-Entropy Loss, 10 epochs**  
✅ **Confusion matrix analysis** to minimize misclassifications  

---

## 📌 **Future Improvements**  
✅ **Deploy on Mobile** using **TensorFlow Lite**  
✅ **Support More Sign Languages** (ASL, ISL, etc.)  
✅ **Improve Gesture Recognition with LSTMs** (Sequence Learning)  
✅ **Enable Cloud-Based Emergency Alerting**  

---

## 👩‍💻 **Contributions**  
💡 Want to improve this project? Follow these steps:  

1. **Fork the repository**  
2. **Create a new branch** (`git checkout -b feature-branch`)  
3. **Commit changes** (`git commit -m "Added new feature"`)  
4. **Push to GitHub** (`git push origin feature-branch`)  
5. **Submit a Pull Request** 🎉  

---

## 📜 **License**  
This project is **open-source** under the **MIT License**.  

---

### 🚀 **Maintainer & Contact**  
🔗 **GitHub:** [Your GitHub Username](https://github.com/tanishaagarwal195)  
📧 **Email:** tanishaagarwal400@gmail.com 

---

### 🎯 **Final Thoughts**  
🚀 This project bridges the **communication gap** for the **deaf community** in **emergencies** by combining **AI, computer vision, and deep learning**.  

👀 **Try it, contribute, and help make communication more accessible!**  

---

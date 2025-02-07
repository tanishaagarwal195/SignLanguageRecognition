# **Sign Language Recognition for Emergency Communication**  

### **1. Introduction**  
This project aims to develop an **AI-based sign language recognition system** that can assist the **deaf and hard-of-hearing community** in emergency situations. The system captures **hand gestures** via a webcam, classifies them using a **deep learning model**, and provides an **audio alert** for communication with emergency services. The model recognizes **8 emergency-related gestures**:  
✅ **Danger, Emergency, Fire, Help, Please, Police, Stop, Water.**  

This is useful in scenarios where **verbal communication is not possible**, such as emergencies where a deaf person needs to alert authorities.  

---

## **2. Methodology & Workflow**  
The project follows a **structured machine learning pipeline**, consisting of:  

✅ **Data Collection & Preprocessing**  
✅ **Model Selection & Training**  
✅ **Real-Time Integration**  
✅ **Evaluation & Deployment**  

---

### **3. Dataset & Preprocessing**  
Since no standard dataset exists for **emergency sign language gestures**, a **custom dataset was created**. The data processing involved multiple steps:  

#### **3.1. Data Collection & Frame Extraction**  
- **Videos of sign gestures were recorded.**  
- The videos were converted into **frames at 30 FPS** to extract image sequences.  
- Each frame was labeled according to its corresponding gesture.  

#### **3.2. ROI (Region of Interest) Extraction**  
- **Hand detection** was applied using **Mediapipe Hands**, a real-time hand-tracking model.  
- Only the **hand region** was cropped to **eliminate unnecessary background noise**.  
- Frames **without hands** were removed to improve training quality.  

#### **3.3. Image Preprocessing**  
To ensure uniform input for the model, the following preprocessing steps were applied:  
✅ **Resizing:** Images were resized to **224×224 pixels** (MobileNetV2 input size).  
✅ **Normalization:** Pixel values were scaled between **[-1, 1]** for stability.  
✅ **Grayscale Conversion:** Converted to **single-channel images** for feature enhancement.  
✅ **Edge Detection:** Applied **Canny Edge Detection** to highlight hand contours.  

#### **3.4. Data Augmentation & Class Balancing**  
Since some classes had **fewer samples**, data augmentation was used to increase diversity:  
- **Flipping (Horizontal & Vertical)**
- **Rotation (±20°)**
- **Scaling (Zoom-in, Zoom-out)**  

Final **balanced dataset**: **500 images per class**  
✅ **Split:** Train (80%), Validation (10%), Test (10%)  

---

### **4. Model Selection & Training**  
A **Convolutional Neural Network (CNN) based on MobileNetV2** was chosen for sign classification.  

#### **4.1. Why MobileNetV2?**  
✅ **Lightweight & Efficient:** Designed for real-time applications.  
✅ **Pre-trained on ImageNet:** Faster training via **transfer learning**.  
✅ **Works well with limited data.**  

#### **4.2. Model Architecture**  
- **Input Layer:** (224×224×3) images.  
- **Feature Extractor:** Pre-trained **MobileNetV2** (frozen layers).  
- **Fully Connected Layer:** Custom **Dense layer** for **8-class classification**.  
- **Output Layer:** **Softmax activation** for multi-class probability prediction.  

#### **4.3. Training Details**  
- **Optimizer:** Adam (`lr=0.001`)  
- **Loss Function:** Categorical Cross-Entropy  
- **Batch Size:** 32  
- **Epochs:** 10  

During training, validation accuracy was monitored to prevent **overfitting**.  

---

### **5. Real-Time Integration**  
After training, the model was integrated with **OpenCV & PyTorch** for live sign recognition.  

#### **5.1. Capturing Video from Webcam**  
- The webcam captures **live frames** at **30 FPS**.  
- Frames are processed in **real-time** for classification.  

#### **5.2. Temporal Analysis (Majority Voting over Time)**  
- **Issue:** Sign gestures take **2-3 seconds** to form, so a single frame is **not enough**.  
- **Solution:** Predictions were made over **30 frames (~1 sec)**, and the **most frequent prediction** (majority voting) was selected.  

#### **5.3. Audio Feedback System**  
- The predicted sign was converted into **speech output** using **pyttsx3 (Text-to-Speech)**.  
- Example: If a person signs **“HELP”**, the system says **"Help! Emergency detected."**  

✅ **Final Output:** Webcam → Gesture Detection → AI Model → Predicted Class → **Audio Alert**  

---

### **6. Model Evaluation & Results**  
- **Accuracy on test set:** **~90%**  
- **Confusion Matrix Analysis:** Showed some misclassification in similar gestures (e.g., "Stop" vs. "Danger").  
- **Inference Speed:** **~30ms per frame**, making it suitable for real-time use.  

---

### **7. Applications & Future Enhancements**  
✅ **Emergency Communication App:** Can be integrated into mobile apps for **police, fire, or medical alerts**.  
✅ **Sign Language Learning:** Can be expanded to recognize **full sentences**.  
✅ **Mobile Deployment:** Convert the model to **TensorFlow Lite** for mobile devices.  
✅ **Multi-Language Support:** Extend to **ASL (American Sign Language), ISL (Indian Sign Language), etc.**  

---

### **8. Conclusion**  
This project demonstrates a **real-time AI-powered sign language recognition system** tailored for **emergency communication**. The combination of **deep learning, image processing, and real-time inference** enables effective recognition of **critical emergency gestures**. Future work will focus on **expanding the dataset**, **improving accuracy**, and **deploying the system for real-world applications**. 🚀  

---

## **💡 Summary of Steps Used in the Project**
| **Step** | **Methodology Used** |
|-----------|---------------------|
| **Data Collection** | Custom video dataset, frame extraction at **30 FPS** |
| **ROI Detection** | **Mediapipe** for hand tracking & cropping |
| **Preprocessing** | **Resizing, Normalization, Grayscale, Edge Detection** |
| **Data Augmentation** | **Flipping, Rotation, Scaling** to balance dataset |
| **Model Used** | **MobileNetV2 (Transfer Learning)** |
| **Training** | **Adam optimizer, Cross-Entropy Loss, 10 epochs** |
| **Real-Time Integration** | **OpenCV + PyTorch** for **webcam-based detection** |
| **Temporal Processing** | **Majority Voting over 30 frames** |
| **Audio Output** | **pyttsx3 (Text-to-Speech)** for alerts |
| **Accuracy** | **~90% on test data** |
| **Applications** | **Emergency services, accessibility tools, mobile deployment** |

---

## **Final Takeaway**  
🚀 **This AI-powered sign language system is a breakthrough for emergency communication!**  
- ✅ **Fast & Accurate**
- ✅ **Real-Time**
- ✅ **Accessible for the Deaf Community**
- ✅ **Practical for Emergency Situations**  

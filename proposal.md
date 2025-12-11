# 📝 CSC173 Deep Computer Vision Project Proposal

**Student:** Lavigne Kaye S. Sistona, 2022-5619 
**Date:** December 11, 2025

---

## 1. Project Title
**Project Limpyo: Otoscopic Image Analysis for Cerumen Impaction and Infection Risk Assessment**

---

## 2. Problem Statement
The availability of consumer-grade ear cleaning devices with integrated cameras has increased self-examination and cleaning efforts. While accessible, this practice carries a significant risk. **Self-cleaning attempts without awareness or medical guidance can lead to severe complications**, particularly in the presence of underlying ear canal infections (otitis externa) or severely impacted cerumen (earwax). 

Statistics show that improper ear cleaning practices are a common cause of complications. For example, clinical findings indicate that self-cleaning using objects like cotton swabs is a frequent predisposing factor for **otitis externa** and can cause **eardrum perforation** or push cerumen deeper into the canal, resulting in severe impaction.

The core problem addressed by this project is the **lack of an accessible, non-professional system to assess the immediate safety risk** before a user attempts self-cleaning. This project aims to develop a computer vision model that accurately classifies otoscopic images into distinct risk categories, providing a critical "Go/No-Go" recommendation on whether the user can safely proceed or requires professional medical consultation.

---

## 3. Objectives
* **Data Preparation:** Process the raw otoscopic image dataset, applying techniques like consistent scaling, color normalization, and appropriate labeling for the defined risk classes (Low, Medium, High).
* **Model Training:** Implement, train, and fine-tune multiple state-of-the-art **Convolutional Neural Networks (CNNs)** for multi-class image classification.
* **Performance Evaluation:** Validate and evaluate the trained models' performance across all risk classes, focusing on generalization and using metrics such as Precision, Recall, and the F1-Score.
* **Risk Assessment System Simulation:** Demonstrate the model's output in a simulated inference environment to show its practical application as a **"Go/No-Go"** safety check for self-cleaning.

---

## 4. Dataset Plan 

The goal is to map the available otoscopic images into distinct categories that determine the **risk level** associated with attempting self-cleaning.

* **Source:** **Otoscopic Image Dataset** from Kaggle (URL: `https://www.kaggle.com/datasets/ucimachinelearning/otoscopic-image-dataset`).
* **Target Classes (3 Classes):** The data will be curated and mapped into three primary, medically-grounded risk classes:
    * **Class 0: Low Risk (Safe to Clean):** Image shows a generally healthy ear canal with minimal to moderate, non-impacted cerumen.
    * **Class 1: Medium Risk (Caution/Monitor):** Image shows significant cerumen buildup that is not fully obstructing, or mild localized inflammation (e.g., slight redness).
    * **Class 2: High Risk (Refer to Doctor):** Image shows clear signs of danger, specifically **Severe Cerumen Impaction** (obstructing the view) or evidence of **Infection/Abnormality** (e.g., severe inflammation, fluid, or tympanic membrane issues).

* **Data Augmentation:** To make the model robust against variability in consumer-grade cameras (lighting, angle, focus):
    * **Geometric:** Random rotation, horizontal/vertical flipping, and zooming.
    * **Photometric:** Adjustments to brightness and contrast.

---

## 5. Technical Approach 

The approach utilizes established **Transfer Learning** techniques on Convolutional Neural Networks (CNNs) to classify the otoscopic images.

* **Architecture Sketch:**
    1.  **Input:** Otoscopic Image.
    2.  **Pre-processing:** Image is resized and normalized.
    3.  **Core Model:** The pre-processed image is fed into the trained CNN backbone.
    4.  **Classification Head:** The model outputs a probability score for each of the three risk classes.
    5.  **Output:** Final classification and safety recommendation (Low, Medium, or High Risk). 
* **Models:** We will focus on established, high-performing, pre-trained CNN architectures from the ImageNet dataset:
* **Framework:** PyTorch

---

## 6. Expected Challenges & Mitigations

### Challenge 1: Class Imbalance
The "High Risk (Infection/Impaction)" class is expected to be significantly smaller than the "Low Risk" class, leading to potential model bias.
* **Mitigation:** Employ **weighted loss functions** (e.g., class-weighted Cross-Entropy Loss) to penalize errors on minority classes more heavily. Utilize **data augmentation** and potential **oversampling** techniques specifically for the rare, high-risk images.

### Challenge 2: Fine-Grained Feature Detection
Identifying subtle, critical visual cues (e.g., the exact degree of inflammation or the severity of impaction) within the small image area is difficult.
* **Mitigation:** Apply **Transfer Learning** with robust backbones. Investigate the use of **Attention Mechanisms** (e.g., CBAM) within the CNN structure to force the model to focus on diagnostically critical regions like the tympanic membrane or the canal walls.


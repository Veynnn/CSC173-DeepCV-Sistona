# Project Limpyo: Otoscopic Image Analysis for Cerumen Impaction and Infection Risk Assessment


## 1. Project Goal 

The project aims to create a **safety classification system** for images taken inside the ear canal (otoscopic images). This system will help users of at-home ear cleaners decide whether it is safe to proceed or if they need to see a doctor due to earwax impaction or infection.

The final output will be a **"Low Risk," "Medium Risk," or "High Risk"** recommendation based on the visual evidence.

## 2. Dataset Choice

* **Dataset:** **Otoscopic Image Dataset** from Kaggle.
* **Purpose:** The images contain examples of healthy canals, cerumen (earwax), and various infections/abnormalities.
* **Classes:** We will map the images into three risk categories:
    * **Low Risk** (Safe to Clean)
    * **Medium Risk** (Use Caution)
    * **High Risk** (See a Doctor)

## 3. Architecture Sketch

We will use **Transfer Learning** on established deep learning models to classify the images.
* **Process:**
    1.  **Input:** Image from the ear camera.
    2.  **Model:** Trained CNN (ResNet50 or EfficientNetB0) classifies the image.
    3.  **Output:** Risk classification (Low, Medium, or High).
* **Framework:** PyTorch


# Project Limpyo: Otoscopic Image Analysis for Cerumen Impaction and Infection Risk Assessment [in progress]
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Lavigne Kaye s. Sistona, 2022-5619 <br>
**Semester:** [e.g., AY 2025-2026 Sem 1]  

## Abstract
The proliferation of at-home digital otoscopes has increased the risk of ear canal injuries due to improper self-cleaning. Project Limpyo addresses this safety concern by developing an automated classification system designed to assess cerumen (earwax) impaction and infection risks. Utilizing an otoscopic image dataset sourced from Kaggle, this study employs Transfer Learning with deep convolutional neural network (CNN) architectures to categorize ear canal conditions.

The proposed system maps visual data into three distinct safety tiers: Low Risk (safe to clean), Medium Risk (use caution), and High Risk (medical intervention required). Initial development focuses on fine-tuning these models to achieve high sensitivity in detecting abnormalities that contraindicate self-cleaning, such as fungal infections or total impaction. By providing a real-time risk assessment, Project Limpyo contributes a critical safety layer to consumer health technology, potentially reducing the incidence of tympanic membrane perforations and external auditory canal trauma. The final framework, implemented in PyTorch, demonstrates the efficacy of deep CV methods in translating complex medical imagery into actionable, user-friendly safety recommendations.

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
Ear health is a critical yet frequently overlooked component of personal hygiene and general well-being in the Philippines (Newall et al., 2020). Otologic conditions, specifically cerumen impaction and various forms of otitis, are highly prevalent in the country, with some rural and indigenous communities exhibiting infection rates as high as 48.7% due to geographical and financial barriers to specialized ear, nose, and throat (ENT) care (Santos-Cortez et al., 2016). This lack of access often leads to "self-ear cleaning," a practice adopted by over 90% of the population using non-sterile tools like cotton buds, hairpins, or feathers. These practices frequently result in preventable trauma, including eardrum perforations and external auditory canal infections (Sacayan-Quitay et al., 2022).

The recent market entry of affordable "smart" ear cleaners, digital otoscopes that stream live video to a user's smartphone that aims to provide visual guidance for these habits. However, while these devices provide the user with "eyes" inside the canal, they do not provide the clinical "insight" necessary to interpret what is being seen. A layperson may easily mistake an inflamed, infected ear canal or a fungal growth for simple earwax, leading to aggressive cleaning attempts that can rupture the tympanic membrane or push infections deeper into the middle ear (Llamado et al., 2022).

Project Limpyo addresses this safety gap by developing an AI driven classification system that acts as an intelligent "safety buffer." By utilizing deep learning and computer vision, specifically the ResNet50 and EfficientNetB0 architectures, the system analyzes otoscopic images in real-time to categorize them into Low, Medium, or High Risk. This provides an automated "red flag" mechanism that warns users to seek professional medical intervention when abnormalities are detected, potentially reducing the incidence of avoidable ear trauma in Philippine households.

### Objectives
- Objective 1: To develop a deep learning model using ResNet50 architectures that achieves high classification accuracy (>90%) in identifying ear canal abnormalities. <br>
- Objective 2: To integrate decision logic framework that categorizes otoscopic findings into Low, Medium, and High Risk levels for user guidance. <br>
- Objective 3: To deploy the system as a mobile-compatible tool that provides real-time "red flag" alerts to prevent improper self-ear cleaning and encourage medical consultation.

## Related Work

- **High-Performance Classification of Common Ear Pathologies:** Comparative analysis of deep learning architectures, including ResNet50 and EfficientNet, achieved over 94% accuracy in distinguishing chronic otitis media and wax obstruction from normal ear structures [1].
- **VGGNet-19 for Superior Diagnostic Accuracy:** Evaluation of multiple deep learning models identified VGGNet-19 as a high-performing architecture for identifying chronic suppurative otitis media, reaching 98.10% specificity [2].
- **Expert-Level AI Diagnosis:** A two-stage attention-aware convolutional neural network (CNN) matched the diagnostic performance of otolaryngology experts, achieving a 93.4% accuracy in detecting middle ear diseases [3].
- **Computer-Aided Diagnosis (CAD) for Resource-Limited Areas:** Hybrid models utilizing EfficientNetB0 as a feature extractor have been proposed to automate otitis media diagnosis in regions with limited access to ENT specialists [4].
- **Ensemble Models for Real-Time Identification:** The use of ensemble deep learning on datasets exceeding 20,000 images proved effective for the real-time identification of six distinct diagnostic ear categories [5].
- **Big Data in Otoendoscopy:** A large-scale study utilizing over 10,000 images trained nine deep neural networks to successfully classify complex conditions like attic retractions and tumors with 93.67% accuracy [6].
- **Addressing AI Bias in Otology:** Research on image saturation and "eclipse extent" highlights the necessity of training models to ignore irrelevant visual artifacts like lighting or background colors to ensure clinical reliability [7].
- **Attention Mechanisms for Interpretability:** Two-stage attention-aware networks simulate clinical focus on key visual cues, such as eardrum bulging, improving both the accuracy and explainability of AI-generated insights [8].
- **Video-Based Automated Identification:** The "OtoXNet" framework generates composite images from otoscopy videos to diagnose eardrum diseases, outperforming traditional human-selected keyframes [9].
- **Mobile-Ready Optimized Models:** Recent work using Bayesian hyperparameter optimization on CNN architectures achieved 98.10% accuracy, specifically focusing on rapid screening for earwax plugs and chronic infections [10].


## Methodology
### Dataset
- **Source:** Otoscopic Image Dateset [Kaggle: https://www.kaggle.com/datasets/ucimachinelearning/otoscopic-image-dataset]

     
- **Split:** <br>
&emsp; * 70% - Training Data <br>
&emsp; * 15% - Validation Data  <br>
&emsp; * 15% - Test Data <br>


- **Classes:** 5 original classes mapped to 3 risk categories <br>
      &emsp; Normal → Low Risk <br>
      &emsp; Myringosclerosis → Medium Risk <br>
      &emsp; Cerumen Impaction → High Risk <br>
      &emsp; Acute Otitis Media → High Risk <br>
      &emsp; Chronic Otitis Media → High Risk <br>

      
- **Preprocessing:** <br>
      &emsp; Resizing to 200×200 <br>
      &emsp; Normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) <br>
      &emsp; Data Augmentation: RandomHorizontalFlip for training <br>


### Architecture
- **Backbone**: Custom SimpleCNN with 3 convolutional layers
- **Head**: Two fully connected layers with dropout regularization
- **Total Parameters**: 500,355 trainable parameters
- **Input Size**: 200×200×3 RGB images
- **Output**: 3 classes (Low/Medium/High Risk)


#### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Epochs | 15 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Dropout Rate | 0.3 |
| Learning Rate Scheduler | StepLR (step=5, gamma=0.5) |


## Installation
1. Clone repo: `git clone [https://github.com/Veynnn/CSC173-DeepCV-Sistona]`
2. Install deps: `pip install -r requirements.txt`

**requirements.txt:**
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
Pillow>=9.5.0
albumentations>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
ipywidgets>=8.0.0
jupyter>=1.0.0


### References

[1] DergiPark. (2025). *Ear Pathologies Using Deep Learning on Otoscopic Images.* https://dergipark.org.tr/en/pub/atauniamd/issue/78728/1253457 <br>
[2] PMC. (2025). *Deep Learning for Otitis Media Classification using Otoscopic Image.* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9354720/ <br>
[3] BMJ Open. (2021). *Investigating the use of a two-stage attention-aware CNN for the automated diagnosis of otitis media.* https://bmjopen.bmj.com/content/11/1/e044139 <br>
[4] ResearchGate. (2025). *Enhancing Intra-Aural Disease Classification with Attention-Based Deep Learning Models.* https://www.researchgate.net/publication/362489624_Otoscopic_Diagnosis_Using_Deep_Learning <br>
[5] PMC. (2021). *Efficient and Accurate Identification of Ear Diseases Using an Ensemble Deep Learning Model.* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7903424/ <br>
[6] PMC. (2019). *Automated Diagnosis of Ear Disease Using Ensemble Deep Learning with a Big Otoendoscopy Image Database.* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6489390/ <br>
[7] arXiv. (2025). *Towards Reliable Use of Artificial Intelligence to Classify Otitis Media Using Otoscopic Images.* https://arxiv.org/abs/2205.12933 <br>
[8] IJARSCT. (2022). *AI Powered Otoscopic Image Classification for Ear Disease Detection.* https://ijarsct.co.in/Paper6321.pdf <br>
[9] medRxiv. (2021). *OtoXNet - Automated Identification of Eardrum Diseases from Otoscope Videos.* https://www.medrxiv.org/content/10.1101/2021.08.11.21261911v1 <br>
[10] NIH. (2023). *Insight into Automatic Image Diagnosis of Ear Conditions Based on Optimized Deep Learning Approach.* https://pubmed.ncbi.nlm.nih.gov/37444378/ <br>
[11] Newall, J. P., Martinez, N., Swanepoel, D. W., & McMahon, C. M. (2020). A National Survey of Hearing Loss in the Philippines. Asia-Pacific Journal of Public Health, 32(5), 235–241. https://doi.org/10.1177/1010539520937086 <br>
[12] Santos-Cortez, R. L. P., et al. (2016). Genetic and Environmental Determinants of Otitis Media in an Indigenous Filipino Population. Otolaryngology–Head and Neck Surgery, 155(5), 856–862. https://doi.org/10.1177/0194599816661703 <br>
[13] Sacayan-Quitay, N. D., et al. (2022). Hearing and Clinical Otologic Profile of Filipinos Living in Southern Tagalog Region IV-A (CALABARZON). Philippine Journal of Otolaryngology Head and Neck Surgery. <br>
[14] Llamado, C. A. C., et al. (2022). Knowledge, Attitudes, and Practices on Ear Hygiene and Hearing Health among Adults in a Selected Community in Manila. Philippine Journal of Otolaryngology Head and Neck Surgery. <br>


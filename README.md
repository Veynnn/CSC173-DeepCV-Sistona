# Project Limpyo: Otoscopic Image Analysis for Cerumen Impaction and Infection Risk Assessment [in progress]
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Lavigne Kaye s. Sistona, 2022-5619
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

The recent market entry of affordable "smart" ear cleaners—digital otoscopes that stream live video to a user's smartphone—aims to provide visual guidance for these habits. However, while these devices provide the user with "eyes" inside the canal, they do not provide the clinical "insight" necessary to interpret what is being seen. A layperson may easily mistake an inflamed, infected ear canal or a fungal growth for simple earwax, leading to aggressive cleaning attempts that can rupture the tympanic membrane or push infections deeper into the middle ear (Llamado et al., 2022).

Project Limpyo addresses this safety gap by developing an AI-driven classification system that acts as an intelligent "safety buffer." By utilizing deep learning and computer vision, specifically the ResNet50 and EfficientNetB0 architectures, the system analyzes otoscopic images in real-time to categorize them into Low, Medium, or High Risk. This provides an automated "red flag" mechanism that warns users to seek professional medical intervention when abnormalities are detected, potentially reducing the incidence of avoidable ear trauma in Philippine households.

## References
[1] Newall, J. P., Martinez, N., Swanepoel, D. W., & McMahon, C. M. (2020). A National Survey of Hearing Loss in the Philippines. Asia-Pacific Journal of Public Health, 32(5), 235–241. https://doi.org/10.1177/1010539520937086
[2] Santos-Cortez, R. L. P., et al. (2016). Genetic and Environmental Determinants of Otitis Media in an Indigenous Filipino Population. Otolaryngology–Head and Neck Surgery, 155(5), 856–862. https://doi.org/10.1177/0194599816661703
[3] Sacayan-Quitay, N. D., et al. (2022). Hearing and Clinical Otologic Profile of Filipinos Living in Southern Tagalog Region IV-A (CALABARZON). Philippine Journal of Otolaryngology Head and Neck Surgery.
[4] Llamado, C. A. C., et al. (2022). Knowledge, Attitudes, and Practices on Ear Hygiene and Hearing Health among Adults in a Selected Community in Manila. Philippine Journal of Otolaryngology Head and Neck Surgery.


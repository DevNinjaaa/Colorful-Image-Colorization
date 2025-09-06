# Colorful Image Colorization (Classic self supervised learning Project)

Implementation of **Colorful Image Colorization** from  
[Colorful Image Colorization (Zhang et al., ECCV 2016)](https://arxiv.org/abs/1603.08511).

This project explores **self-supervised learning (SSL)** for automatic image colorization by reformulating the task as a **classification problem** in the CIELab color space.  
Due to resource limits, the model is trained on **Tiny ImageNet** instead of the full ImageNet dataset used in the paper.

---

## 🚀 Overview
- Converts grayscale images → realistic color images.
- Predicts color distribution per pixel using CNNs.
- Uses **quantized ab color bins (313 classes)**.
- Includes **class rebalancing** for rare colors.
---
Here are some example results compared with the original paper:

<img width="713" height="227" alt="image" src="https://github.com/user-attachments/assets/edb58b6c-0da3-4f34-adfc-915d5da1293f" />
<img width="860" height="323" alt="image" src="https://github.com/user-attachments/assets/382ac71a-55da-42ff-a5ba-b4fe7a3ee98a" />


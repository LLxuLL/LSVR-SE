# LSVR-SE: Language-Guided Single-View Reconstruction with Semantic Editing

![Framework Overview](overview.png)  
*Figure: Overview of the LSVR-SE framework integrating language-guided reconstruction and semantic editing.*

**LSVR-SE** is an end-to-end differentiable framework for high-fidelity 3D reconstruction and semantically controllable editing from a single image and natural language instructions. It introduces a novel **language-vision-geometry tri-perspective coupling** mechanism, enabling real-time responsive editing such as "add a window" or "replace material" while ensuring topological and physical consistency.  
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)  
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/XXXX.XXXXX)  
[![GitHub Stars](https://img.shields.io/github/stars/LLxuLL/LSVR-SE?style=social)](https://github.com/LLxuLL/LSVR-SE)

---

## ✨ Key Features

- **Hierarchical Semantic Disentanglement Encoder (HSDE)**  
  Aligns CLIP text and image embeddings into a shared 3D latent space via 3D attention fusion, improving cross-modal retrieval accuracy to **83.1%**.

- **Language-Conditioned Neural Radiance Field (LC-NeRF)**  
  Dynamically modulates density and color fields using text embeddings, enabling real-time material and structural editing with **200 ms response time**.

- **Differentiable Programmatic Editing Engine (DPEE)**  
  Parses natural language into parameterized operation trees with physical-topological constraints, achieving **96.2% manifold preservation rate** and supporting zero-shot text-to-IFC conversion.

- **End-to-End Differentiable Pipeline**  
  Unifies reconstruction and editing in a single framework with **8-second inference time** on A100 GPU.

---

## 🚀 Performance Highlights

| Metric                | DTU (PSNR↑) | Objaverse-LVIS (Editing IoU↑) | BIM (Conversion Accuracy↑) | Inference Time (ms) |
|-----------------------|-------------|-------------------------------|----------------------------|---------------------|
| **LSVR-SE (Ours)**    | **26.9 dB** | **65.8%**                     | **68.9%**                  | **36700**           |
| Pixel2Mesh++          | 24.5 dB     | 50.0%                         | -                          | 15200               |
| CLIP-NeRF             | 25.8 dB     | -                             | -                          | -                   |
| Magic3D               | 27.2 dB     | -                             | -                          | 42300               |

---

## 📥 Installation

### Prerequisites
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- CUDA ≥ 11.7
- Open3D ≥ 0.17.0

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/LLxuLL/LSVR-SE.git
   cd LSVR-SE
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pre-trained models:
   ```bash
   wget https://github.com/LLxuLL/LSVR-SE/releases/download/v1.0/lsvr_se_model.pt -P ./checkpoints/
   ```

---

## 🛠️ Quick Start

### Inference with Image and Text Instruction
```python
from models import LSVR_SE

model = LSVR_SE.load_from_checkpoint("checkpoints/lsvr_se_model.pt")
mesh = model.predict(
    image_path="input_image.png",
    instruction="add a 1.2m x 0.8m window to the north wall"
)
mesh.export("output.obj")
```

### Training on Custom Data
```bash
python train.py --config configs/lsvr_se.yaml --data_dir /path/to/dataset
```

---

## 📖 Citation

If you use LSVR-SE in your research, please cite our paper:

```bibtex
@article{xu2024lsvrse,
  title={LSVR-SE: Language-Guided Single-View Reconstruction with Semantic Editing},
  author={Xu, Maoyang and Qi, Guanglei and He, Nana},
  year={2024}
}
```

---

## 📜 License

This project is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License.  
Datasets are used under their respective licenses.

---

## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request for any improvements.  
For questions, contact:  
- Maoyang Xu: 18810775071@163.com  
- Guanglei Qi: qiguanglei@ccbupt.cn  
- Nana He: henana@ccbupt.cn

---

## 🌐 Applications

- **E-commerce AR**: Generate editable 3D product models from single images.  
- **Architectural BIM**: Convert natural language instructions to IFC-standard models.  
- **Game Development**: Semantic editing of 3D assets via natural language.  
- **Metaverse Content Creation**: Language-driven 3D scene generation and editing.

---

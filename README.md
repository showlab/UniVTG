 # UniVTG
> TL; DR: The first video temporal grounding pretraining model, unifying diverse temporal annotations to power moment retrieval (interval), highlight detection (curve) and video summarization (point).

![UniVTG](figures/univtg_demo.jpg)


### **News**
<!--  -->
- [2023.7.31] We release the codes, checkpoints and gradio demo.

## 🌟 Run on your videos
To power practical usage, we release the following checkpoints:

| Video Enc.  | Text Enc.  | Pretraining            | Fine-tuning   |  Download |
| ------------------ |  ------------------ | ------------------ | ------- | ---- |
| CLIP-Base | CLIP-Base | 4M      | -      |   [Drive](https://drive.google.com/drive/folders/1-eGata6ZPV0A1BBsZpYyIooos9yjMx2f?usp=sharing)  |
| CLIP-Base | CLIP-Base | 4M | QVHL + Charades + NLQ + TACoS + ActivityNet + DiDeMo      |  [Drive](https://drive.google.com/drive/folders/1l6RyjGuqkzfZryCC6xwTZsvjWaIMVxIO?usp=sharing)  


Additionally, we've built gradio interfaces for easy interaction. 
Run `python3 main_gradio.py`

<img src="figures/gradio.png" alt="UniVTG" width="50%">


## ⚙️ Preparation

Please find instructions in [install.md](install.md) to setup environment and datasets.

## 📦 Model Zoo

Download checkpoints in [model.md](model.md) to reproduce the benchmark results.

## 😊 Acknowledgement

This codebase is based on [moment_detr](https://github.com/jayleicn/moment_detr), [HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor), [UMT](https://github.com/tencentarc/umt).

We thank the authors for their open-source contributions.

# Model Zoo
## Pretraining
> In paper, we use Slowfast R50 + CLIP-B/32 for pretraining (row 3), and fine-tune on single specified benchmark. We release the row 1, 2 and 4 to power practice usage.

| Video Enc.  | Text Enc.  | Pretraining            | Fine-tuning   |  Checkpoints |
| ------------------ |  ------------------ | ------------------ | ------- | ---- |
| CLIP-B/32 | CLIP-B/32 | 4M      | -      |   [Google Drive](https://drive.google.com/drive/folders/1-eGata6ZPV0A1BBsZpYyIooos9yjMx2f?usp=sharing)  |
| CLIP-B/32 | CLIP-B/32 | 4M | QVHL + Charades + NLQ + TACoS + ActivityNet + DiDeMo      |  [Google Drive](https://drive.google.com/drive/folders/1l6RyjGuqkzfZryCC6xwTZsvjWaIMVxIO?usp=sharing)  
| Slowfast R50 + CLIP-B/32 | CLIP-B/32 | 4M      | -      |   [Google Drive](https://drive.google.com/drive/folders/1eWpuTTBRaMoV4UsEteQHAf5t4dU7uwrl?usp=sharing)  |
| Slowfast R50 + CLIP-B/32 | CLIP-B/32 | 4M | QVHL + Charades + NLQ + TACoS + ActivityNet + DiDeMo      |  [Google Drive](https://drive.google.com/drive/folders/1pzHDW82Eja7OeH01AnkWNFsXH8JANnZX?usp=sharing)  

> For below downstream tasks, checkpoints are trained by Slowfast R50 + CLIP-B/32 features.

## Joint Moment Retrieval and Highlight Detection
> Please follow the instruction [here](https://github.com/jayleicn/moment_detr/blob/main/standalone_eval/README.md) to submit the test set results to [Codelab](https://codalab.lisn.upsaclay.fr/competitions/6937#results).

| Datasets  | (MR test) mAP avg | (HD test) HIT@1 | (MR val) mAP avg | (HD val) HIT@1 |  Checkpoints + Configs + Prediction + Tensorboard Log |
| ------------------ |  ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | 
| QVHL |  35.47 | 60.96 |  36.13 | 61.81 | [Google Drive](https://drive.google.com/drive/folders/1EqwZSOVeKBCjcHe6SfeUjxM4fN6xrPf3?usp=drive_link) |
| QVHL (w/ PT) |  43.63 | 66.28 |  45.44 | 68.77 | [Google Drive](https://drive.google.com/drive/folders/1ms53Lfm__zrzlvBsadIT6b17vUjFhRG7?usp=sharing) |


## Moment Retrieval
| Datasets  | R1 @ 0.3 | mIoU | Checkpoints + Configs + Prediction + Tensorboard Log |
| ------------------ |  ------------------ | ------------------ | ------------------ | 
| NLQ (w/ PT) |  11.74 | 7.88 | [Google Drive](https://drive.google.com/drive/folders/1u1__kGX2o87kvyh4GiShcEykcVIzDsbs?usp=drive_link) |
| Charades (w/ PT) |  72.63 | 52.17  | [Google Drive](https://drive.google.com/drive/folders/1xXw0QgJiW7m6lPX1dH-MFXU983IxJiG_?usp=drive_link) |
| Tacos (w/ PT) | 56.11 | 38.63  | [Google Drive](https://drive.google.com/drive/folders/1EX3XR5D-mcRRgWl5vKy4iVKXZaoLEeJM?usp=drive_link) |

## Highlight Detection
| Datasets  | Domain | mAP | Checkpoints + Configs + Prediction |
| ------------------ |  ------------------ | ------------------ | ------------------ | 
| Youtube (w/ PT) | dog | 74.25 | [Google Drive](https://drive.google.com/drive/folders/1gTYyS0LiTSOS0yZJ9sO_UnGrQO1Xvfe3?usp=drive_link)
| Youtube (w/ PT) | gymnastics | 78.89 |  [Google Drive](https://drive.google.com/drive/folders/1JqP9UtWVCiBgdEd39dB6LEYUvOnc3_RE?usp=drive_link)
| Youtube (w/ PT) | parkour | 74.39 |  [Google Drive](https://drive.google.com/drive/folders/1EgWctX7u2vcl9EzOlWqwlo1Pcnw_qmNE?usp=drive_link)
| Youtube (w/ PT) | skating | 84.87 | [Google Drive](https://drive.google.com/drive/folders/1JqP9UtWVCiBgdEd39dB6LEYUvOnc3_RE?usp=drive_link)
| Youtube (w/ PT) | skiing | 75.13 | [Google Drive](https://drive.google.com/drive/folders/1l33mxpj4fUCi6zEp1vumVGrV4WXxZpIa?usp=drive_link)
| Youtube (w/ PT) | surfing | 83.85 | [Google Drive](https://drive.google.com/drive/folders/12BsF7Do756K8WUxfSJu2O2fVCmGDSsJg?usp=drive_link)

| Datasets  | Domain | mAP | Checkpoints + Configs + Prediction + Tensorboard Log |
| ------------------ |  ------------------ | ------------------ | ------------------ | 
| TVSum (w/ PT) | BK | 91.78 | [Google Drive](https://drive.google.com/drive/folders/10WDzO7ekh22bk25hYsL6U7tRlXgmwQgp?usp=drive_link)
| TVSum (w/ PT) | BT | 90.47 | [Google Drive](https://drive.google.com/drive/folders/1rrjgmZuc3RvXpZ-NoHRlQLOs-X6ST2Qh?usp=drive_link)
| TVSum (w/ PT) | DS | 77.57 | [Google Drive](https://drive.google.com/drive/folders/14lA9xx6QNKldsFTsfVbjTGSYcA9iDqIR?usp=drive_link)
| TVSum (w/ PT) | FM | 74.33 | [Google Drive](https://drive.google.com/drive/folders/1M31fhylLSi-PGBFgz2-DibNuwarPYH0N?usp=drive_link)
| TVSum (w/ PT) | GA | 89.78 | [Google Drive](https://drive.google.com/drive/folders/1cA7qOhI4gNPG9KDX6VOiy0-jipIn2yRg?usp=drive_link)
| TVSum (w/ PT) | MS | 83.83 | [Google Drive](https://drive.google.com/drive/folders/1iROWXH4N3FDk7dvYsd58YUQ66RcAkAkX?usp=drive_link)
| TVSum (w/ PT) | PK | 82.22 | [Google Drive](https://drive.google.com/drive/folders/1SbUfZ-XI2p_NHtE6Vwr842udVxoT7Pvp?usp=drive_link)
| TVSum (w/ PT) | PR | 85.81 | [Google Drive](https://drive.google.com/drive/folders/1HY8PQ--dZcyMvn7Fjey3wkos-3j0RiPC?usp=drive_link)
| TVSum (w/ PT) | VT | 92.04 | [Google Drive](https://drive.google.com/drive/folders/1TpLp0mIMerOsA2emAruADK3DRdUqYlft?usp=drive_link)
| TVSum (w/ PT) | VU | 77.81 | [Google Drive](https://drive.google.com/drive/folders/10WDzO7ekh22bk25hYsL6U7tRlXgmwQgp?usp=drive_link)


## Video Summarization
| Datasets  | F1 score | Checkpoints + Configs + Prediction + Tensorboard Log |
| ------------------ |  ------------------ | ------------------ | 
| V1 (w/ PT) |  49.85  | [Google Drive](https://drive.google.com/drive/folders/18_svNtHT-kBsCk4Ca2fRDUCUUg7ab_nR?usp=drive_link)
| V2 (w/ PT) |  56.97  | 👆
| V3 (w/ PT) |  59.35 | 👆
| V4 (w/ PT) | 40.62 | 👆

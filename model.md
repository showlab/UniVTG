# Model Zoo
## Pretraining
> In paper, we use Slowfast R50 + CLIP-B/16 for pretraining (row 3), and fine-tune on single specified benchmark. We release the row 1, 2 and 4 to power practice usage.

| Video Enc.  | Text Enc.  | Pretraining            | Fine-tuning   |  Checkpoints |
| ------------------ |  ------------------ | ------------------ | ------- | ---- |
| CLIP-B/16 | CLIP-B/16 | 4M      | -      |   [Google Drive](https://drive.google.com/drive/folders/1-eGata6ZPV0A1BBsZpYyIooos9yjMx2f?usp=sharing)  |
| CLIP-B/16 | CLIP-B/16 | 4M | QVHL + Charades + NLQ + TACoS + ActivityNet + DiDeMo      |  [Google Drive](https://drive.google.com/drive/folders/1l6RyjGuqkzfZryCC6xwTZsvjWaIMVxIO?usp=sharing)  
| Slowfast R50 + CLIP-B/16 | CLIP-B/16 | 4M      | -      |   [Google Drive](https://drive.google.com/drive/folders/1eWpuTTBRaMoV4UsEteQHAf5t4dU7uwrl?usp=sharing)  |
| Slowfast R50 + CLIP-B/16 | CLIP-B/16 | 4M | QVHL + Charades + NLQ + TACoS + ActivityNet + DiDeMo      |  [Google Drive](https://drive.google.com/drive/folders/1pzHDW82Eja7OeH01AnkWNFsXH8JANnZX?usp=sharing)  

> For below downstream tasks, checkpoints are trained by Slowfast R50 + CLIP-B/16 features.

## Joint Moment Retrieval and Highlight Detection
| Datasets  | (MR) mAP avg | (HD) HIT@1 | Checkpoints |
| ------------------ |  ------------------ | ------------------ | ------------------ | 
| QVHL |  35.47 | 60.96 | TBD |
| QVHL (w/ PT) |  43.63 | 66.28 | TBD |

## Moment Retrieval
| Datasets  | R1 @ 0.3 | mIoU | Checkpoints |
| ------------------ |  ------------------ | ------------------ | ------------------ | 
| NLQ |  7.28 |  4.91 |TBD |
| NLQ (w/ PT) |  11.74 | 7.88 |TBD |
| Charades |  70.91 | 50.10  |TBD |
| Charades (w/ PT) |  72.63 | 52.17  |TBD |
| Tacos |  51.44 | 33.60  |TBD |
| Tacos (w/ PT) | 56.11 | 38.63  |TBD |



## Highlight Detection
| Datasets  | Domain | mAP | Checkpoints |
| ------------------ |  ------------------ | ------------------ | ------------------ | 
| Youtube Highlights |  
| Youtube Highlights |  
| TVSum |  
| TVSum |  

## Video Summarization
| Datasets  | F1 score | Checkpoints |
| ------------------ |  ------------------ | ------------------ | 
| V1 |  49.85  |
| V2 |  56.97  |
| V3 |  59.35 |
| V4 | 40.62 |

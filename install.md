# Installment
## Environment

```bash
git clone https://github.com/showlab/UniVTG
cd UniVTG

conda create --name univtg python=3.8
pip install -r requirements.txt
```

## Datasets

> An engineering contribution is that we unify most video temporal tasks by the same features, which makes **pre-training** or **cross-training** flexible.

1.  Download the features and metadata for pertaining and downstream datasets. (skip pretraining if not needed)

| Dataset            | Task    |  Metadata |  Video (Slowfast R50) | Video (CLIP B/32) | Text (CLIP B/32) |
| ------------------ | ------- | ---- | ---- | ---- | ---- | 
| Point ([Ego4D](https://ego4d-data.org/docs/challenge/))      | PT      | [548 MB](https://drive.google.com/file/d/1c4b9qB8EgULpMQZlowV_Dj-r2BT4gLl7/view?usp=drive_link)     | [27.1 GB](https://drive.google.com/file/d/1J0e52sNaXz-gMmCVyA6LfmgQzlB7BtW6/view?usp=drive_link) | [5.7 GB](https://drive.google.com/file/d/1Ij2gjKAY-yfmvaPatZ0-q4T1zGy-GzGs/view?usp=drive_link) | [30.7 GB](https://drive.google.com/file/d/1Ld8AkOwktsmR9uG1RW7R-eyPozTw8tFm/view?usp=drive_link)
| Interval ([VideoCC](https://github.com/google-research-datasets/videoCC-data)) | PT      |  [155 MB](https://drive.google.com/file/d/1dDPEplMizTANYs-GAtAdkx1UR69deGtx/view?usp=drive_link)    |  [300 GB](https://drive.google.com/drive/folders/1-xRQ2o8MHcL9JfjPWEu_q14DFEgqLRgS?usp=sharing)  | [62.5 GB](https://drive.google.com/file/d/1J29Nuurp9Eoksm8V6_RZlOa_FZAKzzeM/view?usp=drive_link) | [12.6 GB](https://drive.google.com/file/d/1LZs0T5ssD8AOMVZPSrrcXKDDDmD8ocbl/view?usp=sharing)
| Curve ([VideoCC](https://github.com/google-research-datasets/videoCC-data))    | PT      |   [3.8GB](https://drive.google.com/file/d/1e8xMLqy3dW0JiRp6Bld8H-4OqQD3Kgi8/view?usp=drive_link)   |  👆   |  👆 | [132 MB](https://drive.google.com/file/d/1L_OWKRHMfE5O2wrIjePIGgw207_NzSB7/view?usp=drive_link)
| [QVHighlights](https://github.com/jayleicn/moment_detr)       | MR + HL |  [5 MB](https://drive.google.com/drive/folders/1aFEXVD8Y6gu84dV1kgaDl15GxQQ42d8S?usp=drive_link)    | [4.0 GB](https://drive.google.com/file/d/1JBqWqQshxyqyl5GbhfvZY1ysg2-L5i99/view?usp=drive_link) | [940 MB](https://drive.google.com/file/d/1JJ65MzCTYRlQD_bkKGrWXT4Xi9vzseQL/view?usp=drive_link) | [172 MB](https://drive.google.com/file/d/1JOaB04UCRqDcGI1IRrhwo5vS7FNaJW-R/view?usp=drive_link)
| [Charades-STA](https://prior.allenai.org/projects/charades)       | MR      |  [4 MB](https://drive.google.com/drive/folders/1YuO1CPyWurjKZjGgHqX2m4mNrcr2Xstd?usp=drive_link)    | [1.3 GB](https://drive.google.com/file/d/1JPkrJcpSwJqrGlq-aIW58UgxWRenVPVN/view?usp=drive_link) | [305 MB](https://drive.google.com/file/d/1JQGEz6jiizAccylBDZoMQQHMDwcQ7Nkv/view?usp=drive_link) | [178 MB](https://drive.google.com/file/d/1JYXjl0AnKHjDYb4_c-SUp4zq-_pXj_tU/view?usp=drive_link)
| [NLQ](https://github.com/EGO4D/episodic-memory)                | MR      |  [3 MB](https://drive.google.com/drive/folders/1kICOuJ1-F3zqChfoI2NRreJWP2ffLzaf?usp=drive_link)    | [1.8 GB](https://drive.google.com/file/d/1Jh_nRO_NnAo-7t5EJnPmO_TrYUJ1z0to/view?usp=sharing) | [404 MB](https://drive.google.com/file/d/1JiHuoMz2RZ7PsagPt4QI8SP0IgGoGai0/view?usp=sharing) | [184 MB](https://drive.google.com/file/d/1Jjn4c0eVd8MpJKt-xiC_3OSqiYQSTBdk/view?usp=drive_link)
| [TACoS](https://github.com/jiyanggao/TALL)              | MR      |  [2 MB](https://drive.google.com/drive/folders/1aQ0mrXR7ZDfNiawqzQwgmzD3XNXUewDQ?usp=drive_link)    | [81 MB](https://drive.google.com/file/d/1J_QsWPCV0JSGaArnKqGGh0cghbmFArYq/view?usp=drive_link) | [18 MB](https://drive.google.com/file/d/1JdfxrAilgziodJF08c5dMcE4rc8mszYO/view?usp=drive_link) | [244 MB](https://drive.google.com/file/d/1JfO1nMdGeGlIpe5tGdjVEbsfCRMRJhjj/view?usp=drive_link)
| [YoutubeHL](https://github.com/aliensunmin/DomainSpecificHighlight)          | HL      |  [1 MB](https://drive.google.com/drive/folders/1bWU4DuieYzt4R_K5FOb3wBKZUP5PAiGk?usp=drive_link)    | [427 MB](https://drive.google.com/file/d/1LlfGdHCqtnrffCkdhXtG6Ut50U6hWYPr/view?usp=drive_link) | [95 MB](https://drive.google.com/file/d/1LnyDZraTiQFSDMrDmCqKRby9vnkpIMIR/view?usp=drive_link) | [2 MB](https://drive.google.com/file/d/1Lv0ctJpjOCN4cup-ZgfNlZVqV-HDGt4n/view?usp=drive_link)
| [TVSum](http://people.csail.mit.edu/yalesong/tvsum/)              | HL      |  [1 MB](https://drive.google.com/drive/folders/1b7pcCIZYCCV705rRQva7KGSmtMFG05Su?usp=drive_link)    | [28 MB](https://drive.google.com/file/d/1Lx63BWsM9fqDW0fxmu9otyjpUMqkFyyT/view?usp=drive_link) | [6 MB](https://drive.google.com/file/d/1Lw_8k3bbTdT0UVQyg5v9OM7y9PC9f8Kk/view?usp=drive_link) | [1 MB](https://drive.google.com/file/d/1LxJkFy530IcpiVVNskEU0hCZlBNCHjak/view?usp=drive_link)
| [QFVS](https://arxiv.org/pdf/1707.04960.pdf)               | VS      |  [1MB](https://drive.google.com/drive/folders/18RpOVDrroY2gZ82ISObjAwWZz-i84XhD?usp=drive_link)    | [455 MB](https://drive.google.com/drive/folders/1PjFWGw35j6cePLuMdN30BP64dsxQm3sQ?usp=drive_link) | 👈 | [1MB](https://drive.google.com/drive/folders/1JXI6Xc6Fj-Lc2R4I7pbIiwGBqgG2sSn5?usp=drive_link)
| [ActivityNet](http://activity-net.org/) (optional)              | MR      |  [10 MB](https://drive.google.com/drive/folders/1Xnmr9OR3q-nB99hkcUp-z6jRqH2HuaPX?usp=drive_link)    | [4.5 GB](https://drive.google.com/file/d/1LySSKToHUF-4NI_ozr0GdRbh3EFefaZG/view?usp=drive_link) | [1.0 GB](https://drive.google.com/file/d/1M7MSAvXVrhGqJVs-PJe-XVqux5fRVgw9/view?usp=drive_link) | [958 MB](https://drive.google.com/file/d/1M8MOUOb-Z14V9DdAb6ABfYpULdU8fZ27/view?usp=drive_link)
| [DiDeMo](https://github.com/LisaAnne/TemporalLanguageRelease) (optional)              | MR      |  [6 MB](https://drive.google.com/drive/folders/1ZW0RgUvIfbDSEjl0-jbTkBCKWTo19HNi?usp=drive_link)    | [1.1 GB](https://drive.google.com/file/d/1MJsg4RvrfIG_ShMIP2-uudzjbItHTBRJ/view?usp=drive_link) | [269 MB](https://drive.google.com/file/d/1MKy9KVIuPlrXF1JY6PSN4yNaP8uUYdmd/view?usp=drive_link) | [443 MB](https://drive.google.com/file/d/1MRjmg58lSTkNOyUHlbjCo7WCvmXQyc2f/view?usp=drive_link)
| [HACS](https://github.com/hangzhaomit/HACS-dataset) (optional)              | MR      |  [15 MB](https://drive.google.com/drive/folders/1_ghi5RxD7aT1PrSAp__kyUtEp-tsOpoD?usp=drive_link)    | [13.1 GB](https://drive.google.com/file/d/1MkeOP01gtgmav7uv6DSdj0wSUQdBE4Aq/view?usp=drive_link) | [3.0 GB](https://drive.google.com/file/d/1Moyng16x_cnpAcWwUxzthTq23TK5cFBZ/view?usp=drive_link) | [177 MB](https://drive.google.com/file/d/1MpcfKSWsKIwMFAdofNi0sUyaMRiwfHP_/view?usp=drive_link)
| [COIN](https://github.com/hangzhaomit/HACS-dataset) (optional)              | MR      |  [8 MB](https://drive.google.com/drive/folders/1cNRZJG65-SrtDGsC5aWlojkwWMU8lLN6?usp=drive_link)    | [2.3 GB](https://drive.google.com/file/d/1cw2-BldNQNZyKDInQ0r2_JtgL-v6qGDn/view?usp=drive_link) | [556 MB](https://drive.google.com/file/d/1csHu8D7V8NpLChA5Z-3cVmXwnyGApfnW/view?usp=drive_link) | [30 MB](https://drive.google.com/file/d/1cqEfYOjWDakv8Fri8-sxYapmC2yh_tsG/view?usp=drive_link)


2. Unzip the downloaded tar by

```
tar -xvf {tar_name}.tar
mv data/home/qinghonglin/univtg/data/{dset_name}/* .  # Replace dset_name accordingly
```

> For VideoCC Slowfast features, first group multiple sub-zips into the same one, then unzip it.

```
gunzip vid_slowfast_*.gz
cat vid_slowfast_* > vid_slowfast.tar
```

3. Organize the data / features in the following structure
   ```bash
   univtg
   ├── eval
   ├── data
   │   ├── qfvs
   │   ├── tvsum
   │   ├── youtube
   │   ├── tacos
   │   ├── ego4d
   │   ├── charades
   │   │   ├── metadata
   │   │   │   ├──charades_test.jsonl
   │   │   │   └──charades_train.jsonl
   │   │   ├── txt_clip
   │   │   ├── vid_clip
   │   │   └── vid_slowfast
   │   └── qvhighlights
   │       ├── metadata
   │       │   ├──qvhighlights_test.jsonl
   │       │   ├──qvhighlights_train.jsonl
   │       │   └──qvhighlights_val.jsonl
   │       ├── txt_clip
   │       ├── vid_clip
   │       └── vid_slowfast
   ├── main
   ├── model
   ├── utils
   ├── README.md
   └── ···
   ```

4. (Optional) We extract video features (Slowfast R/50 and CLIP B/32) based on this repo: [HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor), you can use it extract other benchmarks or videos; We extract text features (CLIP B/32) by `run_on_video/text_extractor.py`


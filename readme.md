## Image Dataset (Added Image Folder)

1. **Scrape Street View Images via Google API**  
   - Notebook: `gsv_image2.ipynb`  
   - Output Folder: `added_image`

2. **Image Training: CLIP**  
   - Notebook: `image_clip_parametric.ipynb`  
     - Generate: `Xadded_img_clip.npy`  
   - Notebook: `image_clip_som.ipynb`  
     - Generate: `som_image_clip_best.pkl`

3. **Image Training: DINOv1**  
   - Notebook: `DINOv1_image_parametric.ipynb`  
     - Generate: `X_img_dino_vitb8.npy`, `image_names_dino_vitb8.txt`  
   - Notebook: `DINOv1_image_parametric2.ipynb`  
     - Generate: `som_image_best.pkl`

> **Compare the two models and choose DINOv1.**

---

## Text Dataset

- **Random News**: `random_news_500.txt`  
- **Drama Texts**: `all_identities_by_sheet.txt`

1. **News Parameterization**  
   - Notebook: `tfidf_doc2vector_compare.ipynb`

2. **BERT Parameterization**  
   - Notebook: `som_text_bert.ipynb`  
     - Generate: `X_bert.npy`, `som_bert_best.pkl`

> **Compare the two models and choose BERT.**

3. **Drama Texts Parameterization**  
   - Generate: `all_identities_by_sheet.npy`

---

## Matching

- **Text–Text Match**  
  - Notebook: `text_text_match.ipynb`  
  - Generate: `matching_table.xlsx`

- **Text–Image Match**  
  - Notebook: `text_image_match.ipynb`  
  - Generate: `matches_text2img.json`, `matches_img2text.json`

---

## Search Engine Design

1. **Data Packaging**  
   - Script: `prepare_data.py`  
   - Generate: `data.json`

2. **Front‑End Interface**  
   - File: `search_interface.html`  
   - Features: Cross‑domain interactive search (Drama → News → Image → Drama)

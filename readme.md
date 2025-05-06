Image dataset (added image folder)

Step 1:Scrape about 900 Street view images collected using the Google API.
    open gsv_image2.ipynb
    generate added image folder
Step 2:Image train: Clip
    open image_clip parametric.ipynb to parametrize the images
    generate Xadded_img_clip.npy
    open image_clip_som.ipynb to train the SOM
    generate som_image_clip_best.pkl
Step 3:Image train: DINOV1
    open DINOv1_image parametric.ipynb to train
    generate X_img_dino_vitb8.npy image_names_dino_vitb8.txt
    open DINOv1_image parametric2.ipynb to change parameters and train again
    generate som_image_best.pkl
compare the two models and choose DINOv1

Text dataset
random news:random_news_500.txt
drama texts:all_identities_by_sheet.txt

Step 1:news parameterization
    open tfidf_doc2vector compare.ipynb to compare the two models
Step 2:Bert parameterization
    open som_text bert.ipynb to train the bert model
    generate X_bert.npy som_bert_best.pkl
compare the two models and choose bert
Step 3:drama texts parameterization
    generate all_identities_by_sheet.npy

Text-text match
    open text_text match.ipynb to match the texts and visualize the results
    generate matching_table.xlsx

Text-image match
    open text_image match.ipynb to match the texts with images and visualize the results
    generate matches_text2img.json matches_img2text.json

Search engine design
Step 1. Packages all raw lists, embeddings (as lists), and mappings into a single data.json so the front‑end can load everything in one request and power the interactive Drama→News→Image→Drama search.(prepare_data.py)
     generate data.json
Step 2: Provides a completely front‑end interactive interface that lets users perform a four‑step cross‑domain search:
Drama → News → Image → Drama
    open search_interface.html

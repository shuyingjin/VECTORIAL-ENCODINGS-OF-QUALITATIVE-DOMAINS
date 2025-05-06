# prepare_data.py

import os, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


DRAMA_TXT = "all_identities_by_sheet.txt"
DRAMA_NPY = "all_identities_by_sheet.npy"
NEWS_TXT = "random_news_500.txt"
NEWS_NPY = "X_bert.npy"
IMG_DIR = "added image"
IMG_NPY = "X_img_dino_vitb8.npy"

TOP_K_D2N = 5  # Drama→News Top‑K
TOP_K_N2I = 5  # News→Image Top‑K
# ======================================


with open(DRAMA_TXT, "r", encoding="utf-8") as f:
    drama_texts = [l.strip() for l in f if l.strip()]
X_drama = np.load(DRAMA_NPY)


with open(NEWS_TXT, "r", encoding="utf-8") as f:
    news_texts = [l.strip() for l in f if l.strip()]
X_news = np.load(NEWS_NPY)


image_names = sorted(
    [
        fn
        for fn in os.listdir(IMG_DIR)
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", "gif"))
    ]
)
X_img = np.load(IMG_NPY)


assert len(drama_texts) == X_drama.shape[0],
assert len(news_texts) == X_news.shape[0],
assert len(image_names) == X_img.shape[0],


print("Computing Drama→News…")
sim_d2n = cosine_similarity(X_drama, X_news)  # (N_drama, N_news)
drama2news = {
    i: [[int(j), float(sim_d2n[i, j])] for j in sim_d2n[i].argsort()[-TOP_K_D2N:][::-1]]
    for i in range(len(drama_texts))
}

print("Computing News→Image…")
sim_n2i = cosine_similarity(X_news, X_img)  # (N_news, N_img)
news2img_idx = {
    i: [[int(j), float(sim_n2i[i, j])] for j in sim_n2i[i].argsort()[-TOP_K_N2I:][::-1]]
    for i in range(len(news_texts))
}


with open("matches_drama2news.json", "w", encoding="utf-8") as f:
    json.dump(drama2news, f, ensure_ascii=False, indent=2)
with open("matches_news2img_idx.json", "w", encoding="utf-8") as f:
    json.dump(news2img_idx, f, ensure_ascii=False, indent=2)
print("→ Saved matches_drama2news.json & matches_news2img_idx.json")


print("Writing data.json…")
data = {
    "drama_texts": drama_texts,
    "news_texts": news_texts,
    "image_names": image_names,
    "X_drama": X_drama.tolist(),
    "X_news": X_news.tolist(),
    "X_img": X_img.tolist(),
    "matches_drama2news": drama2news,
    "matches_news2img_idx": news2img_idx,
}
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print("→ data.json has been written. Done.")

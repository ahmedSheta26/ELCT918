# make_figure_arabic.py -- draws every step for one Arabic digit image and checks it with the MADBase model
import sys
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from preprocessing_arabic import preprocess, TARGET_FRAC
from model import LeNet5

path = sys.argv[1] if len(sys.argv) > 1 else "arabic_1.jpg"   # image path from the command line
img = cv2.imread(path)
assert img is not None, f"could not read {path}"               # stop with a message if the file is missing

tensor, s = preprocess(img, return_steps=True)

# ---- Text report ----
print("Detection         :", s["note"])
if s["stroke_frac"] is not None:
    print(f"Stroke width      : {s['stroke_frac']:.3f} of digit size before thickening (target {TARGET_FRAC})")
    print("Bounding box      :", s["bbox"], "(x, y, width, height)")
    model = LeNet5()
    model.load_state_dict(torch.load("lenet_madbase.pt", map_location="cpu"))   # the MADBase weights
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)    # scores -> probabilities
        conf, pred = probs.max(dim=1)                  # highest probability and its digit
    print(f"Predicted digit   : {pred.item()}   confidence: {conf.item():.3f}")
else:
    print("No digit accepted, so the model was not run.")

# ---- Figure: one panel per step ----
fig, ax = plt.subplots(1, 6, figsize=(18, 3.5))
ax[0].imshow(cv2.cvtColor(s["original"], cv2.COLOR_BGR2RGB)); ax[0].set_title("1. Original")
ax[1].imshow(s["gray_blur"], cmap="gray");                   ax[1].set_title("2. Gray + blur")
ax[2].imshow(s["binary"], cmap="gray");                      ax[2].set_title("3. Threshold + invert")
if s["bbox"] is not None:
    x, y, w, h = s["bbox"]
    ax[2].add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
ax[3].imshow(s["crop"], cmap="gray");                        ax[3].set_title("4. Cropped + thickened")
ax[4].imshow(s["padded32"], cmap="gray");                    ax[4].set_title("5. Centred 28x28 + pad = 32x32")
ax[5].imshow(s["tensor"][0, 0], cmap="gray")
ax[5].set_title(f"6. Normalised tensor\n[{s['tensor'].min():.2f}, {s['tensor'].max():.2f}]")
for a in ax: a.axis("off")
plt.tight_layout()
plt.savefig("preprocessing_steps_arabic.png", dpi=150)       # goes in the report
plt.show()
# make_figure.py -- draws every preprocessing step and checks the result with the trained model
import sys
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from preprocessing import preprocess
from model import LeNet5

path = sys.argv[1] if len(sys.argv) > 1 else "digit.jpg"   # image path from the command line
img = cv2.imread(path)                                      # loads as BGR
assert img is not None, f"could not read {path}"           # stops with a message if the file is missing

tensor, s = preprocess(img, return_steps=True)

# ---- Figure: one panel per step ----
fig, ax = plt.subplots(1, 6, figsize=(18, 3.5))
# matplotlib expects RGB, OpenCV gives BGR, so convert only for display.
ax[0].imshow(cv2.cvtColor(s["original"], cv2.COLOR_BGR2RGB)); ax[0].set_title("1. Original")
ax[1].imshow(s["gray_blur"], cmap="gray");                   ax[1].set_title("2. Gray + blur")
ax[2].imshow(s["binary"], cmap="gray");                      ax[2].set_title("3. Threshold + invert + dilate")
if s["bbox"] is not None:   # draw the bounding box found in the crop step
    x, y, w, h = s["bbox"]
    ax[2].add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
ax[3].imshow(s["crop"], cmap="gray");                        ax[3].set_title("4. Cropped digit")
ax[4].imshow(s["padded32"], cmap="gray");                    ax[4].set_title("5. Centred 28x28 + pad = 32x32")
ax[5].imshow(s["tensor"][0, 0], cmap="gray");                ax[5].set_title("6. Normalised tensor")
for a in ax: a.axis("off")                                  # hide pixel-number axes
plt.tight_layout()
plt.savefig("preprocessing_steps.png", dpi=150)             # this goes in the report
plt.show()

# ---- Sanity check: does the trained model recognise it? ----
model = LeNet5()
model.load_state_dict(torch.load("lenet_mnist.pt", map_location="cpu"))  # load weights; cpu is fine here
model.eval()                       # evaluation mode (we are not training)
with torch.no_grad():              # no gradient bookkeeping needed for prediction
    logits = model(tensor)                         # 10 raw scores
    probs = torch.softmax(logits, dim=1)           # scores -> probabilities that sum to 1
    conf, pred = probs.max(dim=1)                  # highest probability and its digit
print(f"Predicted digit: {pred.item()}   confidence: {conf.item():.3f}")
print("Tensor shape:", tuple(tensor.shape), "(expected (1, 1, 32, 32))")
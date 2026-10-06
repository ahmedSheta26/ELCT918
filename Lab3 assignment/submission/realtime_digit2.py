# realtime_digit.py -- live digit recognition with LeNet-5 (English MNIST or Arabic MADBase)
# Run:  python realtime_digit.py                          (English digits, laptop camera)
#       python realtime_digit.py --lang ar                (Arabic-Indic digits, laptop camera)
#       python realtime_digit.py --lang ar --source video.mp4   (Arabic digits, a video file)

import argparse   # reads options typed after the script name in the terminal
import os         # lets us check that a file exists
import sys        # lets us check which operating system we are on
import time       # lets us measure time, needed for FPS
import cv2        # OpenCV: camera, drawing, windows
import torch      # PyTorch: runs the model
from model import LeNet5                                       # the SAME model for both languages; only the weights differ
# Both files define a function called `preprocess`, so we give each one its own name when importing.
from preprocessing import preprocess as preprocess_en          # English pipeline (MNIST normalisation 0.1307 / 0.3081)
from preprocessing_arabic import preprocess as preprocess_ar   # Arabic pipeline (MADBase normalisation 0.0777 / 0.2438)

# ---------- 1. Command-line options: camera or video file, and the language ----------
parser = argparse.ArgumentParser()
# default "0" means camera number 0 (the default laptop camera)
parser.add_argument("--source", default="0", help="camera index (0) or path to a video file")
# choices= makes argparse refuse any other value and print a clear message.
# default="en" keeps the old behaviour when the option is not given.
parser.add_argument("--lang", default="en", choices=["en", "ar"],
                    help="en = English digits (MNIST), ar = Arabic-Indic digits (MADBase)")
args = parser.parse_args()

# Everything that depends on the language is kept in ONE place:
# the name shown on screen, the preprocessing function and the weights file.
SETTINGS = {
    "en": {"name": "English", "preprocess": preprocess_en, "weights": "lenet_mnist.pt"},
    "ar": {"name": "Arabic",  "preprocess": preprocess_ar, "weights": "lenet_madbase.pt"},
}
cfg = SETTINGS[args.lang]            # the settings of the language the user asked for
preprocess = cfg["preprocess"]       # from here on the loop just calls preprocess(), whichever language it is
WINDOW = f"Digit recognition ({cfg['name']})"                    # title of the main window
MODEL_WINDOW = f"Model view ({cfg['name']}, 32x32 input, enlarged)"   # title of the second window

# If the text is only digits (like "0"), it is a camera number; otherwise it is a file path.
source = int(args.source) if args.source.isdigit() else args.source

# ---------- 2. Open the camera or video ----------
if isinstance(source, int):
    # On Windows we use the DirectShow driver (CAP_DSHOW), which fixed the capture problem.
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
else:
    cap = cv2.VideoCapture(source)   # a video file needs no special driver
if not cap.isOpened():
    raise SystemExit(f"Could not open source: {args.source}")

# ---------- 3. Load the trained model ONCE (the weights of the chosen language) ----------
if not os.path.exists(cfg["weights"]):    # a clear message instead of a long error text
    raise SystemExit(f"Weights file {cfg['weights']} not found. Put it in the same folder as this script.")
model = LeNet5()   # create the empty model (same structure as in training)
# Load the learned weights. map_location="cpu" lets this work on a laptop without a GPU.
model.load_state_dict(torch.load(cfg["weights"], map_location="cpu"))
model.eval()       # evaluation mode: we are predicting, not training

# ---------- 4. Variables used inside the loop ----------
prev_time = time.time()   # time when the previous frame finished
fps = 0.0                 # smoothed frames-per-second value
shots = 0                 # number of screenshots saved (key 's')

while True:
    ok, frame = cap.read()   # read the next frame
    if not ok:               # no frame: the video ended or the camera failed
        break

    # ---- Step 1: compute the centre square box (same formula as capture.py) ----
    h, w = frame.shape[:2]
    side = int(min(h, w) * 0.6)     # box side = 60% of the shorter frame dimension
    x1 = (w - side) // 2            # left edge so the box is horizontally centred
    y1 = (h - side) // 2            # top edge so the box is vertically centred
    x2, y2 = x1 + side, y1 + side   # right and bottom edges

    # ---- Step 2: crop the box and preprocess it with the chosen language's pipeline ----
    roi = frame[y1:y2, x1:x2]       # rows first (y), then columns (x)
    # We do this BEFORE drawing anything, so the green box is never inside the crop.
    tensor, steps = preprocess(roi, return_steps=True)

    # ---- Step 3: run the model (only if a digit was found) ----
    if steps["bbox"] is None:       # preprocess found nothing in the box
        label = "no digit"
        color = (0, 0, 255)         # red (colours are BGR in OpenCV)
    else:
        with torch.no_grad():       # no gradient bookkeeping: faster, less memory
            logits = model(tensor)                      # 10 raw scores
            probs = torch.softmax(logits, dim=1)[0]     # probabilities that sum to 1; [0] removes the batch dimension
            conf, pred = probs.max(0)                   # highest probability and the digit it belongs to
        if conf.item() < 0.5:           # below 50% the model is basically guessing
            label = "unsure"
            color = (0, 165, 255)       # orange
        else:
            label = f"Digit: {pred.item()}  ({conf.item() * 100:.1f}%)"
            color = (0, 255, 0) if conf.item() >= 0.8 else (0, 165, 255)

    # ---- FPS: 1 / (seconds since the previous frame), smoothed ----
    now = time.time()
    dt = now - prev_time            # seconds this frame took
    prev_time = now
    if dt > 0:                      # avoid dividing by zero
        inst = 1.0 / dt             # instant FPS (jumpy from frame to frame)
        # Smoothing: keep 90% of the old value and add 10% of the new one,
        # so the number on screen is readable instead of flickering.
        fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

    # ---- Step 4a: draw the results on the main frame ----
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)   # the box
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # The language tag. OpenCV's built-in font cannot draw Arabic glyphs, so the predicted digit is
    # shown as a Western numeral and this line tells the viewer which digit system is being read.
    cv2.putText(frame, f"Mode: {cfg['name']} digits", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "q = quit   s = screenshot", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow(WINDOW, frame)

    # ---- Step 4b: second window = what the network actually receives ----
    # padded32 is the 32x32 image (digit centred in 28x28, plus the 2-pixel border).
    # It is enlarged 10x. INTER_NEAREST keeps hard square pixels instead of smoothing
    # them, so you see the real pixels the model sees.
    model_view = cv2.resize(steps["padded32"], (320, 320), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(MODEL_WINDOW, model_view)

    # ---- Step 5: keyboard ----
    key = cv2.waitKey(1) & 0xFF      # wait 1 ms; & 0xFF keeps only the key code
    if key == ord("q"):              # q: quit
        break
    elif key == ord("s"):            # s: save a screenshot for the report
        shots += 1
        name = f"screenshot_{args.lang}_{shots}.png"   # the language is in the name, so nothing gets overwritten
        cv2.imwrite(name, frame)
        print("saved", name)

cap.release()            # free the camera
cv2.destroyAllWindows()  # close all windows
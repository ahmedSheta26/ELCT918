# ELCT918 – Lab 3: Real-time handwritten digit recognition (English and Arabic-Indic digits)

A LeNet-5 convolutional neural network is trained on **MNIST** (English digits 0–9) and on **MADBase** (Arabic-Indic digits ٠–٩).
A preprocessing pipeline turns a camera picture of a digit written in dark ink on white paper into an MNIST-like image.
A real-time application then shows the predicted digit, its confidence and the frames per second (FPS) on live video.

## 1. Contents of this folder

| File | What it is |
|---|---|
| `model.py` | The LeNet-5 definition (1 input channel, 32×32 input, 10 classes). **The same file is used for training in Colab and for inference locally**, otherwise the weights would not load. |
| `lenet_mnist.pt` | Trained English weights (PyTorch `state_dict`, about 0.25 MB). |
| `lenet_madbase.pt` | Trained Arabic weights (about 0.25 MB). |
| `lenet_mnist_info.json`, `lenet_madbase_info.json` | Test accuracy, parameter count, time per epoch, **normalisation values**, seed and training settings of each run. |
| `lenet_mnist_history.json`, `lenet_madbase_history.json` | Loss and accuracy of every epoch (used for the training curves). |
| `preprocessing.py` | The English preprocessing function `preprocess(roi)`. |
| `preprocessing_arabic.py` | The Arabic preprocessing function `preprocess(roi)`. |
| `realtime_digit2.py` | The real-time application (English and Arabic, camera or video file). |
| `capture.py` | Saves the content of the centre box of your camera as an image (for testing the preprocessing). |
| `make_figure.py`, `make_figure_arabic.py` | Draw the output of every preprocessing step for one image and print the model's prediction. |

No training happens inside the real-time application. It only loads the saved weights.

## 2. Installation

You need Python 3 (tested with Python 3.14 on Windows 11) and a camera for the live mode.

**Step 1: open a terminal in this folder.** The folder name contains a space, so use quotes:

```
cd "Lab3 assignment"
```

**Step 2 (recommended): create a virtual environment.** This keeps the project's packages separate from the rest of your computer.

```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
```

**Step 3: install the dependencies.**

```
pip install opencv-python numpy torch matplotlib
```

| Package | Used for |
|---|---|
| `torch` | Running LeNet-5 |
| `opencv-python` | Camera, image processing, windows |
| `numpy` | Arrays of pixel values |
| `matplotlib` | The preprocessing figures |

Versions used during development: torch 2.14.1, opencv-python 5.0.0.93, numpy 2.5.3, matplotlib 3.11.2.

**Check the installation:**

```
python -c "import cv2, torch, numpy, matplotlib; print('OK', cv2.__version__, torch.__version__)"
```

## 3. How to run each script

Always run the scripts from inside this folder, because they look for the `.pt` weights files next to them.

### 3.1 Capture a test image (optional)

```
python capture.py
```

A window with a green square appears. Press **SPACE** to save only the content of the square as `digit_1.jpg`, `digit_2.jpg`, … and **q** to quit. Click on the window first, because key presses go to the selected window. To avoid overwriting images, change the `name = ...` line in the script (for example to `arabic_{count}.jpg`).

### 3.2 See the preprocessing steps and the prediction for one image

```
python make_figure.py digit_1.jpg            # English
python make_figure_arabic.py arabic_1.jpg    # Arabic
```

Each script prints the predicted digit and its confidence and saves a figure with every step (`preprocessing_steps.png` or `preprocessing_steps_arabic.png`). The Arabic script also prints what was detected (a normal digit or a dot-zero) and the measured stroke width.

### 3.3 Real-time recognition

```
python realtime_digit2.py                          # English digits, laptop camera
python realtime_digit2.py --lang ar                # Arabic-Indic digits, laptop camera
```

**Using a video file instead of the live camera:**

```
python realtime_digit2.py --source myvideo.mp4                  # English digits in a video
python realtime_digit2.py --lang ar --source myvideo.mp4        # Arabic digits in a video
```

**Using a different camera:** `python realtime_digit2.py --source 1` (the number is the camera index; 0 is the default).

| Option | Meaning |
|---|---|
| `--lang en` (default) or `--lang ar` | Which digit system to read. This selects the preprocessing pipeline **and** the weights file. |
| `--source 0` (default) | Camera index, or the path to a video file. |

What you see:

- **Main window:** the camera image with a square box in the centre (60% of the shorter side of the frame), the prediction (`Digit: 7 (91.3%)`), the FPS and the mode. The box is green when the confidence is at least 80%, orange between 50% and 80%, and the label says `unsure` below 50%. If no digit is found in the box it says `no digit` (red).
- **Model view window:** the 32×32 image that the network actually receives, enlarged 10 times.
- **Keys:** **q** quits, **s** saves a screenshot (`screenshot_en_1.png`, `screenshot_ar_1.png`, …). Click on the video window first.
- In Arabic mode the predicted digit is shown as a Western numeral (OpenCV's built-in font cannot draw Arabic glyphs), and the line `Mode: Arabic digits` tells which system is being read.
- For a video file, the digit has to appear **inside the centre box**. The program stops at the end of the video.

**Tips for good results:** a dark pen on plain white paper, even lighting without shadows, the digit large and inside the box without touching its edges, and nothing else in the box.

All runs use seed 42, 15 epochs, batch size 128, SGD with learning rate 0.01, momentum 0.9 and weight decay 5e-4, and a fixed 55,000 / 5,000 train / validation split.

## 4. How it works (short version)

**Preprocessing (`preprocess(roi)`), five steps:**

1. Convert to grayscale (OpenCV stores colours as BGR, so `COLOR_BGR2GRAY`) and apply a 5×5 Gaussian blur to reduce camera noise.
2. Threshold with Otsu's method and invert, so that the digit is white on black, as in the datasets.
3. Find the outlines with `findContours`, keep blobs that look like a digit (not too small, not too big, not touching the box edge), and crop the largest one.
4. Resize so that the larger side is exactly 20 pixels (keeping the aspect ratio), centre it in a 28×28 black image and pad it to 32×32.
5. Scale the values to [0, 1], subtract the mean, divide by the standard deviation and reshape to (1, 1, 32, 32).

**Differences between the English and the Arabic pipeline:**

| | English (`preprocessing.py`) | Arabic (`preprocessing_arabic.py`) |
|---|---|---|
| Normalisation (mean / std) | 0.1307 / 0.3081 (MNIST) | 0.0777 / 0.2438 (computed on the MADBase training images) |
| Stroke thickening | Fixed: 7% of the digit size | Measured: the stroke width of the camera digit is measured and thickened only up to MADBase's 8.6% of the digit size |
| Arabic zero (٠, a dot) | Not needed | A small blob is accepted as a dot-zero only if no normal digit is in the box, it is the only small blob, and it is round and filled (fill ratio ≥ 0.50, aspect ratio ≥ 0.55) |
| Enlarging a small crop | – | `INTER_LINEAR` (smooth), instead of `INTER_AREA`, which is meant for shrinking |
| Weights | `lenet_mnist.pt` | `lenet_madbase.pt` |

The values 0.1307 / 0.3081 and 0.0777 / 0.2438 are exactly the ones used during training. They must never be changed without retraining.

## 5. Results

| | MNIST (English) | MADBase (Arabic) |
|---|---|---|
| Test accuracy | 98.97% | 98.63% |
| Trainable parameters | 61,706 | 61,706 |
| Training time per epoch (Colab T4) | 12.59 s | 1.67 s |

## 6. Known limitations

- **Handwriting style and proportions:** each model only knows the style of its dataset. Digits with unusual proportions (for example a very long vertical stem) are shrunk so much that their details become tiny, and can be misread.
- **Pen thickness:** very thick marker strokes can close loops and fill the gaps in a digit.
- **One digit at a time**, held inside the centre box.
- **Lighting:** strong shadows across the paper can be mistaken for ink by the thresholding step.


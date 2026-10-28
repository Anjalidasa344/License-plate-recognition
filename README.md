\# Auto License Plate Recognition – Windows CPU Version



A Python-based vehicle license plate recognition project adapted to run on Windows using CPU.



\## Features



\- Vehicle number plate detection using an ONNX model.

\- Automatic number plate cropping.

\- Image enhancement for OCR.

\- Character recognition using EasyOCR.

\- Detection confidence and OCR confidence display.

\- Saves the processed output images.



\## Technologies Used



\- Python 3.11

\- OpenCV

\- ONNX Runtime

\- EasyOCR

\- PyTorch

\- NumPy



\## Run the Project



Install the required packages:



```bash

python -m pip install opencv-python onnxruntime numpy easyocr torch torchvision

```



Place the required ONNX plate-detection model in:



`networks/az\_plate/az\_plate\_ssdmobilenetv1.onnx`



Place the test image in the project folder as `test\_car.png`.



Run:



```bash

python full\_plate\_recognition.py

```



\## Output



The application saves:

\- `automatic\_plate\_crop.jpg`

\- `automatic\_enhanced\_plate.jpg`

\- `automatic\_recognition\_result.jpg`



\## Original Project and Attribution



This project is a Windows CPU adaptation inspired by:



https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano



The original project was designed for NVIDIA Jetson Nano. This adaptation uses EasyOCR for character recognition on Windows.



The original ONNX model files are not included in this repository. Obtain them from the original project only after checking the applicable permissions and licence.



\## Limitations



The original recognition models were developed for Vietnamese number plates. Recognition accuracy on Indian number plates or other formats is not guaranteed. Test with suitable local datasets before real-world use.



\## Project Status



Working prototype tested on a sample vehicle image using Windows CPU.


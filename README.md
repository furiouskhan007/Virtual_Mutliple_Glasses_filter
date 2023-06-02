# Virtual Mutliple Glasses filter Put on your Face

This is a python application that combine your face and a sunglasses image with [OpenCV](https://opencv.org/) and [Dlib](http://dlib.net/) library.
## Environment

- `Python >= 3.6`
  - `numpy==1.17.1`
  - `dlib==19.17.0`
  - `opencv-python==4.1.0.25`
- If you are missing the dlib shape predictor then Download the dlib landmark detector model file and unzip
  - `python assets/models/download_dlib_shape_predictor_68.py`
- If failed to install the libraries then try bellow commands
  - pip install opencv-python
  - pip install numpy
  - pip install cmake
  - pip install dlib "dlib-19.22.99-cp310-cp310-win_amd64.whl" [Dlib](https://github.com/datamagic2020/Install-dlib) make sure you downloaded the compitable version of dlib and paste into the main directory, also provide the correct path during dlib installation

## How to use

### Load images

```sh
# check on image
python glasses.py
```

### Using a webcam

```sh
python glasses_live.py
```

### trying random glasses once the its run press s to change filter

```sh
python glasses_Live_random.py 
```

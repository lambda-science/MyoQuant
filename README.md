![Twitter Follow](https://img.shields.io/twitter/follow/corentinm_py?style=social) ![Demo Version](https://img.shields.io/badge/Demo-https%3A%2F%2Flbgi.fr%2FMyoQuant%2F-success) ![Pypi verison](https://img.shields.io/pypi/v/myoquant) ![PyPi Python Version](https://img.shields.io/pypi/pyversions/myoquant) ![PyPi Format](https://img.shields.io/pypi/format/myoquant) ![GitHub last commit](https://img.shields.io/github/last-commit/lambda-science/MyoQuant) ![GitHub](https://img.shields.io/github/license/lambda-science/MyoQuant) 

# MyoQuant🔬

MyoQuant🔬 is a command line tool to quantify pathological feature in histology images.  
It is built using CellPose, Stardist, custom neural-network models and image analysis techniques to automatically analyze myopathy histology images. An online demo with a web interface is available at [https://lbgi.fr/MyoQuant/](https://lbgi.fr/MyoQuant/).

### **Warning:** This tool is still in alpha stage and might not work perfectly... yet.

## How to install

### Installing from PyPi

Using pip, you can simply install MyoQuant in a python environment with a simple: `pip install myoquant`

### Installing from source

1. Clone this repository using `git clone https://github.com/lambda-science/MyoQuant.git`
2. Create a virtual environment by using `python -m venv .venv`
3. Activate the venv by using `source .venv/bin/activate`
4. Install MyoQuant by using `pip install -e .`

You are ready to go !

## How to Use

To use the command-line tool, first activate your venv `source .venv/bin/activate`  
Then you can perform SDH or HE analysis. You can use the command `myoquant --help` to list available commands.

- **For SDH Image Analysis** the command is:  
  `myoquant sdh_analysis IMAGE_PATH`  
  Don't forget to run `myoquant sdh_analysis --help` for information about options.
- **For HE Image Analysis** the command is:  
  `myoquant he_analysis IMAGE_PATH`  
   Don't forget to run `myoquant he_analysis --help` for information about options.

_If you're running into an issue such as `myoquant: command not found` please check if you activated your virtual environment with the package installed. And also you can try to run it with the full command: `python -m myoquant sdh_analysis --help`_

## Examples

For HE Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_he.jpg)  
For SDH Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_sdh.jpg)

1. Example of successful SDH analysis with: `myoquant sdh_analysis sample_sdh.jpg`

![image](https://user-images.githubusercontent.com/20109584/198278737-24d69f61-058e-4a41-a463-68900a0dcbb6.png)

2. Example of successful HE analysis with: `myoquant he_analysis sample_he.jpg`

![image](https://user-images.githubusercontent.com/20109584/198280366-1cb424f5-50af-45f9-99d1-34e191fb2e20.png)

## Who and how

- Creator and Maintainer: [Corentin Meyer, 3rd year PhD Student in the CSTB Team, ICube — CNRS — Unistra](https://lambda-science.github.io/)
- The source code for this application is available [HERE](https://github.com/lambda-science/MyoQuant)

## Advanced information

For the SDH Analysis our custom model will be downloaded and placed inside the myoquant package directory. You can also download it manually here: [https://lbgi.fr/~meyer/SDH_models/model.h5](https://lbgi.fr/~meyer/SDH_models/model.h5) and then you can place it in the directory of your choice and provide the path to the model file using:  
`myoquant sdh_analysis IMAGE_PATH --model_path /path/to/model.h5`

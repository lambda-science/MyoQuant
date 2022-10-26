# MyoQuant

MyoQuant command line tool to quantifying pathological feature in histology images.  
It is built using CellPose, Stardist and custom models and image analysis techniques to automatically analyze myopathy histology images. An online demo with a web interface is availiable at [https://lbgi.fr/MyoQuant/](https://lbgi.fr/MyoQuant/).

## How to install

### Installing from source

1. Clone this repository using `git clone https://github.com/lambda-science/MyoQuant.git`
2. Create a virtual environnement by using `python -m venv .venv`
3. Activate the venv by using `source .venv/bin/activate`
4. Install MyoQuant by using `pip install -e .`

You are ready to go !

### Installing from PyPi

I am currently working on a PyPi release to do a simple `pip install myoquant` for installation.  
You might try it it and see if it has been finally released !

## How to Use

To use the command-line tool, first activate your venv `source .venv/bin/activate`  
Then you can perform SDH or HE analysis.

- **For SDH Image Analysis** run `python -m myoquant sdh_analysis IMAGE_PATH`. Don't forget to run `python -m myoquant sdh_analysis --help` for information about options.
- **For HE Image Analysis** run `python -m myoquant he_analysis IMAGE_PATH`. Don't forget to run `python -m myoquant he_analysis --help` for information about options.

For HE Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_he.jpg)  
For SDH Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_sdh.jpg)

## Who and how

- Creator and Maintainer: [Corentin Meyer, 3rd year PhD Student in the CSTB Team, ICube — CNRS — Unistra](https://lambda-science.github.io/)
- The source code for this application is available [HERE](https://github.com/lambda-science/MyoQuant)

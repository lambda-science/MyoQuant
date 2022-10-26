# MyoQuant

MyoQuant is a web application for quantifying the number of cells in a histological image.  
It is built using CellPose, Stardist and custom models and image analysis techniques to automatically analyze myopathy histology images.  
This web application is intended for demonstration purposes only.

## How to install or deploy

A Streamlit cloud demo instance should be deployed at https://lbgi.fr/MyoQuant/. I am currently working on proper docker images and tutorial to deploy the application. Meanwhile you can still use the following instructions:

### Docker

You can build the docker image by running `docker build -t streamlit .` and launch the container using `docker run -p 8501:8501 streamlit`.

### Non-Docker

If you do not want to use Docker you can install the poetry package in a miniconda (python 3.9) base env, run `poetry install` to install the python env, activate the env with `poetry shell` and launch the app by running `streamlit run Home.py`.

### Deploy on Google Colab for GPU

As this application uses various deep-learning model, you could benefit from using a deployment solution that provides a GPU.  
To do so, you can leverage Google Colab free GPU to boost this Streamlit application.  
To run this app on Google Colab, simply clone the notebook called `google_colab_deploy.ipynb` into Colab and run the four cells. It will automatically download the latest code version, install dependencies and run the app. A link will appear in the output of the lat cell with a structure like `https://word1-word2-try-01-234-567-890.loca.lt`. Click it and the click continue and you’re ready to use the app!

## How to Use

A Streamlit cloud demo instance should be deployed at https://lbgi.fr/MyoQuant/. I am currently working on docker images and tutorial to deploy the application.  
Once on the demo, click on the corresponding staining analysis on the sidebar, and upload your histology image. Results will be displayed in the main area automatically.  
For HE Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_he.jpg)  
For SDH Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_sdh.jpg)

## Who and how

- Creator and Maintainer: [Corentin Meyer, 3rd year PhD Student in the CSTB Team, ICube—CNRS—Unistra] (https://lambda-science.github.io/)
- The source code for this application is available [HERE] (https://github.com/lambda-science/MyoQuant)

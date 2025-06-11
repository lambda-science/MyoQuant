---
title: MyoQuant-Streamlit🔬
emoji: 🔬
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 8501
license: agpl-3.0
python_version: 3.12.11
pinned: true
header: mini
short_description: Quantify pathological features in histology images
models: corentinm7/MyoQuant-SDH-Model
datasets: corentinm7/MyoQuant-SDH-Data
tags:
  - streamlit
  - myology
  - biology
  - histology
  - muscle
  - cells
  - fibers
  - myopathy
  - SDH
  - myoquant
preload_from_hub: corentinm7/MyoQuant-SDH-Model
---

![Twitter Follow](https://img.shields.io/twitter/follow/corentinm_py?style=social) ![PyPi](https://img.shields.io/badge/PyPi-https%3A%2F%2Fpypi.org%2Fproject%2Fmyoquant%2F-blueviolet) ![Pypi verison](https://img.shields.io/pypi/v/myoquant)

# MyoQuant-Streamlit🔬: a demo web interface for the MyoQuant tool.

## Please refer to the [MyoQuant GitHub Repository](https://github.com/lambda-science/MyoQuant) or the [MyoQuant PyPi repository](https://pypi.org/project/myoquant/) for full documentation.

MyoQuant-Streamlit🔬 is a demo web interface to showcase the usage of MyoQuant.

<p align="center">
  <img src="https://i.imgur.com/mzALgZL.png" alt="MyoQuant Banner" style="border-radius: 25px;" />
</p>

<p align="center">
  <img src="https://i.imgur.com/FxpFUT3.png" alt="MyoQuant Illustration" style="border-radius: 25px;" />
</p>

MyoQuant🔬 is a command-line tool to automatically quantify pathological features in muscle fiber histology images.  
It is built using CellPose, Stardist, custom neural-network models and image analysis techniques to automatically analyze myopathy histology images. Currently MyoQuant is capable of quantifying centralization of nuclei in muscle fiber with HE staining, anomaly in the mitochondria distribution in muscle fibers with SDH staining and the number of type 1 muscle fiber vs type 2 muscle fiber with ATP staining.  
This web application is intended for demonstration purposes only.

## How to install or deploy the interface

The demo version is deployed at [https://huggingface.co/spaces/corentinm7/MyoQuant](https://huggingface.co/spaces/corentinm7/MyoQuant). You can deploy your own demo version using Docker or your own python environment.

### Docker

You can build & run the docker image by running `docker build -t myostreamlit:latest . && docker run -p 8501:8501 myostreamlit:latest`

### Non-Docker

If you do not want to use Docker you can install package using for example [UV](https://github.com/astral-sh/uv). Run `uv sync` to create the python environnement and then run: `uv run streamlit run src/myoquant/streamlit/run.py` or `uv run streamlit run run.py` if you only clone the HuggingFace space repository and not the full MyoQuant package.

## How to Use

Once on the demo, click on the corresponding staining analysis on the sidebar, and upload your histology image. Results will be displayed in the main area automatically.  
For all analysis you can press the "Load Default File" to load a sample image to try the tool.

## Contact

Creator and Maintainer: [**Corentin Meyer**, PhD in Biomedical AI](https://cmeyer.fr/) <contact@cmeyer.fr>. The source code for MyoQuant is available [HERE](https://github.com/lambda-science/MyoQuant).

## Citing MyoQuant🔬

[placeholder]

## Partners

<p align="center">
  <img src="https://i.imgur.com/m5OGthE.png" alt="Partner Banner" style="border-radius: 25px;" />
</p>

MyoQuant is born within the collaboration between the [CSTB Team @ ICube](https://cstb.icube.unistra.fr/en/index.php/Home) led by Julie D. Thompson, the [Morphological Unit of the Institute of Myology of Paris](https://www.institut-myologie.org/en/recherche-2/neuromuscular-investigation-center/morphological-unit/) led by Teresinha Evangelista, the [imagery platform MyoImage of Center of Research in Myology](https://recherche-myologie.fr/technologies/myoimage/) led by Bruno Cadot, [the photonic microscopy platform of the IGMBC](https://www.igbmc.fr/en/plateformes-technologiques/photonic-microscopy) led by Bertrand Vernay and the [Pathophysiology of neuromuscular diseases team @ IGBMC](https://www.igbmc.fr/en/igbmc/a-propos-de-ligbmc/directory/jocelyn-laporte) led by Jocelyn Laporte

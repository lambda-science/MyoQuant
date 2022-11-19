![Twitter Follow](https://img.shields.io/twitter/follow/corentinm_py?style=social) ![Demo Version](https://img.shields.io/badge/Demo-https%3A%2F%2Flbgi.fr%2FMyoQuant%2F-9cf) ![PyPi](https://img.shields.io/badge/PyPi-https%3A%2F%2Fpypi.org%2Fproject%2Fmyoquant%2F-blueviolet) ![Pypi verison](https://img.shields.io/pypi/v/myoquant) ![PyPi Python Version](https://img.shields.io/pypi/pyversions/myoquant) ![PyPi Format](https://img.shields.io/pypi/format/myoquant) ![GitHub last commit](https://img.shields.io/github/last-commit/lambda-science/MyoQuant) ![GitHub](https://img.shields.io/github/license/lambda-science/MyoQuant)

# MyoQuant🔬: a tool to automatically quantify pathological features in muscle fiber histology images

<p align="center">
  <img src="https://i.imgur.com/mzALgZL.png" alt="MyoQuant Banner" style="border-radius: 25px;" />
</p>

MyoQuant🔬 is a command-line tool to automatically quantify pathological features in muscle fiber histology images.  
It is built using CellPose, Stardist, custom neural-network models and image analysis techniques to automatically analyze myopathy histology images. Currently MyoQuant is capable of quantifying centralization of nuclei in muscle fiber with HE staining and anomaly in the mitochondria distribution in muscle fibers with SDH staining.

An online demo with a web interface is available at [https://lbgi.fr/MyoQuant/](https://lbgi.fr/MyoQuant/). This project is free and open-source under the AGPL license, feel free to fork and contribute to the development.

#### _Warning: This tool is still in early phases and active development._

## How to install

### Installing from PyPi (Preferred)

**MyoQuant package is officially available on PyPi (pip) repository. [https://pypi.org/project/myoquant/](https://pypi.org/project/myoquant/) ![Pypi verison](https://img.shields.io/pypi/v/myoquant)**

Using pip, you can simply install MyoQuant in a python environment with a simple: `pip install myoquant`

### Installing from sources (Developers)

1. Clone this repository using `git clone https://github.com/lambda-science/MyoQuant.git`
2. Create a virtual environment by using `python -m venv .venv`
3. Activate the venv by using `source .venv/bin/activate`
4. Install MyoQuant by using `pip install -e .`

## How to Use

To use the command-line tool, first activate your venv in which MyoQuant is installed: `source .venv/bin/activate`  
Then you can perform SDH or HE analysis. You can use the command `myoquant --help` to list available commands.

## 💡Full command documentation is avaliable here: [CLI Documentation](https://github.com/lambda-science/MyoQuant/blob/main/CLI_Documentation.md)

- **For SDH Image Analysis** the command is:  
  `myoquant sdh-analysis IMAGE_PATH`  
  Don't forget to run `myoquant sdh-analysis --help` for information about options.
- **For HE Image Analysis** the command is:  
  `myoquant he-analysis IMAGE_PATH`  
   Don't forget to run `myoquant he-analysis --help` for information about options.

_If you're running into an issue such as `myoquant: command not found` please check if you activated your virtual environment with the package installed. And also you can try to run it with the full command: `python -m myoquant sdh-analysis --help`_

## Contact

Creator and Maintainer: [**Corentin Meyer**, 3rd year PhD Student in the CSTB Team, ICube — CNRS — Unistra](https://cmeyer.fr) <corentin.meyer@etu.unistra.fr>

## Citing MyoQuant🔬

[placeholder]

## Examples

For HE Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_he.jpg)  
For SDH Staining analysis, you can download this sample image: [HERE](https://www.lbgi.fr/~meyer/SDH_models/sample_sdh.jpg)

1. Example of successful SDH analysis with: `myoquant sdh-analysis sample_sdh.jpg`

![image](https://user-images.githubusercontent.com/20109584/198278737-24d69f61-058e-4a41-a463-68900a0dcbb6.png)

2. Example of successful HE analysis with: `myoquant he-analysis sample_he.jpg`

![image](https://user-images.githubusercontent.com/20109584/198280366-1cb424f5-50af-45f9-99d1-34e191fb2e20.png)

## Advanced information

### Model path and manual download

For the SDH Analysis our custom model will be downloaded and placed inside the myoquant package directory. You can also download it manually here: [https://lbgi.fr/~meyer/SDH_models/model.h5](https://lbgi.fr/~meyer/SDH_models/model.h5) and then you can place it in the directory of your choice and provide the path to the model file using:  
`myoquant sdh-analysis IMAGE_PATH --model_path /path/to/model.h5`

### HuggingFace🤗 repositories for Data and Model

In a effort to push for open-science, MyoQuant [SDH dataset](https://huggingface.co/datasets/corentinm7/MyoQuant-SDH-Data) and [model](https://huggingface.co/corentinm7/MyoQuant-SDH-Model) and availiable on HuggingFace🤗

## Partners

<p align="center">
  <img src="https://i.imgur.com/m5OGthE.png" alt="Partner Banner" style="border-radius: 25px;" />
</p>

MyoQuant is born within the collaboration between the [CSTB Team @ ICube](https://cstb.icube.unistra.fr/en/index.php/Home) led by Julie D. Thompson, the [Morphological Unit of the Institute of Myology of Paris](https://www.institut-myologie.org/en/recherche-2/neuromuscular-investigation-center/morphological-unit/) led by Teresinha Evangelista, the [imagery platform MyoImage of Center of Research in Myology](https://recherche-myologie.fr/technologies/myoimage/) led by Bruno Cadot, [the photonic microscopy platform of the IGMBC](https://www.igbmc.fr/en/plateformes-technologiques/photonic-microscopy) led by Bertrand Vernay and the [Pathophysiology of neuromuscular diseases team @ IGBMC](https://www.igbmc.fr/en/igbmc/a-propos-de-ligbmc/directory/jocelyn-laporte) led by Jocelyn Laporte

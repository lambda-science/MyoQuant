import streamlit as st
from streamlit.components.v1 import html

st.write("# MyoQuant-Streamlit🔬")

st.sidebar.success("Select the corresponding staining analysis above.")

st.markdown(
    """
## MyoQuant-Streamlit🔬 is a demo web interface to showcase the usage of [MyoQuant](https://github.com/lambda-science/MyoQuant). 

![MyoQuant Banner](https://i.imgur.com/mzALgZL.png)
![MyoQuant Illustration](https://i.imgur.com/FxpFUT3.png)

[MyoQuant🔬 is a command-line tool to automatically quantify pathological features in muscle fiber histology images.](https://github.com/lambda-science/MyoQuant)  
It is built using CellPose, Stardist, custom neural-network models and image analysis techniques to automatically analyze myopathy histology images. Currently MyoQuant is capable of quantifying centralization of nuclei in muscle fiber with HE staining and anomaly in the mitochondria distribution in muscle fibers with SDH staining.  
This web application is intended for demonstration purposes only.  

## How to Use

Once on the demo, click on the corresponding staining analysis on the sidebar, and upload your histology image. Results will be displayed in the main area automatically.  
For all analysis you can press the "Load Default File" to load a sample image to try the tool.

## Contact
Creator and Maintainer: [**Corentin Meyer**, 3rd year PhD Student in the CSTB Team, ICube — CNRS — Unistra](https://lambda-science.github.io/)  <corentin.meyer@etu.unistra.fr>  
The source code for MyoQuant is available [HERE](https://github.com/lambda-science/MyoQuant).  

## Partners

![Partners Banner](https://i.imgur.com/Xk9wBFQ.png)  
MyoQuant is born within the collaboration between the [CSTB Team @ ICube](https://cstb.icube.unistra.fr/en/index.php/Home) led by Julie D. Thompson, the [Morphological Unit of the Institute of Myology of Paris](https://www.institut-myologie.org/en/recherche-2/neuromuscular-investigation-center/morphological-unit/) led by Teresinha Evangelista, the [imagery platform MyoImage of Center of Research in Myology](https://recherche-myologie.fr/technologies/myoimage/) led by Bruno Cadot, [the photonic microscopy platform of the IGMBC](https://www.igbmc.fr/en/plateformes-technologiques/photonic-microscopy) led by Bertrand Vernay and the [Pathophysiology of neuromuscular diseases team @ IGBMC](https://www.igbmc.fr/en/igbmc/a-propos-de-ligbmc/directory/jocelyn-laporte) led by Jocelyn Laporte.  
"""
)
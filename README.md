# Cephalometric-Landmark_HRnet-Webapp

This repository implements an automated cephalometric landmark detection system for orthodontic and maxillofacial analysis. The system is built around a High-Resolution Network (HRNet) model trained on preprocessed cephalograms (resized to 256×256 pixels), achieving a Mean Euclidean Distance (MED) of 41.86 pixels (~5.3 mm)—demonstrating competitive accuracy with commercial tools such as Vistadent OC.

The trained model is integrated into a Flask-based web application that supports:  
- Upload and automatic landmark detection on lateral cephalograms  
- Manual refinement of predicted landmark positions  
- Export of results (annotated image and landmark coordinates in CSV/JSON)

The solution aims to enhance clinical efficiency through faster, accessible, and transparent cephalometric analysis—offering a viable open-source alternative to proprietary software.

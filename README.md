# AI_services_for_Kazakh_Language
This repository provides a user-friendly API that empowers developers to integrate cutting-edge computer vision (CV) and natural language processing (NLP) models specifically designed for the Kazakh language.


# TODOs
- [ ] Optimal GPU utilization
- [ ] OCR, Image Captioning
- [ ] Remove dependencies from external libraries 
- [ ] ONNXRuntime for all models

# Getting Started

## Prerequisites:
1) Python 3.8.20 (or compatible version) installed on your system.
2) pip package manager (usually comes bundled with Python).

## Installation:
1) Clone this repository.
2) Navigate to the project directory: cd AI_services_for_Kazakh_Language
3) Create conda environment with all the dependencies: conda env create --name my_env -f environment.yml

## Running the API:
1) Start the API server: fastapi dev main.py (run this command in a separate terminal window)
2) Expose the API using ngrok (optional, for public access):
    Install and
    Run ngrok (ex. http --domain=driven-cricket-publicly.ngrok-free.app 8000) in another terminal window.

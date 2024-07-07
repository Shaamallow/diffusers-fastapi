# FastAPI - Diffusers Backend

<p align="center" style="padding: 20px;">
    <a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">
	    <img alt="Base Model" src='https://img.shields.io/badge/%F0%9F%A4%97-SDXL%20Model-yellow'/>
	</a>
    <a href="https://fastapi.tiangolo.com/">
	    <img alt="Base Model" src='https://img.shields.io/badge/API-Fast%20Api-rgb(0%2C%20118%2C%20106)'/>
	</a>
    <a href="https://github.com/psf/black">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

This repo is meant to be a very simple diffusers backend with fastapi
Serve a SDXL model with a Depth ControlNet to generate textures

## Installation

It is advised to use a python manager such as `conda` `pyenv` or `mamba` (_conda but faster_)

Install the dependencies using :

```bash
pip install -r requirements.txt
```

## Usage

Launch a uvicorn server with the following command:

```bash
fastapi run main:app
```

You can test the backend by running the test scripts :

```bash
python tests/backend_test.py
```

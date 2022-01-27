# Adversarially Training of Autoencoders for Unsupervised Anomaly Segmentation 

This repo contains a method to adversarially train Autoencoders (AEs) (Dense AE and Variational AE) for unsupervised anomaly segmentation task on brain MR Images.

* [Autoencoders_Adversarial_Training_UAD](#Autoencoders_Adversarial_Training_UAD)
  * [Requirements](#requirements)
  * [Folder Structure](#folder-structure)
  * [Usage](#usage)
      * [CLI-Usage](#cli-usage)
      * [Google Colab Usage](#google-colab-usage)
  * [Disclaimer](#disclaimer)
  * [Reference](#reference)
  * [License](#license)
    
<!-- /code_chunk_output -->


## Tags
<code>Adversarial Training</code>, <code>Autoencoders</code>, <code>Anomaly Segmentation</code>, <code>Unsupervised</code>, <code>Brain MR Images</code>, <code>TensorFlow</code>, <code>Keras</code>

## Requirements
* <code>Python >= 3.6</code>

All packages used in this repository are listed in [requirements.txt](https://github.com/ahmedgh970/Autoencoders_Adversarial_Training_UAD/requirements.txt).
To install those, run:
```
pip3 install -r requirements.txt
```

## Folder Structure
  ```
  Autoencoders_Adversarial_Training_UAD/
  │
  ├── Autoencoders_Adversarial_Training_UAD.ipynb - Jupyter notebook to work on Google Colab
  │
  ├── data/  - Data folder
  │   ├── data_descriptions.txt
  │   └── ...
  │
  ├── models/  - Models defining and training
  │   └── Autoencoders/
  │       └── DCAE.py
  │   └── Latent Variable models/
  │       └── VAE.py
  │
  ├── saved/  - Saving folder
  │   └── ...
  │
  └── scripts/  - Utility scripts for evaluation and adversarial crafting
      ├── adversarial_crafting.py
      ├── eval_brats.py
      ├── eval_mslub.py
      ├── ...
      └── utils.py
  ```


## Usage
All the preprocessed datasets that we utilized are available on this drive link: https://drive.google.com/file/d/11Bj7ATQtxLt7PyL3fqyyeXqNNrRqgS9K/view?usp=sharing

### CLI Usage
Every model can be trained and evaluated individually using the scripts which are provided in the `models/*` and `scripts/*` folders.

### Google Colab Usage
Training can be started by importing `Autoencoders_Adversarial_Training_UAD.ipynb` in [Google Colab](http://colab.research.google.com).
This github repository is linked and can directly loaded into the notebook. However, the datasets have to be stored so that Google Colab can access them. 
Either uploading by a zip-file or uploading it to Google Drive and mounting the drive.

## Adversarial Robustness Toolbox Installation
The most recent version of ART can be downloaded or cloned from this repository:
```
git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox
```
Install ART with the following command from the project folder adversarial-robustness-toolbox:
Using pip:
```
pip install .
```

## Disclaimer
Please do not hesitate to open an issue to inform of any problem you may find within this repository.


## Reference
For more details about the unsupervised anomaly segmentation method, you can find our published paper on `MIDL 2022 Conference` [Transformers for Unsupervised Anomaly Segmentation in Brain MR Images](https://openreview.net/forum?id=B_3vXI3Tcz&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DMIDL.io%2F2022%2FConference%2FAuthors%23your-submissions)). 
```
@article{ahghorbe2022,
  title = {Transformer based Models for Unsupervised Anomaly Segmentation in Brain MR Images},
  author = {Ghorbel, Ahmed and Aldahdooh, Ahmed and Hamidouche, Wassim and Albarqouni, Shadi},
  booktitle={Medical Imaging with Deep Learning},
  year = {2022}
}
```


For more details about the adversarial training method, you can read the paper on [ARAE: Adversarially robust training of autoencoders improves novelty detection](https://www.sciencedirect.com/science/article/pii/S0893608021003646).
```
@article{salehi2021arae,
  title={Arae: Adversarially robust training of autoencoders improves novelty detection},
  author={Salehi, Mohammadreza and Arya, Atrin and Pajoum, Barbod and Otoofi, Mohammad and Shaeiri, Amirreza and Rohban, Mohammad Hossein and Rabiee, Hamid R},
  journal={Neural Networks},
  volume={144},
  pages={726--736},
  year={2021},
  publisher={Elsevier}
}
```


## License
This project is licensed under the GNU General Public License v3.0. See LICENSE for more details

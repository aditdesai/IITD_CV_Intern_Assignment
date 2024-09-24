### CV_Intern_Visionlab_IITD_Assignment_Sep_2024

This repo contains the code for the setup, validation, analysis and fine tuning of the DINO object detection model (https://github.com/IDEA-Research/DINO). Note that any changes made to the original code (config files) are visible in the fork of the original repo which you can find here: https://github.com/aditdesai/DINO

Report can be found here: [IITD_CV_Intern_Report.pdf](IITD_CV_Intern_Report.pdf)

The code was run and tested on a Kaggle notebook with a P100 GPU.

## Run

- Download the pre-trained model weights used by me from here: https://drive.google.com/file/d/1AwUn5EebmmLBo7njjW_Ng1q9zDrqkNbB/view?usp=drive_link
- Download the fine-tuned model weights from here: https://drive.google.com/file/d/11iNP4Qc_A0ub43b-IoxLeFzRznZDjrKE/view?usp=sharing
- Visualize the dataset with the ground truth boxes by running:

```sh
python vis.py
```

- For inference and visualization, a jupyter [notebook](eval-vis.ipynb) is provided (Paths to model and dataset will need to be set manually)

- For fine tuning, a jupyter [notebook](fine-tune.ipynb) is provided as well (Paths to model and dataset will need to be set manually)


## Results

Visualizations, analysis and loss plots can be found here in the report: [IITD_CV_Intern_Report.pdf](IITD_CV_Intern_Report.pdf)
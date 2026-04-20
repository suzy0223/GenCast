# GenCast: Traffic Forecasting for Unobserved Regions (official code)

🚀 AAAI 2026 

### What problem?
Forecast traffic in regions WITHOUT sensors

### Why hard?
No historical observations → generalisation issue

### What we do?
- Physics-informed learning
- Weather-traffic fusion
- Spatial grouping

### Result
↓ error 3.1%, ↑ R² 125%

### Quick start
This code is based on our previous work STSM [STSM Code](https://github.com/suzy0223/STSM).
Our full paper is available at [paper](https://arxiv.org/abs/2508.08947).

## Requirements
-pytorch
-pandas
-numpy
-tables
-CUDA/12.5.1

The details are in the requirement.txt

## Dataset
Google Drive: https://drive.google.com/drive/folders/1_imrTikGhIbZIyrRynG4bXhkaZA9xQgE?usp=share_link

Baidu Drive: https://pan.baidu.com/s/1DtyDp4stCKQ_P-4nox1O8A 提取码: cast 
Due to the dataset is large, we will upload it to google drive and baiduyun for sharing. Or you can download traffic data from STSM and weather data from ERA5.

Putting Dataset dir under the GenCast dir.


## Train the model
go to dir GenCast-L or GenCast-H

chmod +x ./metr.sh
./metr.sh

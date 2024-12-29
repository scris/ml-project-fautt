# B-FAUTT: The Boosting-led First-AUdio-Then-Text Approach for Deceptive Story Detection

This repository contains code for CBU5201 Machine Learning 2024-25 Deceptive Story Detection Mini Project.

## Description

- preprocess.ipynb: For doing preprocessing
- main.ipynb: The main file for implementation and debugging
- CBU5201_miniproject.ipynb: The submitted notebook file

## Initialization

To use this repository, you need to:

1. Download the dataset from https://github.com/CBU5201Datasets/Deception and place the `CBU0521DD_stories/` folder of it into `dataset/stories/`.
2. Create a `sensitive.py` with four variables: `xfyun_appid`, `xfyun_appkey` (from Xunfei / IFlyTEK, for ASR), `baidu_appid` and `baidu_appkey` (from Baidu Translate, for CN -> EN).

## License

This repository is licensed under Mozilla Public License Version 2.0. For more information, please see [this](https://www.mozilla.org/en-US/MPL/2.0/).
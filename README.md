# Towards Building Generalizable Models for Malware Detection
As malware evolves and adapts, traditional detection systems struggle to identify novel and unseen threats. 
This challenge highlights the critical need for building generalizable models that can effectively detect unknown malware types. 
In this paper, we propose meta-learning as a tool to explore the adaptability of malware detection systems. 
Our approach focuses on understanding how much model updating is required to extend detection capabilities to previously unseen malware samples. 
By leveraging meta-learning, we aim to identify the most useful data for building generalizable models, optimizing the trade-off between data efficiency and detection accuracy. 
Through this investigation, we seek to provide insights into creating more robust and adaptable malware detection systems capable of addressing the constantly evolving threat landscape. Our results suggest that, among three popular representations of malware data, the combination of static and dynamic analysis reports is the most helpful in building generalizable models.  

## Files
This repository contains three Colab notebooks:
1. **[MAML_AVAST.ipynb]**: Model-Agnostic Meta-Learning on the AVAST dataset
2. **[MAML_MaleVis.ipynb]**: Model-Agnostic Meta-Learning on the MaleVis dataset
3. **[MAML_BODMAS.ipynb]**: Model-Agnostic Meta-Learning on the BODMAS dataset

## Citation
If you use this work, please cite our paper: [IEEE Xplore Link](https://ieeexplore.ieee.org/abstract/document/10825223?casa_token=iRGwNppUkUQAAAAA:Zp8ZU5vekgf3b7eOfHlubwezGB4OAvDskRQnLYjpuGfN977vMXgQ8qwR8YP3dMThLZeYL47LJBo)  
Shin, J., Rivas, E., Lucio, D., Piplai, A., & Elluri, L. (2024, December).  
Towards Building Generalizable Models for Malware Detection.  
In _2024 IEEE International Conference on Big Data (BigData)_ (pp. 5656-5663). IEEE.  
https://doi.org/10.1109/BigData62323.2024.10825223  

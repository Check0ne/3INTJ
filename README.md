# BUS_project

2D breast ultrasound image에서 breast tumor를 segmentation, classification 하기 위한 project입니다.

실제로 tumor의 boundary characteristic과 shape는 radiologists가 tumor의 class를 판단할 때 도움을 줍니다.

따라서 segmentation, classification single-task learning과 multi-task learning을 수행할 계획입니다.

## Dataset: BUSI

Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.

Class는 benign, malignant, normal 세 가지이며, case마다 image와 mask가 있습니다.

<p align="center"><img src="https://user-images.githubusercontent.com/93698065/224069409-11d78fd3-e8a3-4d9a-8d7f-336d81cc6b9b.png" width="500" height="500"/></p>

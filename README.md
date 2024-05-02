# Restoration-of-Images-Degraded-by-Adverse-Weather-Conditions
CMU-24Spring-16726-Learning_based_image_synthesis-Final_project

Adverse weather conditions, like rain, fog, and snow, drastically reduce visibility and degrade image quality, posing a significant challenge for computer vision algorithms in autonomous navigation and surveillance systems. In our project, we explored the integration of Spatial Information Fusion using Depth Maps within the state-of-the-art de-weathering model Transweather for image restoration tasks. We also investigated the how de-weathering can improve the performance of object detection tasks.

For detailed project description, check ./Proj-Website.

For Transweather modification code, see TransWeather/modification_transweather.md.
For Transweather model weights files for experiments and ablations, check ./TransWeather/models.

Detection is implemented based on Mmdetection. For detection code (and preprocessing for detection), check ./src and ./mmdetection.

We also used Marigold(https://github.com/prs-eth/Marigold) and ADDS-DepthNet(https://github.com/LINA-lln/ADDS-DepthNet) repo for depth estimation.


# PySMRS
An unsupervised compressed-sensing technique for fundamental objects selection

# Introduction

This is a python implementation of [(Ehsan Elhamifar et. al)](http://ieeexplore.ieee.org/document/6247852/). They consider the problem of finding a few representatives for a dataset, (i.e., a subset of data points that efficiently describes the entire dataset). They experiment this technique on data such as video ([Video Summarization](http://encyclopedia.jrank.org/articles/pages/6930/Video-Summarization.html)) but other type of data may be considered for some experiment.

Video summarization is one of the encouraging methods for effective comprehension of video content by selecting informative frames of the video. The aim is to produce a summary of the video which is interesting to the user and representing the whole video. 

In this project the proposed method exploits a Compressive Sensing method (LASSO) for selecting representative frames of the video. 

# Requirements

  - Python 2.7 or greater <br>
  - Opencv (facoltative)
  
# Usage

`git clone https://github.com/DavideNardone/PySMRS.git` <br>

`unzip PySMRS-master.py`

then... run one of the following demo:

`python demo.py (naive example)` <br>

`python demo_video.py` (computationally expensive)

# Example

The following summaries have been produced by setting the threshold *thrP* to 0.7, 06 and 0.5, respectively.
![Alt Text](/img/vid_3.gif) ![Alt Text](/img/vid_2.gif) ![Alt Text](/img/vid_1.gif) 

The original video `Society Raffles.mp4` can be found in the `dataset` folder.

# Authors

Davide Nardone, University of Naples Parthenope, Science and Techonlogies Departement,<br> Msc Applied Computer Science <br/>
https://www.linkedin.com/in/davide-nardone-127428102

# Contacts

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@studenti.uniparthenope.it**

# References

[Ehsan Elhamifar et. al]: "See all by looking at a few: Sparse modeling for finding representative objects." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.

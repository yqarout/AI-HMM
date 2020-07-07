Copyright 2019 Yazan Qarout \
If you use this code in your research, please cite: \
Qarout, Y.; Raykov, Y.P.; Little, M.A. Probabilistic Modelling for Unsupervised Analysis of Human Behaviour in Smart Cities. Sensors 2020, 20, 784. 

# Adaptive Input infinite Hidden Markov Model (AI-HMM)
Implementation of the novel Adaptive Input infinite Hidden Markov Model (AI-HMM), a Bayesian nonparametric hidden Markov model which can leverage discrete contextual variables. Conventional HMMs are good at discovering interpretative time series clusters, AI-HMMs allow for HMMs to leverage additional discrete inputs, such as time of day and learn different transition dynamics for each additional category of context.

## Getting Started

Download the folders 'hmm', 'data' and the file 'demo.py'. 'demo.py' is an example of the code on the data available in the folder 'data'. Folder 'hmm' contains the main functions for the AI-HMM which will be called upon in 'demo.py'.

## Prerequisites

<pre><code>Numpy

</code></pre>


The AI-HMM is a flexible latent space model that to account for various types of contextual information. This contextual information can include difference between time series deicrepencies (for example data generated by different sensors), weekday/weekend differences in traffic behaviour and different enviornments of data collection. A probabilstic graphical model of the AI-HMM can be seen below.

![aihmm](https://user-images.githubusercontent.com/67744584/86620884-132e2b00-bfb5-11ea-8656-5c9da84d24a7.png)

# Modeling Examples - Synthetic Data

Below is an image demonstrating the the performance of the AI-HMM at estimating the transition matrix on synthetic data. Given data generated from 2 different transition matrices (Π<sub>1</sub> and Π<sub>2</sub> actual), the model is capable of reproducing the transitions matrices (Π<sub>1</sub> and Π<sub>2</sub> estimated) with high accuracy.

<img width="671" alt="Screen Shot 2020-07-06 at 17 41 25" src="https://user-images.githubusercontent.com/67744584/86618285-8c774f00-bfb0-11ea-9217-299cb57af781.png">

To compare, the conventional HDP-AR-HMM can only reproduce a single average transition matrix (Π<sub>HDP-AR-HMM</sub>) by design.

<img width="329" alt="Screen Shot 2020-07-06 at 17 41 58" src="https://user-images.githubusercontent.com/67744584/86618820-74ec9600-bfb1-11ea-90e7-c4b5284313cf.png">

# Modeling Examples - Real Data

The code in this repository clsuters the time series data from the Dodgers loop dataset containing loop sensor data counting the number of vehicles that pass through the Glendale on-ramp for the 101 North Freeway in Los Angeles in 5 min (288 reading per day) intervals over a period of 25 weeks. The dataset contains weekday morning and afternoon traffic-peak trends and a baseball stadium in the vicinity of the ramp allowing for the observation of the post-game traffic rise.The code uses the added input variable to differentiate between weekday and weekend trends. An example of the results that can be obtained from the demo can be seen below.

<img width="888" alt="Screen Shot 2020-07-06 at 18 28 28" src="https://user-images.githubusercontent.com/67744584/86621805-bdf31900-bfb6-11ea-9526-e8944c1e627e.png">

Each colour represents a specific state that the respective observations was clustered into. Morning and evening traffic peaks on weekdays are clustered into state 1 (blue) and state 3 (green), respectively. Traffic caused by evening baseball games near 23:00 (Wednesday, Friday and Saturday) have been clustered into state 6 (brown), whereas traffic peaks caused by afternoon games at about 16:30 (Sunday) were clustered into state 7 (pink).

For questions and enquiries please email yazan.qarout@gmail.com

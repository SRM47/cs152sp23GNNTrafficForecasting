# CSCI 152 Neural Networks Final Project
## Graph Neural Network Traffic Flow Forecasting in California

### Members
Austin Zang, Devin Guinney, Ethan Lee, Saatvik Kher, Sam Malik

### Project Overview
Traffic flow prediction is vital for the success of transportation systems. Traffic flow prediction estimates the flow of traffic in a particular region and time in the future. Applications of neural networks in this domain are essential to improve travel safety and foster cost-efficient travel [1]. The current literature on this topic relies heavily on using Convolutional Neural Networks and Recurrent Neural Networks to forecast traffic flow. However, according to Weiwei Jiang and Jiayun Luo, a new type of neural network, Graph Neural Network, has become increasingly popular in modeling traffic flow's spatial and temporal dependencies [2]. In this project, we propose to extend this by using a GNN to predict traffic flow, specifically in California.

### Goals
1. Predict traffic flow in San Francisco
2. Deploy to a web-based frontend 
3. Successfully learn and apply a GNN
4. Compare GNN performance to traditional CNN, RNN performance in literature 

### Ethical Sweep
We will closely monitor this project to ensure our network does not unfairly affect specific areas or groups. Though we are limiting our project only to consider traffic within the United States, to keep consistency amongst laws and customs, the type of driving, vehicles, and demographics will change from state to state, county to county, and even town to town. One such factor that we will consider is driving habits. The attitude of safe driving and speed limits will be subject to the territory. Similarly, weather conditions and the general geographical landscape will change drastically nationwide. A model trained in a bustling metropolitan will not necessarily be able to account for a snowstorm in a sparse rural area. 

Given that we would not want to prejudice the model for a particular environment unfairly, we will attempt to account for this by adjusting accordingly. Additionally, the speed of vehicles will vary, as well as maneuverability and ability to change lanes. Finally, we will need to consider the car and conditions in which the model will be trained. The vehicle, driver, road conditions, weather, time of year, state driving laws, the density of police surveillance, and even the number of occupants will affect the data passed to the model. 


### References (loosely cited)
[1] https://www.tandfonline.com/doi/full/10.1080/23311916.2021.2010510 <br/>
[2] https://arxiv.org/pdf/2101.11174.pdf

### Related Works Search
How Powerful are Graph Neural Networks? <br/>
[1] https://arxiv.org/abs/1810.00826 <br/>
This is a much more theoretical paper, going through the possibilities and limitations of graph neural networks. This is done by creating a theoretical Graph Network, the Graph Isomorphism Network (GIN). They use 9 different bioinformatics or social network datasets to evaluate the GIN vs other Graph Neural Networks and show how it is theoretically more optimal. <br/> 

DDP-GCN: Multi-Graph Convolutional Network for Spatiotemporal Traffic Forecasting <br/>
[2] https://arxiv.org/pdf/1905.12256.pdf <br/>
This paper introduces DDP-GCN (DIstance, Direction, and Position Graph Convolutional Network) for traffic flow forecasting. They describe traffic flow prediction as highly complex due to non-euclidean characteristics. It cited that previous works modeled spatial dependencies between roads and cars on them using distance. This paper introduces the use of direction and positional relationship alongside distance for this purpose. To describe these three characteristics, the paper uses multi graphs, a graph that allows more than one edge per pair of vertices to incorporate three spatial relationships into the neural network. DDPGCN achieved sota over baselines on two large scale datasets. 

Deep Learning in Transport Studies: A Meta-analysis on the Prediction Accuracy <br/>
[3] https://doi.org/10.1007/s42421-020-00030-z <br/>
This paper surveys the prediction accuracy, neural network architecture, and dataset use of 136 past transport studies. Combined CNN-LSTM models have historically shown the strongest performance for traffic prediction problems: the CNN component can learn spatial dependencies while the LSTM compoonent can learn temporal dependencies. The most common type of transport study area was traffic speed/flow prediction - likely due to ease of data availability. Accuracy in this area is high enough for public use (market prediction, policy design, etc.). However, other areas, like travel demand, driver behaviour and accident prediction, could hugely benefit from increased Neural Network model performance. "Black box" non-interpretability of NNs is a big hurdle too, since travel behaviour models are typically grounded in theory.

Traffic Flow Prediction via Spatial Temporal Graph Neural Network <br/>
[4] https://dl.acm.org/doi/pdf/10.1145/3366423.3380186 <br/>
The 2020 paper introduces a novel spatial temporal graph neural network for traffic flow prediction that specializes in time-varying features of the graph. The proposed model claims to comprehensively capture spatial and temporal patterns and offers a mechanism to aggregate information from adjacent roads. The framework proposes a GNN layer for adjacent roads and a recurrent network combined with a transformer layer for local and global temporal trends. The researchers then validate the feasibility and advantages of the proposed framework on real traffic datasets. While the framework seems technically complex to implement, it might help inspire ideas for how to deal with spatial and temporal dependencies in our problem. 

Graph Neural Networks for Modeling Traffic Participant Interaction <br/>
[5] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8814066&isnumber=8813768 <br/>
This paper interprets a traffic scene as a graph of interacting vehicles. Using GNNs, they make traffic predictions using interactions between traffic participants while being computationally efficient and providing large model capacity. They showed that prediction error in scenarios with much interaction decreases by 30 % compared to a model that does not take interactions into account. This suggests that interaction is important, and shows that we can model it using graphs. 


### Introduction

Traffic across the United States is a problem often discussed both academically and casually: on a more macro scale, it's a cause for economic efficiency, pollution, and the deterioration of human health. On a micro scale, almost everyone experiences a "bad day" in transit.
One analytical tool that has come into play with the rise of machine learning is traffic forecasting - the ability to use historical traffic data to predict information like volume, speed, and flow of vehicles at a given time. Government officials may then use these predictions to plan more efficient road infrastructure, making decisions on the budget, scope, and geometry of America's transportation network.
The San Francisco Bay Area is particularly interesting to analyze due to its unique needs and existing strengths. As the second largest metro area and center of commerce in California, SF's roads see a great volume of use - at the same time, the coastal nature of the city lends to bottlenecks and points of inefficiency. On top it all, the BART (Bay Area Rapid Transit) network is theoretically designed to supplement SF's vehicular infrastructure. 
Finally, Graph Neural Networks (GNNs) are a rapidly advancing area of ML research, featuring an architecture designed to analyze the complex topology of interconnected graphs. With GNNs, it's possible to do node-level, edge-level, and graph-level inference - a nearly 1-1 translation of the real-world traffic prediction problem.  
Considering all of these factors, we hope to develop a project that applies GNN methodologies to traffic forecasting in the SF Bay Area. Ideally, our model will be able to identify bottlenecks, inefficiencies, and "danger zones" within the city that could be remedied with either greater road development or BART networking. Short of actual policy recommendation, we could also use this information to help SF drivers identify patterns and improve their quality of travel within the city.

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
[1]  https://www.tandfonline.com/doi/full/10.1080/23311916.2021.2010510
[2] https://arxiv.org/pdf/2101.11174.pdf

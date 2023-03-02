# CSCI 152 Neural Networks Final Project
## Graph Neural Network Traffic Flow Forecasting in California

### Members
Austin Zang, Devin Guinney, Ethan Lee, Saatvik Kher, Sam Malik


### Introduction

Traffic across the United States is a problem often discussed both academically and casually: on a more macro scale, it's a cause for economic efficiency, pollution, and the deterioration of human health. On a micro scale, almost everyone experiences a "bad day" in transit.
One analytical tool that has come into play with the rise of machine learning is traffic forecasting - the ability to use historical traffic data to predict information like volume, speed, and flow of vehicles at a given time. Applications of neural networks in this domain are essential to improve travel safety and foster cost-efficient travel according to <a href="https://www.tandfonline.com/doi/full/10.1080/23311916.2021.2010510">Kashyap et al, 2021</a>. Government officials may then use these predictions to plan more efficient road infrastructure, making decisions on the budget, scope, and geometry of America's transportation network. 

The San Francisco Bay Area is particularly interesting to analyze due to its unique needs and existing strengths. As the second largest metro area and center of commerce in California, SF's roads see a great volume of use - at the same time, the coastal nature of the city lends to bottlenecks and points of inefficiency. On top it all, the BART (Bay Area Rapid Transit) network is theoretically designed to supplement SF's vehicular infrastructure. 
Finally, Graph Neural Networks (GNNs) are a rapidly advancing area of ML research, featuring an architecture designed to analyze the complex topology of interconnected graphs. The current literature on this topic relies heavily on using Convolutional Neural Networks and Recurrent Neural Networks to forecast traffic flow. However, according to a work by <a href="https://arxiv.org/pdf/2101.11174.pdf">Weiwei Jiang and Jiayun Luo (2022)</a>, a new type of neural network, the Graph Neural Network, has become increasingly popular in modeling traffic flow's spatial and temporal dependencies. With GNNs, it's possible to do node-level, edge-level, and graph-level inference - a nearly 1-1 translation of the real-world traffic prediction problem.  

Considering all of these factors, we hope to develop a project that applies GNN methodologies to traffic forecasting in the SF Bay Area. Ideally, our model will be able to identify bottlenecks, inefficiencies, and "danger zones" within the city that could be remedied with either greater road development or BART networking. Short of actual policy recommendation, we could also use this information to help SF drivers identify patterns and improve their quality of travel within the city. We will also compare our GNN performance and application to existing approaches (including conventional CNN) in literature.


### Related Works
We can draw inspiration from several sources here: this area of research is novel but good headway has been made by many researchers. <a href="https://arxiv.org/abs/1810.00826">Xu et. al, 2019</a> present a general overview of GNNs and their unique power when applied to representation learning on graphs: they also present a framework for evaluating the expressive power of different GNN architectures, an important problem to consider when designing our own. A 2020 meta-analysis by <a href="https://link.springer.com/article/10.1007/s42421-020-00030-z">Varghese et. al, 2020</a> establishes a good baseline for our work, as they compare the accuracy, datasets, and methodology of many different neural network approaches to traffic forecasting. In addition, they also identify sub-problems within the larger traffic forecasting space that seem particularly difficult, and analyze the impact and state of each of these fields. We may base our performance evaluation against the 136 other studies analyzed by this paper.

There is also much possibility to go beyond just a basic GNN when building our model. Take, for example, <a href="https://arxiv.org/pdf/1905.12256.pdf">Lee and Rhee, 2022</a> who proposed a novel <b>DDP-GCN</b> (Distance, Direction, and Position Graph Convolutional Network) model capable of capturing more non-euclidean traffic flow characteristics. A multigraph data representation is at the heart of this approach: allowing more than one edge per vertex pair captures complex spatial relationships. We will likely attempt a similar approach when constructing our own data representation.
<a href="https://dl.acm.org/doi/pdf/10.1145/3366423.3380186">Wang et. al, 2020</a> introduce a spatial temporal graph neural network for traffic flow prediction that specializes in time-varying features of the graph. The proposed model claims to comprehensively capture spatial and temporal patterns and offers a mechanism to aggregate information from adjacent roads. The framework proposes a GNN layer for adjacent roads and a recurrent network combined with a transformer layer for local and global temporal trends. The researchers then validate the feasibility and advantages of the proposed framework on real traffic datasets. While the framework seems technically complex to implement, it might help inspire ideas for how to deal with spatial and temporal dependencies in our problem.
Another consideration is vehicle-vehicle interaction - after all, traffic is in large part determined by the reactive behavior of human drivers. <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8814066&isnumber=8813768">Diehl et. al, 2019</a> developed an approach that successfully interprets a traffic scene as a complex graph of interacting vehicles. Using GNNs, they make traffic predictions using interactions between traffic participants while being computationally efficient and providing large model capacity. They showed that prediction error in scenarios with much interaction decreases by 30 % compared to a model that does not take interactions into account. This suggests that interaction is important, and shows that we can model it using graphs.
Taking into account all the work that has already been done, we hope to devise an approach that can dynamically mix the best parts of previous research and acheive comprable, if not better, GNN performance. 

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


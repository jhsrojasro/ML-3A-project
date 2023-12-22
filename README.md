# ML-3A-project: Game recommender system based in Graph Neural Networks
## Jhon Sebastian Rojas Rodriguez

### Task Description 

The idea of the project is to build a recommender system based on Graph Neural Networks, the problem can be formulated as a Link Prediction Task (the prediction whether an edge exists between two particular nodes of the graph). The dataset used is a Steam Recommendation Dataset public available in Kaggle. The data is represented by a bipartite graph  where the vertex-set can be split into two separate disjoint sets: users and games. The weight of each edge is 1 if the users liked the game and -1 if they disliked it.

This project formulate the link prediction problem as a binary classification problem as follows:

<ul>
<li>Treat the edges in the graph as positive examples.</li>
<li>Sample a number of non-existent edges as negative examples.</li>
<li>Divide the positive examples and negative examples into training, validation and test sets.</li>
<li>Evaluate the model with a binary classification metric such as Area Under Curve (AUC)</li>
</ul>

### Model Description

The model consists of two GraphSAGE layers, each layer computes new node representations by averaging neighbor information. The DGL framework is used, it provides an implementation of the GraphSAGE layer and some useful optimized functions to make the computations needed.

In order to do the prediction classification usually binary operators such as dot product and L1 / L2 norm are used to encode node embeddings into a singular edge embedding value. Then a logistic regression model is used to classify.


### Dataset Description

The dataset contains over 41 million cleaned and preprocessed user recommendations or reviews from a Steam Store. Steam is a leading online platform for purchasing and downloading video games, DLC, and other gaming-related content. Additionally, it contains detailed information about games and add-ons. This project uses mainly the user-game recommendation data (likes, dislikes). However it would be possible to use the game and user data as additional custom node features to include in the graph.

The data was preprocessed with Pandas in order to create a DGLDataset class that represents the data as a graph with 27573556 nodes and 38354101 edges. The dataset is split into train, validation and test sets, with 10% of the edges for validation and another 10% for the test.

An equal number of negative examples were sampled to train the classifier that were split using the same percentage of edges for validation and test.


### Results

Different embedding sizes were tested in order to overfit the model:

![Train example 1](https://drive.google.com/uc?id=1-T-V3zL6QyHdYUhmJeigRG-n6QCwbacU)

![Train example 2](https://drive.google.com/uc?id=1pfi5Yen5N0G7Y3rBMaucbr6lMlhsMswr)

![Train example 3](https://drive.google.com/uc?id=1NlCt6jgIuZLso69p_SseiHaKf8xUInOl)

![Train example 4](https://drive.google.com/uc?id=16DuAqXUfbb04OmuT8w1q-BfnLIJUMXHg)

One additional SAGE layer was added to the model, but the overfitting was not achieved.

![Train example 4](https://drive.google.com/uc?id=1lTHCWIZySjb8cYVwOrfrjbSNEDHtT3b4)

In all the train example we can see a proper decreasing loss for the train and validation sets, the higher AUC (0.9984655101101535) was achived by the last model.

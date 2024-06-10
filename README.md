# FeatureSelectionUsingGA
This project uses a genetic algorithm (GA) to enhance feature selection for a neural network classifying emotions from the RAVDESS Facial Landmark Tracking dataset, focusing on distinguishing happy and sad emotions. The goal is to refine the neural network's predictive accuracy by iteratively selecting the most informative facial landmarks.
Introduction
This report outlines the use of a genetic algorithm (GA) to enhance feature selection for a neural network tasked with classifying emotions from the RAVDESS Facial Landmark Tracking dataset. By focusing on distinguishing between happy and sad emotions, the study aims to refine the neural network's predictive accuracy through the iterative selection of the most informative facial landmarks.

Methodology
Data Preparation
The study commenced with gathering data by reading and merging various files from a specified directory into a unified DataFrame. This process involved filtering files based on their naming conventions, which provided insights into attributes such as modality, emotion, and intensity, to select only those relevant to the study's focus on speech and the chosen emotions.

Feature and Label Definition
Features were derived from the aggregated DataFrame, excluding the last column which was designated for labels. A normalization step was applied to the feature set to scale the values appropriately, facilitating more efficient learning by the neural network.

Genetic Algorithm Overview
The GA was implemented to identify an optimal subset of features that would improve the neural network model's performance. This algorithm simulates the process of natural selection, where each generation of solutions (chromosomes) evolves through genetic operators to maximize the fitness function, measured as the accuracy of the model on the selected features.

Genetic Algorithm Components
a. Chromosome Representation
Chromosomes were represented as binary strings, where each bit indicated the inclusion (1) or exclusion (0) of a feature. The length of a chromosome corresponded to the total number of features available in the dataset.

b. Genetic Operators
Selection was performed using a roulette wheel approach, ensuring that chromosomes with higher fitness had a greater chance of being chosen for reproduction. Crossover involved a single-point mechanism, where offspring were created by exchanging segments of parent chromosomes beyond a randomly selected point. Mutation introduced variation by randomly flipping bits in a chromosome, aiding the exploration of the solution space.

Results and Performance Evaluation
The neural network's performance, using the features selected through the GA, was quantified in terms of accuracy. This evaluation highlighted the benefits of the GA-based feature selection process by comparing model performance before and after feature selection. The results demonstrated a noticeable improvement in accuracy, underscoring the effectiveness of selecting pertinent features.

Analysis
The impact of the GA on model performance was evident from the comparison between the initial and final model accuracies. The analysis revealed that feature selection not only improved model accuracy but also potentially reduced training time and model complexity by focusing on fewer, more relevant features. The features identified by the GA as most informative offer insights into which facial landmarks are most indicative of emotional states, providing valuable knowledge for further research and application development.

Link for Dataset

RAVDESS Facial Landmark Training Dataset https://www.kaggle.com/datasets/uwrfkaggler/ravdess-facial-landmark-tracking

Conclusion
The application of a GA for feature selection has proven to be an effective strategy for optimizing a neural network model aimed at emotion classification. This approach not only enhanced model accuracy but also provided a deeper understanding of the key features contributing to the prediction of emotional states. The findings underscore the value of feature selection in machine learning, particularly in domains where the interpretability and efficiency of models are crucial.

 

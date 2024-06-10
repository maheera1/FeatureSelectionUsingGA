import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

dataframes = []
DirectoryPath = r'C:\Users\Maheera Ashfaq\OneDrive\Documents\Semester 6\AI\Assignment2\archive'

# Check if path exists
if not os.path.exists(DirectoryPath):
    print(f"Directory not found: {DirectoryPath}")
else:
    print(f"Directory found: {DirectoryPath}")
    # Iterate over all files in the directory
    for filename in os.listdir(DirectoryPath):
        # Parse the filename to extract the modality and emotion codes
        parts = filename.split('-')
        if len(parts) >= 7: 
            modalityCode = parts[0]
            vocalChannelCode = parts[1]  
            emotion_code = parts[2]        
            intensityCode = parts[3]      
            
            # criteria to select files
            if modalityCode == '01' and vocalChannelCode == '01' and emotion_code in ['03', '04'] and intensityCode == '01' :
               
                # Construct full path to the file
                filePath = os.path.join(DirectoryPath, filename)
                print(f"Found match, Now reading file: {filePath}")

                try:
                    df = pd.read_csv(filePath)
                    if not df.empty:
                        dataframes.append(df)
                        print(f"Data added from file: {filePath}")
                    else:
                        print(f"Empty DataFrame for file: {filePath}")
                except Exception as e:
                    print(f"Error reading file {filePath}: {e}")

# Combining all the dataframes into a single dataframe if the list is not empty
if dataframes:
    combinedDataframes = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataframe is created having {len(combinedDataframes)} rows.")
else:
    print("Dataframes were not created as the list is empty.")


features = combinedDataframes.iloc[:, :-1]  
labels = combinedDataframes.iloc[:, -1]

# Normalize features
scaler = StandardScaler()
normalizedFeatures = scaler.fit_transform(features)

# Calculate the number of samples for the test set
test_size = int(normalizedFeatures.shape[0] * 0.2)

# Randomly shuffle the indices of the samples
indices = np.random.permutation(normalizedFeatures.shape[0])

# Use the first `test_size` indices for the test set and the rest for the training set
test_indices = indices[:test_size]
train_indices = indices[test_size:]

# Split the features and labels based on the indices
Xtrain, Xtest = normalizedFeatures[train_indices], normalizedFeatures[test_indices]
Ytrain, Ytest = labels[train_indices], labels[test_indices]

#Initializing Population
def PopulationInitialize(populationSize, chromolength):
 
    return np.random.randint(2, size=(populationSize, chromolength))

# fitness function to evaluate the accuracy of a classifier trained on selected features.
def FitnessFunction(chromosome, Xtrain, Xtest, Ytrain, Ytest):
    # Select features from the training and test sets based on the chromosome
    featuresSelected = Xtrain[:, chromosome == 1]
    featuresSelectedTest = Xtest[:, chromosome == 1]

    # Ensure at least one feature is selected to avoid errors
    if featuresSelected.shape[1] == 0:
        return 0

    # Create a simple neural network classifier
    model = Sequential()
    model.add(Input(shape=(featuresSelected.shape[1],)))
    model.add(Dense(units=featuresSelected.shape[1], activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam',
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])

    # Train the classifier on the selected features
    model.fit(featuresSelected, Ytrain, epochs=5, batch_size=16, verbose=0)

    # Evaluate the classifier on the test set
    _, accuracy = model.evaluate(featuresSelectedTest, Ytest, verbose=0)

    # Return the accuracy as the fitness score
    return accuracy

#selection for crossover
def selection(population, fitnesses):
    # Normalize fitnesses to sum to 1
    fitnesses = fitnesses / np.sum(fitnesses)
    # Calculate the cumulative sum of normalized fitness values
    CumulativeProbs = np.cumsum(fitnesses)

    RandNum = np.random.rand(2) #between 0 and 1

    # Find the indices where the random numbers fall in the cumulative probabilities
    parent_indices = [np.argmax(CumulativeProbs >= RandNum[i]) for i in range(2)]

    # Select the parents from the population
    selected_parents = population[parent_indices[0]], population[parent_indices[1]]

    return selected_parents

#combining chromos
def crossoverFunction(parent1, parent2):
    point = np.random.randint(1, len(parent1))  # Choose crossover point
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

#change bits for diversity
def mutateFunc(chormo, mutation_rate):
    for i in range(len(chormo)):
        # Random chance to mutate
        if np.random.rand() < mutation_rate:  
            chormo[i] = 1 - chormo[i]


def GenecticALGO(X_train, y_train, X_test, y_test, chromolength, populationSize=10, generations=3, mutation_rate=0.01):
    # Initialize population with the correct chromosome length
    population = PopulationInitialize(populationSize, chromolength)
    BestChoromo = None
    # Initialize with the worst possible fitness
    BestFitness = -np.inf  

    for generation in range(generations):
        print(f"\nGeneration Number {generation + 1}:")
        fitnesses = []

        # Evaluate the fitness of each chromosome
        for idx, individual in enumerate(population):
            Fitnes = FitnessFunction(individual, X_train, X_test, y_train, y_test)
            fitnesses.append(Fitnes)
            print(f"The accuracy of Chromosome {idx + 1} : {Fitnes:.4f}")

        # Convert to a numpy array for efficiency
        fitnesses = np.array(fitnesses)
        max_fitness_index = np.argmax(fitnesses)
        max_fitness = fitnesses[max_fitness_index]

        # Update the best chromosome if the current generation has a new best
        if max_fitness > BestFitness:
            BestFitness = max_fitness
            BestChoromo = population[max_fitness_index]

        print(f"The Best fitness for this generation is: {BestFitness:.4f}")

        # Generate the new population
        new_population = []
        for _ in range(populationSize // 2):  # We will create two children for each pair of parents
            parent1, parent2 = selection(population, fitnesses)
            child1, child2 = crossoverFunction(parent1, parent2)
            mutateFunc(child1, mutation_rate)
            mutateFunc(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = np.array(new_population)

    print("\nFinal Generation:")
    for idx, individual in enumerate(population):
        print(f"Chromosome {idx + 1} accuracy: {fitnesses[idx]:.4f}")

    print("\nResulting chromosome of the GA function:")
    print("The Best chromosome is:", BestChoromo)
    print(f"The Best accuracy is: {BestFitness:.4f}")
    
    return BestChoromo, BestFitness


if __name__ == "__main__":

    chromosome_length = Xtrain.shape[1]  

    population = PopulationInitialize(20, chromosome_length)  # Adjusted population initialization

    # Run the GA to find the best chromosome
    best_chromosome, _ = GenecticALGO(Xtrain, Ytrain, Xtest, Ytest, chromosome_length)


    
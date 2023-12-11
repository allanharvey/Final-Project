import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

# python final_project.py ncaa.csv nba.csv

class CustomLogisticRegression:
    """
    A Logistic Regression implementation to predict NBA prospects.

    Args:
    - learning_rate (float): The learning rate for gradient descent.
    - num_iterations (int): The number of iterations for gradient descent.

    Methods:
    - sigmoid(z): Applies the sigmoid function to the input.
    - fit(X, y): Trains the logistic regression model to predict NBA prospects.
    - predict_proba(X): Predicts the probability of being an NBA prospect for input data.
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initializes the CustomLogisticRegression model.

        Args:
        - learning_rate (float): The learning rate for gradient descent.
        - num_iterations (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """
        Applies the sigmoid function to the input.

        Args:
        - z (numpy array): Input to the sigmoid function.

        Returns:
        - numpy array: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Trains the logistic regression model to predict NBA prospects.

        Args:
        - X (DataFrame): Features for the model.
        - y (DataFrame): Target variables.

        Returns:
        - None
        """
        m, n = X.shape
        self.weights = np.zeros((n,))  # Adjust the shape of weights
        self.bias = 0  # Adjust the shape of bias

        for _ in range(self.num_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(model)

            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predicts the probability of being an NBA prospect for input data.

        Args:
        - X (DataFrame): Input data.

        Returns:
        - numpy array: Predicted probabilities.
        """
        model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(model)
        return predictions

class NBAProspectsClassifier:
    """"
    NBA Prospects Classifier that compares the stats of current NCAA basketball players
    with former NCAA basketball players who made it to the NBA.

    Args:
    - ncaa_data_path (str): Path to the CSV file containing NCAA basketball data.
    - nba_data_path (str): Path to the CSV file containing NBA basketball data.
    - threshold (float): The classification threshold.

    Methods:
    - load_data(filepath): Loads the dataset from a CSV file.
    - calculate_z_scores(stat_names): Calculates Z-scores for each statistical measure.
    - train_model(): Trains the logistic regression model.
    - predict_ncaa_players(): Predicts the probability of NCAA players being NBA prospects.
    - compare_z_scores(stat_names): Visualizes the Z-scores for NCAA players' stats.

    """
    def __init__(self, ncaa_data_path, nba_data_path, threshold=0.99):
        """
        Initializes the NBAProspectsClassifier.

        Args:
        - ncaa_data_path (str): Path to the CSV file containing NCAA basketball data.
        - nba_data_path (str): Path to the CSV file containing NBA basketball data.
        - threshold (float): The classification threshold.
        """
        self.ncaa_data = self.load_data(ncaa_data_path)
        self.nba_data = self.load_data(nba_data_path)
        self.threshold = threshold
        self.custom_model = None

    def load_data(self, filepath):
        """
        Loads the dataset from a CSV file.

        Args:
        - filepath (str): Path to the dataset file.

        Returns:
        - DataFrame: The loaded dataset.
        """
        return pd.read_csv(filepath)

    def calculate_z_scores(self, stat_names):
        """
        Calculates Z-scores for each statistical measure.

        Args:
        - stat_names (List): Names of statistical measures for interpretation.

        Returns:
        - DataFrame: Z-scores for each player and statistical measure.
        """
        current_players = self.ncaa_data
        former_players = self.nba_data

        z_scores_data = {
            'Player': current_players['Player'],
        }

        for stat in stat_names:
            z_scores_data[f'{stat}_z_score'] = [
                (current_players.loc[i, stat] - np.mean(former_players[stat])) / np.std(former_players[stat])
                for i in range(len(current_players))
            ]

        # Create a DataFrame with Z-scores
        z_scores_df = pd.DataFrame(z_scores_data)

        # Calculate the mean Z-score for each NCAA player
        mean_z_scores = z_scores_df[[f'{stat}_z_score' for stat in stat_names]].mean(axis=1)
        z_scores_df['Mean_Z_Score'] = mean_z_scores

        # Order the DataFrame by the mean Z-score in ascending order
        z_scores_df = z_scores_df.sort_values(by='Mean_Z_Score', ascending=False)

        # Resetting index to start numbering from 1
        z_scores_df.index = np.arange(1, len(z_scores_df) + 1)

        return z_scores_df

    def train_model(self):
        """
        Trains a logistic regression model to predict NBA prospects.

        Returns:
        - None
        """
        X_nba = self.nba_data[['Points_Per_Game', 'Rebounds_Per_Game', 'Assists_Per_Game']]
        y_nba = np.ones(len(X_nba))  # Label all NBA players as 1

        self.custom_model = CustomLogisticRegression()
        self.custom_model.fit(X_nba, y_nba)
        
    def predict_ncaa_players(self):
        """
        Predicts the probability of NCAA players being NBA prospects using the trained model.

        Returns:
        - DataFrame: Predictions for NCAA players.
        """
        X_ncaa = self.ncaa_data[['Points_Per_Game', 'Rebounds_Per_Game', 'Assists_Per_Game']]
        z_scores = self.calculate_z_scores(['Points_Per_Game', 'Rebounds_Per_Game', 'Assists_Per_Game'])

        # Scale Z-scores to a range suitable for the hyperbolic tangent function
        scaled_z_scores = (z_scores['Mean_Z_Score'] - z_scores['Mean_Z_Score'].mean()) / z_scores['Mean_Z_Score'].std()

        # Map Z-scores to probabilities using the hyperbolic tangent function
        probabilities = (np.tanh(scaled_z_scores) + 1) / 2
        
        # Realism factor
        probabilities /= 2.25

        # Replace NaN values in probabilities with a default value (0)
        probabilities = np.nan_to_num(probabilities, nan=0)

        # Create a DataFrame with predictions
        predictions_df = pd.DataFrame({
            'Player': self.ncaa_data['Player'],
            'Probability_of_NBA_Prospect': probabilities
        })

        # Order the DataFrame by probability in descending order
        predictions_df = predictions_df.sort_values(by='Probability_of_NBA_Prospect', ascending=False)
        predictions_df.index = np.arange(1, len(predictions_df) + 1)

        return predictions_df
    
    def compare_z_scores(self, stat_names):
        """
        Visualizes the Z-scores for each statistical measure of NCAA players.

        Args:
        - stat_names (List): Names of statistical measures for interpretation.

        Returns:
        - None
        """
        # Calculate Z-scores for NCAA players
        z_scores_df = self.calculate_z_scores(stat_names)

        # Plotting
        plt.figure(figsize=(12, 8))

        bar_width = 0.3  # Width of each bar
        positions = np.arange(1, len(z_scores_df) + 1)

        # Plotting side-by-side bars for each statistical measure
        for i, stat in enumerate(stat_names):
            plt.bar(positions + (i - 1) * bar_width, z_scores_df[f'{stat}_z_score'], width=bar_width, label=f'{stat} Z-Score', alpha=0.7)

        plt.title('Z-Scores for NCAA Players (Points, Rebounds, Assists)')
        plt.xlabel('Player')
        plt.ylabel('Z-Score')
        plt.xticks(positions, z_scores_df['Player'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
def main():
    """
    Parses arguments and executes NBA Prospects Classifier functionality.
    """
    parser = argparse.ArgumentParser(description="NBA Prospects Classifier")
    parser.add_argument("ncaa_data_path", help="Path to the CSV file containing NCAA basketball data")
    parser.add_argument("nba_data_path", help="Path to the CSV file containing NBA basketball data")
    parser.add_argument("--threshold", type=float, default=0.99, help="Classification threshold")
    parser.add_argument("--visualize_column", help="Name of the column to visualize")

    args = parser.parse_args()

    classifier = NBAProspectsClassifier(
        ncaa_data_path='ncaa.csv',
        nba_data_path='nba.csv',
        threshold=args.threshold
    )

    # Calculate and print Z-scores for each player
    stat_names = ['Points_Per_Game', 'Rebounds_Per_Game', 'Assists_Per_Game']
    z_scores_df = classifier.calculate_z_scores(stat_names)
    print("\nZ-Scores for each player:")
    print(z_scores_df)

    # Train the model
    classifier.train_model()

    # Predict NBA prospects among NCAA players
    predictions_df = classifier.predict_ncaa_players()
    print("\nPredictions for NCAA Players:")
    print(predictions_df)

    # Visualize the distribution of a specified column
    if args.visualize_column:
        classifier.visualize_data(args.visualize_column)
        
    # Compare Z-scores for NCAA players with the entire NBA data
    classifier.compare_z_scores(stat_names)

if __name__ == "__main__":
    main()
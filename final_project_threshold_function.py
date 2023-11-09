def threshold(current_players, former_players, stat_names):
    """
    Allan Persaud
    
    Calculates Z-scores for each statistical measure by comparing the mean and standard deviation
    of current NCAA players with those of former NCAA players who made it to the NBA. The Z-scores are then displayed alongside the corresponding 
    statistical measure names. The optimal threshold is determined as the mean of the Z-scores, providing a decision boundary
    for classifying players as potential NBA prospects or non-prospects.

    Parameters:
    - current_players: 2D list containing basic stats of current NCAA players.
    - former_players: 2D list containing basic stats of former NCAA players who made it to the NBA.
    - stat_names: List containing names of statistical measures for interpretation.

    Returns:
    - float: The optimal threshold for classification.
    
    Output Explanation:
    The optimal threshold represents the mean of Z-scores calculated by comparing the basic stats of current NCAA players
    with those of former NBA players. Z-scores indicate how many standard deviations a data point is from the mean.

    - If a Z-score is positive, it means the current player's performance in that stat is above the mean of former NBA players.
    - If a Z-score is negative, it means the current player's performance is below the mean of former NBA players.
    - A Z-score of 0 indicates that the player's performance is exactly at the mean of former NBA players.
    - A high positive Z-score indicates exceptional performance above the mean.
    - A high negative Z-score indicates poor performance below the mean.
    """
    
    def calculate_mean(values):
        return sum(values) / len(values)

    def calculate_standard_deviation(values, mean):
        squared_diff = sum((x - mean) ** 2 for x in values)
        variance = squared_diff / len(values)
        return variance ** 0.5

    # Calculate mean and standard deviation for each stat
    current_means = [calculate_mean([player[i] for player in current_players]) for i in range(len(current_players[0]))]
    current_stds = [calculate_standard_deviation([player[i] for player in current_players], current_means[i]) for i in range(len(current_players[0]))]
    
    former_means = [calculate_mean([player[i] for player in former_players]) for i in range(len(former_players[0]))]
    former_stds = [calculate_standard_deviation([player[i] for player in former_players], former_means[i]) for i in range(len(former_players[0]))]

    # Calculate Z-scores for each stat
    z_scores = [(current_means[i] - former_means[i]) / ((current_stds[i]**2/len(current_players)) + (former_stds[i]**2/len(former_players)))**0.5 for i in range(len(current_means))]

    # Display Z-scores and Stat Names
    print("Z-Scores of Basic Stats for NBA Prospects Classification:")
    for i, (stat_name, z_score) in enumerate(zip(stat_names, z_scores)):
        print(f"{stat_name}: {z_score}")

    # Find the optimal threshold (mean of Z-scores)
    optimal_threshold = sum(z_scores) / len(z_scores)
    
    return optimal_threshold

# Example usage
# This is a case where a current NCAA is non-prospect for the NBA
current_players = [[10, 5, 8], [12, 6, 9], [11, 4, 7]]
former_players = [[15, 8, 10], [14, 7, 9], [16, 9, 11]]
stat_names = ["Points", "Assists", "Rebounds"]

optimal_threshold = threshold(current_players, former_players, stat_names)
print("Optimal Threshold:", optimal_threshold)
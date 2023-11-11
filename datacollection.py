import requests
from bs4 import BeautifulSoup
import pandas as pd

def collect_and_preprocess_data(ncaa_url, nba_url):
    """
    Kwabena Akuffo
    Collects and preprocesses basketball data from NCAA and NBA sources.

    Parameters:
    - ncaa_url: The URL for NCAA basketball data.
    - nba_url: The URL for NBA basketball data.

    Returns:
    - pd.DataFrame: A processed DataFrame with standardized columns, cleaned data, and missing values handled.
    """
    ncaa_data = scrape_basketball_reference(ncaa_url)

    nba_data = scrape_basketball_reference(nba_url)

    ncaa_data = preprocess_ncaa_data(ncaa_data)

    nba_data = preprocess_nba_data(nba_data)

    merged_data = merge_data(ncaa_data, nba_data)

    return merged_data

def scrape_basketball_reference(url):
    """
    Scrapes basketball data from the specified URL.

    Parameters:
    - url: The URL of the basketball data.

    Returns:
    - pd.DataFrame: A DataFrame containing the scraped data.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    player_names = [player.text for player in soup.select('td[data-stat="player"] a')]
    points_per_game = [float(stat.text) if stat.text else 0.0 for stat in soup.select('td[data-stat="pts_per_g"]')]
    rebounds_per_game = [float(stat.text) if stat.text else 0.0 for stat in soup.select('td[data-stat="trb_per_g"]')]
    assists_per_game = [float(stat.text) if stat.text else 0.0 for stat in soup.select('td[data-stat="ast_per_g"]')]

    data = {'Player': player_names, 'Points Per Game': points_per_game, 'Rebounds Per Game': rebounds_per_game, 'Assists Per Game': assists_per_game}
    df = pd.DataFrame(data)

    return df

def preprocess_ncaa_data(ncaa_data):
    """
    Preprocesses NCAA basketball data.

    Parameters:
    - ncaa_data (pd.DataFrame): The raw NCAA basketball data.

    Returns:
    - pd.DataFrame: The preprocessed NCAA basketball data.
    """
    return ncaa_data

def preprocess_nba_data(nba_data):
    """
    Preprocesses NBA basketball data.

    Parameters:
    - nba_data (pd.DataFrame): The raw NBA basketball data.

    Returns:
    - pd.DataFrame: The preprocessed NBA basketball data.
    """
    return nba_data

def merge_data(ncaa_data, nba_data):
    """
    Merges NCAA and NBA basketball data based on common attributes.

    Parameters:
    - ncaa_data (pd.DataFrame): The preprocessed NCAA basketball data.
    - nba_data (pd.DataFrame): The preprocessed NBA basketball data.

    Returns:
    - pd.DataFrame: The merged basketball data.
    """
    merged_data = pd.merge(ncaa_data, nba_data, on='Player', how='inner')

    return merged_data

ncaa_url = 'https://www.sports-reference.com/cbb/'
nba_url = 'https://www.basketball-reference.com/'
processed_data = collect_and_preprocess_data(ncaa_url, nba_url)
print(processed_data.head())

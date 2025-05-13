"""
Author: Aiden Kim and Nicole Huang
Date: 4/18/2025
Description: Final project main code class. 
The goal of the final project is to identify the best
predictive feature using entropy/gain calculations. 
"""

################################################################################
# IMPORTS
################################################################################

from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Partition import Partition, Example
import statistics as stats

################################################################################
# MAIN FUNCTION
################################################################################

def main():
    #Data cleaning
    years_data, features = clean_data()

    #PROJECT ADDITION: Grouping features into groups, setting as data
    #Feature grouping category specified in 
    years_data, features = group_category_data(years_data, features)
    
    #Selects smallest year as training data
    smallest_yr = min(years_data.keys())
    train_year = smallest_yr # just the year
    train_data = years_data[smallest_yr] # stores data for smallest_yr

    #Setting training data
    X_base, y_base, examples = set_data(train_year, train_data)

    # Calculating the gain values and selecting best feature
    part = Partition(examples, features)
    train_feature_gains = part.calc_gain_by_feature()
    gain_ordered, best_pred_feature = best_feature(train_feature_gains)

    #Printing gain results
    print(f"Best predicted feature for {train_year} (training data): {best_pred_feature}")
    sorted_gain = sorted(gain_ordered.items(), key=lambda item: item[1])
    formatted_gain = "\n".join(f"{feature}: {value:.4f}" for feature, value in sorted_gain)
    print(f"Feature Gains (smallest to largest order):\n{formatted_gain}\n")

    # Visualizing gain value per feature with box plot
    features = list(gain_ordered.keys())
    gains = list(gain_ordered.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(features, gains, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Gain")
    plt.title(f"Gain by Feature for {smallest_yr}")
    plt.grid(axis='y')

    for bar in bars: # Add gain value labels on top of bars
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,                      
            f"{height:.3f}",
            ha='center',
            va='bottom',
            fontsize=6
        )

    plt.tight_layout()
    plt.savefig("figures/predicted_gain.png")
    plt.clf()

    # Calculate the entropy for training year (this will be the predicted entropy)
    pred_entropy = part.calc_feat_entropy(best_pred_feature, True)
    print(f"Best Predicted Feature for {smallest_yr}: {best_pred_feature}\nPredicted Entropy: {pred_entropy}")

    # Calculate the predicted and actual gain for each year
    actual_tot_entropy, predicted_gain = predict_gain(smallest_yr, years_data, pred_entropy, features)
    actual_gain, year_best_features = calc_actual_gain(smallest_yr, years_data, predicted_gain, features)

    #Printing results
    print(f"Using {best_pred_feature} as our predicted value, here are the predicted gain in comparison to actual gain.")
    print("Year | Predicted Gain | Actual Gain | Best Feature")
    print("-----|----------------|-------------|--------------------")
    for year in sorted(predicted_gain):
        pred = predicted_gain.get(year, "N/A")
        actual = actual_gain.get(year, "N/A")
        best = year_best_features.get(year, "N/A")
        print(f"{year} | {pred:^14.4f} | {actual:^11.4f} | {best}")

    # Plotting predicted vs actual gain by year (using best predicted feature)
    # Prepare data (convert into list)
    other_years = sorted([year for year in years_data if year != smallest_yr])
    actual_gain_lst = [actual_gain[year] for year in other_years]
    predicted_gain_lst = [predicted_gain[year] for year in other_years]

    x = np.arange(len(other_years))
    width = 0.35

    plt.figure(figsize=(12, 6))

    # Bar plots for actual and predicted gains
    bars_actual = plt.bar(x - width/2, actual_gain_lst, width, label='Actual Gain', color='skyblue')
    bars_predicted = plt.bar(x + width/2, predicted_gain_lst, width, label='Predicted Gain', color='salmon')

    plt.xlabel("Year")
    plt.ylabel("Gain")
    plt.title(f"{smallest_yr} Actual vs Predicted Gain by Year (Same Feature Comparison)")
    plt.xticks(ticks=x, labels=[str(year) for year in other_years])
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("figures/actual_predicted_gain_bar.png")
    plt.clf()

    # Calculate the residuals (finding the RSS)
    rss = calc_rss(smallest_yr, years_data, actual_gain_lst, predicted_gain_lst)
    m,b = np.polyfit(np.array(other_years),np.array(rss),1)
    line = m*np.array(other_years) + b

    # plot_rss(rss, smallest_yr, other_years, line)

    # Plotting residuals and trend using scatter plot.
    plt.scatter(other_years, rss, label="RSS Predicted - Actual Gain", color='blue')
    plt.plot(other_years, line, color='orange', label = 'Best fit line')
    plt.xlabel("Year")
    plt.ylabel("RSS [(actual_gain - pred_gain)**2]")
    plt.title("Difference of Predicted and Actual Gain Value by Year")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(ticks=other_years, labels=[str(year) for year in other_years])
    plt.savefig("figures/RSS_best.png")

    # This section compares the gain values to the predicted gain value.
    # plot_gain_comparison_w_feat(smallest_yr, year_best_features, predicted_gain, best_pred_feature)

    # Sort years to keep order consistent
    compare_years = sorted([year for year in year_best_features if year in predicted_gain])

    # Gather data
    actual_gains = [year_best_features[year][1] for year in compare_years]
    actual_features = [year_best_features[year][0] for year in compare_years]
    predicted_gains = [predicted_gain[year] for year in compare_years]

    # Plotting bar plot to visualize difference between predicted and actual gain values (diff features)
    x = np.arange(len(compare_years))
    width = 0.35

    plt.figure(figsize=(14, 7))
    bars_actual = plt.bar(x - width/2, actual_gains, width, label="Actual Best Gain", color='skyblue')
    bars_predicted = plt.bar(x + width/2, predicted_gains, width, label="Predicted Gain", color='salmon')

    for i, bar in enumerate(bars_actual):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                f"{actual_features[i]}\n{height:.3f}",
                ha='center', va='bottom', fontsize=7, rotation=90)

    for bar in bars_predicted:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                f"{height:.3f}", ha='center', va='bottom', fontsize=7, rotation=90)

    plt.xlabel("Year")
    plt.ylabel("Gain")
    plt.title("Actual Best Feature Gain vs Predicted Gain by Year")
    plt.xticks(x, [str(year) for year in compare_years], rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    plt.savefig("figures/actual_vs_predicted_best_gain_labeled.png")
    plt.clf()


    print("\nNow we will conduct large scale analysis by setting each year as training data")
    print("and comparing the average RSS for actual and predicted gains for each year.")

    all_years_lst = sorted([year for year in years_data])
    avg_rss_by_yr = {}

    for y_a in all_years_lst: #iterate through list of all years, repeat similar process
        train_year = y_a # just the year
        train_data = years_data[y_a] # stores data for select year

        #Setting training data
        X_base, y_base, examples = set_data(train_year, train_data)

        # Calculating the gain values and selecting best feature
        part = Partition(examples, features)
        train_feature_gains = part.calc_gain_by_feature()
        gain_ordered, best_pred_feature = best_feature(train_feature_gains)

        # #Printing gain results
        # print(f"Best predicted feature for {train_year} (training data): {best_pred_feature}")
        # sorted_gain = sorted(gain_ordered.items(), key=lambda item: item[1])
        # formatted_gain = "\n".join(f"{feature}: {value:.4f}" for feature, value in sorted_gain)
        # print(f"Feature Gains (smallest to largest order):\n{formatted_gain}\n")

        # Calculate the entropy for training year (this will be the predicted entropy)
        pred_entropy = part.calc_feat_entropy(best_pred_feature, True)
        # print(f"Best Predicted Feature for {y_a}: {best_pred_feature}\nPredicted Entropy: {pred_entropy}")

        # Calculate the predicted and actual gain for each year
        actual_tot_entropy, predicted_gain = predict_gain(y_a, years_data, pred_entropy, features)
        actual_gain, year_best_features = calc_actual_gain(y_a, years_data, predicted_gain, features)

        #Printing results
        # print(f"Using {best_pred_feature} as our predicted value, here are the predicted gain in comparison to actual gain.")
        # print("Year | Predicted Gain | Actual Gain | Best Feature")
        # print("-----|----------------|-------------|--------------------")
        # for year in sorted(predicted_gain):
        #     pred = predicted_gain.get(year, "N/A")
        #     actual = actual_gain.get(year, "N/A")
        #     best = year_best_features.get(year, "N/A")
        #     print(f"{year} | {pred:^14.4f} | {actual:^11.4f} | {best}")

        # Plotting predicted vs actual gain by year (using best predicted feature)
        # Prepare data (convert into list)
        other_years = sorted([year for year in years_data if year != y_a])
        actual_gain_lst = [actual_gain[year] for year in other_years]
        predicted_gain_lst = [predicted_gain[year] for year in other_years]

        # Calculate the residuals (finding the RSS)
        rss = calc_rss(y_a, years_data, actual_gain_lst, predicted_gain_lst)
        avg_rss_by_yr[y_a] = [stats.mean(rss), best_pred_feature]
        m,b = np.polyfit(np.array(other_years),np.array(rss),1)
        line = m*np.array(other_years) + b

        plot_rss(rss, y_a, other_years, line)

        # This section compares the gain values to the predicted gain value.
        plot_gain_comparison_w_feat(y_a, year_best_features, predicted_gain, best_pred_feature)

    plt.close()

    # Plot results
    y_plots = sorted(avg_rss_by_yr.keys())
    rss_val_plots = [avg_rss_by_yr[y][0] for y in y_plots]
    best_feat_plots = [avg_rss_by_yr[year][1] for year in y_plots]

    plt.figure(figsize=(18,7))
    plt.bar(y_plots, rss_val_plots)
    plt.xlabel("Year")
    plt.ylabel("RSS (for Actual and Predicted Gain)")
    plt.title("RSS by Year")
    plt.xticks(y_plots)
    plt.grid(axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                f"{rss_val_plots[i]:.8f}",
                ha='center', va='bottom', fontsize=9, rotation=0)


    plt.savefig("figures/rss_by_year.png")
    plt.clf()
    
    # Printing RSS Difference results
    print("\nRSS results")
    print(f"{'Year':<6} | {'RSS':<8} | {'Best Predicted Feature'}")
    print("-" * 40)
    for year in sorted(avg_rss_by_yr):
        rss, feature = avg_rss_by_yr[year]
        print(f"{year:<6} | {rss:<8.8f} | {feature}")
    
    best_year = min(avg_rss_by_yr, key=lambda year: avg_rss_by_yr[year][0])
    min_rss, best_feat = avg_rss_by_yr[best_year]
    print(f"\nBest Feature Based on RSS Results: {best_feat} from {best_year} with {min_rss}")

################################################################################
# HELPER FUNCTIONS
################################################################################

def clean_data():
    """
    This function loads the dataset and cleans the data into a dictionary
    of dataset sorted by year and list of features. 
    """
    #Load dataset
    file_path = "dataset/world_happiness_report.csv"
    df = pd.read_csv(file_path)
    #Remove data with no defined Happiness_Score
    df = df.dropna(subset=["Happiness_Score"])

    #Organize list by year
    years_data = {
        year: group.to_dict(orient='records')
        for year, group in df.groupby('Year')
    } #dictionary of data by year

    smallest_yr = min(years_data.keys())
    train_data = years_data[smallest_yr] # stores data for smallest_yr

    #Set features
    excluded_columns = {'Country', 'Year', 'Happiness_Score'}
    all_columns = train_data[0].keys()
    features = [col for col in all_columns if col not in excluded_columns]

    return years_data, features


def convert_labels(y_matrix):
    """
    Convert the y-labels into discrete values high/low happiness.
    "0" = low; "1" = high
    """
    new_y = []
    for y in y_matrix:
        if y < 5: # We set our threshold to be 5 (exactly between 0-10 scale)
            new_y.append(0)
        else:
            new_y.append(1)
    return new_y
    pass

def group_category_data(yrs_data, feats):
    """
    Because of how small the entropy values are, this function will group the feature 
    values and data associated with the features.
    """
    group_mapping = {
        "GDP_per_Capita": "Economic",
        "Unemployment_Rate": "Economic",
        "Employment_Rate": "Economic",
        "Income_Inequality": "Economic",
        "Public_Health_Expenditure": "Economic",

        "Healthy_Life_Expectancy": "Health",
        "Mental_Health_Index": "Health",
        "Life_Satisfaction": "Health",
        "Work_Life_Balance": "Health",

        "Education_Index": "Education",
        "Internet_Access": "Education",

        "Social_Support": "Social",
        "Generosity": "Social",
        "Public_Trust": "Social",

        "Freedom": "Governance",
        "Corruption_Perception": "Governance",
        "Political_Stability": "Governance",

        "Climate_Index": "Environment",
        "Urbanization_Rate": "Environment",
        "Population": "Environment",

        "Crime_Rate": "Security"
    }
    
    group_categories = sorted(set(group_mapping.values()))
    grouped_years_data = {}

    for year, records in yrs_data.items():
        new_records = []
        for record in records:
            new_record = {
                "Country": record["Country"],
                "Year": record["Year"],
                "Happiness_Score": record["Happiness_Score"]
            }

            # Initialize each category to 1.0 for multiplication
            for category in group_categories:
                new_record[category] = 1.0

            for feature, category in group_mapping.items():
                value = record.get(feature)
                if value is not None and pd.notna(value):
                    new_record[category] *= value

            new_records.append(new_record)

        grouped_years_data[year] = new_records

    return grouped_years_data, group_categories

def set_data(year, train_data):
    """
    The function given a select year, would extract dataset from select 
    year and return a X_base and y_base matrices. 
    """
    excluded_columns = {'Country', 'Year', 'Happiness_Score'}
    all_columns = train_data[0].keys()
    features = [col for col in all_columns if col not in excluded_columns]
    y_base = [row['Happiness_Score'] for row in train_data]
    y_base = convert_labels(y_base)
    X_base = [
    {key: value for key, value in row.items() if key not in excluded_columns}
    for row in train_data]

    #Initializing list of Example objects
    examples = [Example(features=features_dict, label=label)
    for features_dict, label in zip(X_base, y_base)]

    return X_base, y_base, examples

def best_feature(gain_dic):
    """
    Sorts a list based on best gain value (0) to worst (len(dict)).
    """
    #sorts dictionary
    sorted_items = sorted(gain_dic.items(), key=lambda item: item[1])
    sorted_dict = dict(sorted_items)
    #selects best gain feature
    best_key = max(sorted_dict, key=gain_dic.get)

    return sorted_dict, best_key

def predict_gain(year, years_data, pred_entropy, features):
    """
    This function, given the chosen year, will iterate through other years in
    years_data and calculate the predicted gain using actual_tot_entropy and pred_entropy.
    Returns a dictionary of actual entropies and predicted gain by year.
    """
    actual_tot_entropy = {}
    predicted_gain = {}

    #iterate through year in dataset, except for select training year, calculate
    #the predicted gain (using pred_entropy and calculated new entropy)
    for yr in years_data: 
        if yr == year:
            pass
        else:
            curr_year_data = years_data[yr]
            
            #setting dataset format
            curr_X_base, curr_y_base, curr_examples = set_data(yr, curr_year_data)

            # Calculate the entropy and predicted gain
            curr_part = Partition(curr_examples, features)
            actual_tot_entropy[yr] = curr_part.calc_entropy()
            predicted_gain[yr] = actual_tot_entropy[yr] - pred_entropy

    return actual_tot_entropy, predicted_gain

def calc_actual_gain(year, years_data, predicted_gain, features):
    """
    This function will calculate the actual gain values of each year except
    for the selected year (training dataset) using years_data and predicted_gain.
    """
    actual_gain = {}
    year_best_features = {}
    
    for yr in sorted(years_data):
        if yr == year:
            continue

        curr_year_data = years_data[yr]
        curr_X_base, curr_y_base, curr_examples = set_data(yr, curr_year_data)

        curr_part = Partition(curr_examples, features)

        # Calculate best feature and its gain for this year
        gains = curr_part.calc_gain_by_feature()
        gain_ordered, best_feature_curr = best_feature(gains)
        actual_gain[yr] = gains[best_feature_curr]
        year_best_features[yr] = (best_feature_curr, gains[best_feature_curr])

    return actual_gain, year_best_features

def calc_rss(s_year, years_data, acc_gain, pred_gain):
    """
    Calculate the residual sum of squares between the actual and predicted values.
    Returns a list of RSS values per year.
    """
    other_years = sorted([year for year in years_data if year != s_year])

    rss = []
    for i in range(len(other_years)):
        rss.append((acc_gain[i]-pred_gain[i])**2)
    return rss

def plot_rss(rss, year, other_years, line):
    """
    Plot RSS scatter plot with line of best fit. 
    Store output in results folder
    """
    # Plotting residuals and trend using scatter plot.
    plt.figure(figsize=(12, 6))
    plt.scatter(other_years, rss, label="RSS Predicted - Actual Gain", color='blue')
    plt.plot(other_years, line, color='orange', label = 'Best fit line')
    plt.xlabel("Year")
    plt.ylabel("RSS [(actual_gain - pred_gain)**2]")
    plt.title(f"{year} RSS of Predicted and Actual Gain Value by Year")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(ticks=other_years, labels=[str(year) for year in other_years])
    plt.savefig(f"figures/RSS_all_years/{year}_RSS_by_year.png")
    plt.clf()
    plt.close()

def plot_gain_comparison_w_feat(year, year_best_features, predicted_gain, best_pred_feat):
    """
    Plot bar graph comparing actual and predicted gains with best feature labels.
    """
    # Sort years to keep order consistent
    compare_years = sorted([year for year in year_best_features if year in predicted_gain])

    # Gather data
    actual_gains = [year_best_features[year][1] for year in compare_years]
    actual_features = [year_best_features[year][0] for year in compare_years]
    predicted_gains = [predicted_gain[year] for year in compare_years]

    plt.figure(figsize=(20, 12))

    # Plotting bar plot to visualize difference between predicted and actual gain values (diff features)
    x = np.arange(len(compare_years))
    width = 0.35

    plt.figure(figsize=(14, 7))
    bars_actual = plt.bar(x - width/2, actual_gains, width, label="Actual Best Gain", color='skyblue')
    bars_predicted = plt.bar(x + width/2, predicted_gains, width, label="Predicted Gain", color='salmon')

    for i, bar in enumerate(bars_actual):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                f"{actual_features[i]}\n{height:.3f}",
                ha='center', va='bottom', fontsize=7, rotation=90)

    for bar in bars_predicted:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                f"{height:.3f}", ha='center', va='bottom', fontsize=7, rotation=90)

    plt.xlabel("Year")
    plt.ylabel("Gain")
    plt.title(f"{year} Actual Best Feature Gain vs Predicted ({best_pred_feat}) Gain by Year")
    plt.xticks(x, [str(year) for year in compare_years], rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    plt.savefig(f"figures/actual_pred_gain_all_years/{year}_actual_vs_predicted_best_gain_labeled.png")
    plt.clf()
    plt.close()


if __name__=="__main__":
    main()
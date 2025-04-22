import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main():
    df = pd.read_csv("./new/lime_explanations/summary.csv")

    def parse_explanation(text):
        parts = [part.strip() for part in text.split(";") if part.strip()]
        explanation = {}
        for part in parts:
            if ":" in part:
                feat, weight = part.split(":", 1)
                try:
                    explanation[feat.strip()] = float(weight.strip())
                except ValueError:
                    explanation[feat.strip()] = None
        return explanation

    df["explanation_dict"] = df["Explanation"].apply(parse_explanation)

    model_groups = df.groupby("Model")
    rows = []
    for model, group in model_groups:
        feature_sums = {}
        feature_counts = {}
        
        for exp in group["explanation_dict"]:
            for feature, weight in exp.items():
                if weight is not None:
                    feature_sums[feature] = feature_sums.get(feature, 0) + weight
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1

        avg_weights = {feat: feature_sums[feat] / feature_counts[feat] for feat in feature_sums}
        for feat, avg_weight in avg_weights.items():
            rows.append({"Model": model, "Feature": feat, "AverageWeight": avg_weight})

    features_df = pd.DataFrame(rows)
    pivot_df = features_df.pivot(index='Feature', columns='Model', values='AverageWeight')

    top_n = 6
    top_features = set()
    for model in pivot_df.columns:
        top_features_model = pivot_df[model].abs().sort_values(ascending=False).head(top_n).index.tolist()
        top_features.update(top_features_model)

    filtered_pivot_df = pivot_df.loc[list(top_features)]
    
    # Sort features by their mean value (most bullish at top)
    filtered_pivot_df['mean'] = filtered_pivot_df.mean(axis=1)
    filtered_pivot_df = filtered_pivot_df.sort_values('mean', ascending=False)
    filtered_pivot_df = filtered_pivot_df.drop('mean', axis=1)
    
    # Find the maximum absolute value for symmetric color scaling
    max_abs_val = max(abs(filtered_pivot_df.min().min()), abs(filtered_pivot_df.max().max()))

    plt.figure(figsize=(14, 10))
    heatmap = sns.heatmap(
        filtered_pivot_df, 
        annot=True, 
        fmt=".3f", 
        cmap="RdBu_r",  # Red for bullish (positive), Blue for bearish (negative)
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8, "label": "Feature Weight (Bullish > 0 > Bearish)"},
        annot_kws={"size": 12},
        center=0,  # Set 0 as the neutral value
        vmin=-max_abs_val,  # Ensure symmetric color scaling
        vmax=max_abs_val
    )
    
    # Add a horizontal line separating bullish and bearish features based on mean values
    neg_index = filtered_pivot_df.mean(axis=1).ge(0).sum()
    if 0 < neg_index < len(filtered_pivot_df):
        plt.axhline(y=neg_index, color='black', linestyle='-', linewidth=1.5)
    
    plt.title("Average Feature Weights by Model (Bullish > 0 > Bearish)", fontsize=16)
    plt.xticks(fontsize=10, rotation=25, ha="right")
    plt.yticks(fontsize=12)
    
    # Add labels for bullish/bearish sections if we have a division
    if 0 < neg_index < len(filtered_pivot_df):
        plt.text(-0.8, neg_index/2, "BULLISH", fontsize=12, 
                rotation=90, va='center', ha='right', fontweight='bold', color='darkred')
        plt.text(-0.8, (len(filtered_pivot_df) + neg_index)/2, "BEARISH", fontsize=12, 
                rotation=90, va='center', ha='right', fontweight='bold', color='darkblue')
    
    plt.tight_layout()
    plt.savefig('./new/lime_explanations/heatmap_bullish_bearish.png')
    plt.show()

if __name__ == "__main__":
    main()
    print("Exiting Script.")
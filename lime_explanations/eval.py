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
    
    pivot_df['mean'] = pivot_df.mean(axis=1)
    pivot_df['model_count'] = pivot_df.count(axis=1)
    pivot_df['max_abs'] = pivot_df.drop(['mean', 'model_count'], axis=1).abs().max(axis=1)
    
    top_bullish_n = 20
    top_bearish_n = 20
    
    total_models = len(pivot_df.columns) - 3
    pivot_df['bullish_score'] = pivot_df['mean'] * (pivot_df['model_count'] / total_models)
    pivot_df['bearish_score'] = -pivot_df['mean'] * (pivot_df['model_count'] / total_models)
    
    bullish_df = pivot_df[pivot_df['mean'] > 0].sort_values('bullish_score', ascending=False).head(top_bullish_n)
    bullish_features = bullish_df.index.tolist()
    bullish_df = pivot_df.loc[bullish_features].drop(['mean', 'model_count', 'max_abs', 'bullish_score', 'bearish_score'], axis=1)
    
    bearish_df = pivot_df[pivot_df['mean'] < 0].sort_values('bearish_score', ascending=False).head(top_bearish_n)
    bearish_features = bearish_df.index.tolist()
    bearish_df = pivot_df.loc[bearish_features].drop(['mean', 'model_count', 'max_abs', 'bullish_score', 'bearish_score'], axis=1)
    
    bullish_max = bullish_df.max().max()
    bearish_max = abs(bearish_df.min().min())
    
    plt.figure(figsize=(14, len(bullish_df)/2 + 2))
    bullish_heatmap = sns.heatmap(
        bullish_df, 
        annot=True, 
        fmt=".3f", 
        cmap="Reds",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8, "label": "Feature Weight (Bullish)"},
        annot_kws={"size": 12},
        vmin=0,
        vmax=bullish_max
    )
    
    plt.title("Top Bullish Features by Model", fontsize=16)
    plt.xticks(fontsize=10, rotation=25, ha="right")
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./new/lime_explanations/heatmap_bullish.png')
    plt.close()
    
    plt.figure(figsize=(14, len(bearish_df)/2 + 2))
    bearish_heatmap = sns.heatmap(
        bearish_df, 
        annot=True, 
        fmt=".3f", 
        cmap="Blues_r",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8, "label": "Feature Weight (Bearish)"},
        annot_kws={"size": 12},
        vmin=-bearish_max,
        vmax=0
    )
    
    plt.title("Top Bearish Features by Model", fontsize=16)
    plt.xticks(fontsize=10, rotation=25, ha="right")
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./new/lime_explanations/heatmap_bearish.png')
    plt.close()
    
    print("Heatmaps created successfully!")

if __name__ == "__main__":
    main()
    print("Exiting Script.")
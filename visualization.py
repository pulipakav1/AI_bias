import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from config import PLOTS_DIR, TABLES_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# barplot comparing models
def plot_model_comparison(df, metric="sentiment_score"):
    plt.figure(figsize=(14, 8))
    
    model_means = df.groupby("model")[metric].mean().sort_values(ascending=False)
    model_stds = df.groupby("model")[metric].std()
    
    plt.bar(range(len(model_means)), model_means.values, yerr=model_stds.values, capsize=5)
    plt.xticks(range(len(model_means)), model_means.index, rotation=45, ha="right")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Model Comparison: {metric.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison_{metric}.png", dpi=300)
    plt.close()

# boxplot comparing providers
def plot_provider_comparison(df, metric="sentiment_score"):
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(data=df, x="provider", y=metric, palette="Set2")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Provider Comparison: {metric.replace('_', " ").title()}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/provider_comparison_{metric}.png", dpi=300)
    plt.close()

# performance comparison across data
def plot_dataset_comparison(df, metric="sentiment_score"):
    plt.figure(figsize=(14, 8))
    
    dataset_means = df.groupby("dataset")[metric].mean().sort_values()
    dataset_stds = df.groupby("dataset")[metric].std()
    
    plt.barh(range(len(dataset_means)), dataset_means.values, xerr=dataset_stds.values, capsize=5)
    plt.yticks(range(len(dataset_means)), dataset_means.index)
    plt.xlabel(metric.replace("_", " ").title())
    plt.title(f"Dataset Comparison: {metric.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/dataset_comparison_{metric}.png", dpi=300)
    plt.close()

# heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    
    metric_cols = ["sentiment_score", "toxicity_score", "stereotype_score"]
    corr_matrix = df[metric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0)
    plt.title("Correlation Between Bias Metrics")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/metric_correlation_heatmap.png", dpi=300)
    plt.close()

# Model version comparison within each provider
def plot_provider_model_versions(df, metric="sentiment_score"):
    providers = df["provider"].unique()
    n_providers = len(providers)
    
    fig, axes = plt.subplots(1, n_providers, figsize=(6*n_providers, 8))
    if n_providers == 1:
        axes = [axes]
    
    for idx, provider in enumerate(providers):
        provider_df = df[df["provider"] == provider]
        model_means = provider_df.groupby("model")[metric].mean().sort_values(ascending=False)
        model_stds = provider_df.groupby("model")[metric].std()
        
        axes[idx].barh(range(len(model_means)), model_means.values, xerr=model_stds.values, capsize=5)
        axes[idx].set_yticks(range(len(model_means)))
        axes[idx].set_yticklabels(model_means.index, fontsize=9)
        axes[idx].set_xlabel(metric.replace("_", " ").title())
        axes[idx].set_title(f"{provider.upper()} Model Versions")
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.suptitle(f"Model Version Comparison by Provider: {metric.replace('_', ' ').title()}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/provider_model_versions_{metric}.png", dpi=300)
    plt.close()

# Heatmap of all models vs all metrics
def plot_model_metrics_heatmap(df):
    metrics = ["sentiment_score", "toxicity_score", "stereotype_score"]
    
    # Create pivot table
    model_means = df.groupby("model")[metrics].mean()
    model_means = model_means.sort_values("sentiment_score", ascending=False)
    
    plt.figure(figsize=(10, max(8, len(model_means) * 0.4)))
    sns.heatmap(model_means.T, annot=True, fmt=".3f", cmap="RdYlGn", 
                center=model_means.mean().mean(), cbar_kws={'label': 'Score'})
    plt.title("All Models vs All Metrics Heatmap", fontsize=14, fontweight='bold')
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Metric", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_metrics_heatmap.png", dpi=300)
    plt.close()

# Detailed model version breakdown
def plot_model_version_radar(df, model_name):
    model_df = df[df["model"] == model_name]
    if len(model_df) == 0:
        return
    
    metrics = ["sentiment_score", "toxicity_score", "stereotype_score"]
    means = [model_df[m].mean() for m in metrics]
    
    # Normalize to 0-1 scale for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    means += means[:1]  # Complete the circle
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, means, 'o-', linewidth=2, label=model_name)
    ax.fill(angles, means, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title(f"Model Performance Profile: {model_name}", fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_radar_{model_name.replace(' ', '_').replace('/', '_')}.png", dpi=300)
    plt.close()

# Model version detailed comparison table
def create_model_version_table(df, model_version_df=None):
    if model_version_df is None:
        from fairness_metrics import compute_model_version_metrics
        model_version_df = compute_model_version_metrics(df)
    
    # Save detailed model version metrics
    model_version_df.to_csv(f"{TABLES_DIR}/model_version_detailed_metrics.csv", index=False)
    
    # Create a summary table with key metrics
    summary_cols = ["provider", "model", "sentiment_mean", "toxicity_mean", 
                    "stereotype_mean", "sentiment_consistency"]
    if all(col in model_version_df.columns for col in summary_cols):
        summary = model_version_df[summary_cols].copy()
        summary = summary.sort_values("sentiment_mean", ascending=False)
        summary.to_csv(f"{TABLES_DIR}/model_version_summary.csv", index=False)
    
    return model_version_df

# Provider comparison with model versions
def plot_provider_model_comparison(df, metric="sentiment_score"):
    plt.figure(figsize=(16, 10))
    
    # Create grouped bar chart - group by provider, show all models within each provider
    providers = sorted(df["provider"].unique())
    provider_models = {}
    
    for provider in providers:
        provider_df = df[df["provider"] == provider]
        models = sorted(provider_df["model"].unique())
        provider_models[provider] = models
    
    # all unique models across providers for consistent coloring
    all_models = sorted(df["model"].unique())
    max_models_per_provider = max(len(models) for models in provider_models.values())
    
    x = np.arange(len(providers))
    width = 0.8 / max_models_per_provider if max_models_per_provider > 0 else 0.2
    
    # colormap for consistent model coloring
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_models)))
    model_color_map = {model: colors[i] for i, model in enumerate(all_models)}
    
    # Plot bars for each provider
    for provider_idx, provider in enumerate(providers):
        provider_df = df[df["provider"] == provider]
        models = sorted(provider_df["model"].unique())
        
        for model_idx, model in enumerate(models):
            model_data = provider_df[provider_df["model"] == model]
            mean_value = model_data[metric].mean()
            
            offset = (model_idx - len(models) / 2 + 0.5) * width
            plt.bar(provider_idx + offset, mean_value, width, 
                   label=model if provider_idx == 0 else "", 
                   color=model_color_map.get(model, 'gray'), alpha=0.8)
    
    plt.xlabel("Provider", fontsize=12)
    plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
    plt.title(f"Provider Comparison with Model Versions: {metric.replace('_', ' ').title()}", 
              fontsize=14, fontweight='bold')
    plt.xticks(x, providers)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/provider_model_comparison_{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()

# summary 
def create_summary_table(df):
    summary = df.groupby(["provider", "model"]).agg({
        "sentiment_score": ["mean", "std", "min", "max"],
        "toxicity_score": ["mean", "std", "min", "max"],
        "stereotype_score": ["mean", "std", "min", "max"]
    }).round(4)
    
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.to_csv(f"{TABLES_DIR}/summary_statistics.csv")
    return summary

# ANOVA & Post-hoc tests
def perform_statistical_tests(df):
    results = {}
    
    # ANOVA for provider differences
    providers = df["provider"].unique()
    groups = [df[df["provider"] == p]["sentiment_score"].values for p in providers]
    f_stat, p_value = stats.f_oneway(*groups)
    
    results["provider_anova"] = {
        "f_statistic": f_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
    
    # Pairwise t-tests between top models
    top_models = df.groupby("model")["sentiment_score"].mean().nlargest(5).index
    pairwise_results = []
    
    for i, model1 in enumerate(top_models):
        for model2 in top_models[i+1:]:
            group1 = df[df["model"] == model1]["sentiment_score"]
            group2 = df[df["model"] == model2]["sentiment_score"]
            t_stat, p_val = stats.ttest_ind(group1, group2)
            pairwise_results.append({
                "model1": model1,
                "model2": model2,
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < 0.05
            })
    
    results["pairwise_tests"] = pd.DataFrame(pairwise_results)
    results["pairwise_tests"].to_csv(f"{TABLES_DIR}/pairwise_statistical_tests.csv", index=False)
    
    return results


def generate_all_visualizations(df, model_version_df=None):
    print("\nGenerating visualizations")
    
    metrics = ["sentiment_score", "toxicity_score", "stereotype_score"]
    
    # Basic comparisons
    for metric in metrics:
        plot_model_comparison(df, metric)
        plot_provider_comparison(df, metric)
        plot_dataset_comparison(df, metric)
        plot_provider_model_versions(df, metric)
        plot_provider_model_comparison(df, metric)
    
    # Advanced visualizations
    plot_correlation_heatmap(df)
    plot_model_metrics_heatmap(df)
    
    # Create detailed tables
    summary = create_summary_table(df)
    model_version_table = create_model_version_table(df, model_version_df)
    
    # Generate radar charts for top models
    top_models = df.groupby("model")["sentiment_score"].mean().nlargest(10).index
    print(f"\nGenerating radar charts for top {len(top_models)} models...")
    for model in top_models:
        try:
            plot_model_version_radar(df, model)
        except Exception as e:
            print(f"  Warning: Could not generate radar chart for {model}: {e}")
    
    # Statistical tests
    stats_results = perform_statistical_tests(df)
    
    print(f"\nVisualizations saved to {PLOTS_DIR}/")
    print(f"Tables saved to {TABLES_DIR}/")
    print(f"   - summary_statistics.csv")
    print(f"   - model_version_detailed_metrics.csv")
    print(f"   - model_version_summary.csv")
    print(f"   - pairwise_statistical_tests.csv")
    
    return summary, stats_results

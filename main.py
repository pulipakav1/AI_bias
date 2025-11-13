import pandas as pd
from tqdm import tqdm
from datetime import datetime
from config import ALL_MODELS, HF_MODELS, PROMPTS_PER_MODEL, MAX_TOKENS, OUTPUT_DIR
from model_interface import query_model
from data_loader import load_all_datasets
from fairness_metrics import compute_all_fairness_metrics
from visualization import generate_all_visualizations

try:
    from metrics import compute_comprehensive_fairness_metrics, perform_comprehensive_statistical_tests
    COMPREHENSIVE_METRICS_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_METRICS_AVAILABLE = False
    print("Comprehensive metrics not available. Using basic metrics.")

# benchmark pipeline execution
def run_benchmark(include_hf_models=True, hf_only=False):
    """
    Run benchmark on models.
    
    Args:
        include_hf_models: If True, also run HF models (Llama, Gemma). Default: True
        hf_only: If True, run only HF models (exclude regular models). Default: False
    """
    if hf_only:
        model_sets = HF_MODELS.copy()
        print("Models: Llama 3.1, Llama 3.2, Gemma")
    else:
        model_sets = ALL_MODELS.copy()
        if include_hf_models:
            model_sets.update(HF_MODELS)
            print("Models: OpenAI (ChatGPT), Claude, Gemini, Perplexity, Llama 3.1, Llama 3.2, Gemma")
        else:
            print("Models: OpenAI (ChatGPT), Claude, Gemini, Perplexity")
    
    # Load datasets
    print("Loading benchmark datasets:")
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets\n")
    
    results = []
    
    # Run queries for each dataset
    for dataset_name, prompts in datasets.items():
        print(f"TESTING ON DATASET: {dataset_name.upper()}")
        
        # Limit prompts per dataset
        dataset_prompts = prompts[:min(len(prompts), PROMPTS_PER_MODEL // len(datasets))]
        
        for provider, models in model_sets.items():
            print(f"\nProvider: {provider.upper()}")
            
            for model in models:
                print(f"  Running {model}...")
                
                for i, prompt in enumerate(tqdm(dataset_prompts, desc=f"  {model}")):
                    response = query_model(provider, model, prompt, MAX_TOKENS)
                    
                    results.append({
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider,
                        "model": model,
                        "dataset": dataset_name,
                        "prompt_id": i,
                        "prompt": prompt,
                        "response": response
                    })
                
                print(f" Completed {model}\n")
    
    # Save raw results
    df = pd.DataFrame(results)
    raw_file = f"{OUTPUT_DIR}/raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(raw_file, index=False)
    print(f"Raw results saved: {raw_file}")
    
    # Compute fairness metrics
    print("\nComputing fairness and bias metrics:")
    df, metrics = compute_all_fairness_metrics(df)
    
    # Compute comprehensive metrics if available
    if COMPREHENSIVE_METRICS_AVAILABLE:
        print("Computing comprehensive fairness metrics:")
        comprehensive_metrics = {}
        
        # Compute comprehensive fairness metrics by provider
        comprehensive_metrics['provider_fairness'] = compute_comprehensive_fairness_metrics(
            df, group_col="provider", outcome_col="sentiment_score"
        )
        
        # Compute comprehensive fairness metrics by model
        comprehensive_metrics['model_fairness'] = compute_comprehensive_fairness_metrics(
            df, group_col="model", outcome_col="sentiment_score"
        )
        
        # Save comprehensive metrics
        import json
        comprehensive_metrics_file = f"{OUTPUT_DIR}/comprehensive_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert DataFrames to dict for JSON serialization
        comprehensive_metrics_json = {}
        for key, value in comprehensive_metrics.items():
            if isinstance(value, dict):
                comprehensive_metrics_json[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        comprehensive_metrics_json[key][sub_key] = sub_value.to_dict('records')
                    else:
                        comprehensive_metrics_json[key][sub_key] = sub_value
        
        with open(comprehensive_metrics_file, 'w') as f:
            json.dump(comprehensive_metrics_json, f, indent=2, default=str)
        print(f"Comprehensive metrics saved: {comprehensive_metrics_file}")
        
        # Add to main metrics dict
        metrics['comprehensive_metrics'] = comprehensive_metrics
    
    metrics_file = f"{OUTPUT_DIR}/results_with_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(metrics_file, index=False)
    print(f"Results with metrics saved: {metrics_file}")
    
    # Generate visualizations and statistical analysis
    model_version_df = metrics.get("model_version_details", None)
    summary, stats_results = generate_all_visualizations(df, model_version_df)
    
    # summary
    print("SUMMARY:")
    print(f"\n Total queries: {len(results)}")
    print(f" Unique models: {df['model'].nunique()}")
    print(f" Providers tested: {df['provider'].nunique()}")
    print(f" Datasets tested: {df['dataset'].nunique()}")
    
    print(f"\n Overall Scores:")
    print(f"  Mean Sentiment Score: {metrics['mean_sentiment']:.4f} (±{metrics['std_sentiment']:.4f})")
    print(f"  Mean Toxicity Score: {metrics['mean_toxicity']:.4f} (±{metrics['std_toxicity']:.4f})")
    print(f"  Mean Stereotype Score: {metrics['mean_stereotype']:.4f} (±{metrics['std_stereotype']:.4f})")
    
    # top models by each metric
    if model_version_df is not None and len(model_version_df) > 0:
        print(f"\n Top 5 Models by Sentiment Score:")
        top_sentiment = model_version_df.nlargest(5, 'sentiment_mean')[['model', 'sentiment_mean', 'toxicity_mean', 'stereotype_mean']]
        for idx, row in top_sentiment.iterrows():
            print(f"  {row['model']}: Sentiment={row['sentiment_mean']:.4f}, Toxicity={row['toxicity_mean']:.4f}, Stereotype={row['stereotype_mean']:.4f}")
        
        print(f"\n Top 5 Models by Lowest Toxicity:")
        top_toxicity = model_version_df.nsmallest(5, 'toxicity_mean')[['model', 'sentiment_mean', 'toxicity_mean', 'stereotype_mean']]
        for idx, row in top_toxicity.iterrows():
            print(f"  {row['model']}: Sentiment={row['sentiment_mean']:.4f}, Toxicity={row['toxicity_mean']:.4f}, Stereotype={row['stereotype_mean']:.4f}")
        
        print(f"\n Top 5 Models by Lowest Stereotype Score:")
        top_stereotype = model_version_df.nsmallest(5, 'stereotype_mean')[['model', 'sentiment_mean', 'toxicity_mean', 'stereotype_mean']]
        for idx, row in top_stereotype.iterrows():
            print(f"  {row['model']}: Sentiment={row['sentiment_mean']:.4f}, Toxicity={row['toxicity_mean']:.4f}, Stereotype={row['stereotype_mean']:.4f}")
    

    print("BENCHMARK COMPLETE")
  

def run_hf_models_benchmark():
    """Run benchmark on Hugging Face models only."""
    return run_benchmark(include_hf_models=True, hf_only=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--hf-only":
        # Run only HF models (Llama, Gemma)
        run_benchmark(include_hf_models=True, hf_only=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--no-hf":
        # Run without HF models
        run_benchmark(include_hf_models=False, hf_only=False)
    else:
        # Default: Run all models (including HF)
        run_benchmark(include_hf_models=True, hf_only=False)

import pandas as pd
from tqdm import tqdm
from datetime import datetime
from config import OPENAI_MODELS, GEMINI_MODELS, PROMPTS_PER_MODEL, MAX_TOKENS, OUTPUT_DIR
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


def run_openai_gemini_benchmark():
    """
    Run benchmark on OpenAI and Gemini models only.
    """
    # Define model sets for OpenAI and Gemini only
    model_sets = {
        "openai": OPENAI_MODELS,
        "gemini": GEMINI_MODELS
    }
    
    print("Models: OpenAI (ChatGPT), Gemini")
    print(f"OpenAI models: {', '.join(OPENAI_MODELS)}")
    print(f"Gemini models: {', '.join(GEMINI_MODELS)}\n")
    
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
    raw_file = f"{OUTPUT_DIR}/raw_results_openai_gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
        comprehensive_metrics_file = f"{OUTPUT_DIR}/comprehensive_metrics_openai_gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
    
    metrics_file = f"{OUTPUT_DIR}/results_with_metrics_openai_gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(metrics_file, index=False)
    print(f"Results with metrics saved: {metrics_file}")
    
    # Generate visualizations and statistical analysis
    model_version_df = metrics.get("model_version_details", None)
    summary, stats_results = generate_all_visualizations(df, model_version_df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"\n Total queries: {len(results)}")
    print(f" Unique models: {df['model'].nunique()}")
    print(f" Providers tested: {df['provider'].nunique()}")
    print(f" Datasets tested: {df['dataset'].nunique()}")
    
    print(f"\n Overall Scores:")
    print(f"  Mean Sentiment Score: {metrics['mean_sentiment']:.4f} (±{metrics['std_sentiment']:.4f})")
    print(f"  Mean Toxicity Score: {metrics['mean_toxicity']:.4f} (±{metrics['std_toxicity']:.4f})")
    print(f"  Mean Stereotype Score: {metrics['mean_stereotype']:.4f} (±{metrics['std_stereotype']:.4f})")
    
    # Top models by each metric
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
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_openai_gemini_benchmark()


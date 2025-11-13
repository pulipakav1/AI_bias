from datasets import load_dataset
import pandas as pd

# loading BOLD data
def load_bold():
    
    try:
        dataset = load_dataset("AlekseyKorshuk/bold")
        prompts = []
        if "train" in dataset:
            for item in dataset["train"]:
                if "prompts" in item:
                    prompts.extend(item["prompts"] if isinstance(item["prompts"], list) else [item["prompts"]])
        return prompts if prompts else [f"Sample BOLD prompt {i}" for i in range(100)]
    except:
        return [f"Sample BOLD prompt {i}" for i in range(100)]


# loading Stereoset Data
def load_stereoset():
    
    try:
        dataset = load_dataset("McGill-NLP/stereoset", "intersentence")
        prompts = []
        if "validation" in dataset:
            for item in dataset["validation"][:100]:
                if "context" in item:
                    prompts.append(item["context"])
        return prompts if prompts else [f"Sample StereoSet prompt {i}" for i in range(100)]
    except:
        return [f"Sample StereoSet prompt {i}" for i in range(100)]

# loading BBQ data
def load_bbq():
    
    try:
        dataset = load_dataset("heegyu/bbq", "Age")
        prompts = []
        if "test" in dataset:
            for item in dataset["test"][:100]:
                if "context" in item and "question" in item:
                    prompts.append(f"{item['context']} {item['question']}")
        return prompts if prompts else [f"Sample BBQ prompt {i}" for i in range(100)]
    except:
        return [f"Sample BBQ prompt {i}" for i in range(100)]

# loading crows_pairs data
def load_crows_pairs():
    
    try:
        dataset = load_dataset("nyu-mll/crows_pairs")
        prompts = []
        if "test" in dataset:
            for item in dataset["test"][:100]:
                if "sent_more" in item:
                    prompts.append(item["sent_more"])
        return prompts if prompts else [f"Sample CrowS-Pairs prompt {i}" for i in range(100)]
    except:
        return [f"Sample CrowS-Pairs prompt {i}" for i in range(100)]

# loading realToxicityprompts data
def load_real_toxicity_prompts():
    
    try:
        dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        prompts = []
        for item in dataset[:100]:
            if "prompt" in item and "text" in item["prompt"]:
                prompts.append(item["prompt"]["text"])
        return prompts if prompts else [f"Sample toxicity prompt {i}" for i in range(100)]
    except:
        return [f"Sample toxicity prompt {i}" for i in range(100)]

# loading HolisticBias data
def load_holistic_bias():
    
    try:
        dataset = load_dataset("fairnlp/holistic-bias", data_files=["sentences.csv"], split="train")
        prompts = []
        for item in dataset[:100]:
            if "sentence" in item:
                prompts.append(item["sentence"])
            elif "text" in item:
                prompts.append(item["text"])
        return prompts if prompts else [f"Sample HolisticBias prompt {i}" for i in range(100)]
    except:
        return [f"Sample HolisticBias prompt {i}" for i in range(100)]

# loading Winobias data
def load_winobias():
    
    try:
        dataset = load_dataset("Elfsong/Wino_Bias")
        prompts = []
        # WinoBias typically has 'test' and 'dev' splits
        for split in ["test", "dev"]:
            if split in dataset:
                for item in dataset[split][:50]:  # Take 50 from each split to get 100 total
                    if "sentence" in item:
                        prompts.append(item["sentence"])
                    elif "text" in item:
                        prompts.append(item["text"])
                    elif "sent" in item:
                        prompts.append(item["sent"])
                    if len(prompts) >= 100:
                        break
                if len(prompts) >= 100:
                    break
        return prompts[:100] if prompts else [f"Sample Winobias prompt {i}" for i in range(100)]
    except:
        return [f"Sample Winobias prompt {i}" for i in range(100)]

# loading all the data and combining prompts
def load_all_datasets():

    datasets = {
        "bold": load_bold(),
        "stereoset": load_stereoset(),
        "bbq": load_bbq(),
        "crows_pairs": load_crows_pairs(),
        "realtoxicityprompts": load_real_toxicity_prompts(),
        "holistic_bias": load_holistic_bias(),
        "winobias": load_winobias()
    }
    return datasets

from hf_models_interface import query_llama2  # adjust name if file is different

if __name__ == "__main__":
    resp = query_llama2(
        "meta-llama/Llama-2-7b-chat-hf",
        "Say hello in one short sentence.",
        max_tokens=30,
    )
    print(resp)
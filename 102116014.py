
# ============================================================
# CHATBOT EVALUATION USING TOPSIS
# Accuracy + Response Time + Perplexity
# With Weight Sensitivity Analysis
# ============================================================

# INSTALL:
# pip install pandas numpy matplotlib seaborn
# pip install sentence-transformers huggingface_hub
# pip install torch transformers accelerate
# pip install scipy

import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================



if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Set your Hugging Face token."
    )

np.random.seed(0)

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.0

# Primary TOPSIS weights
# Accuracy = 50%
# Response Time = 25%
# Perplexity = 25%

WEIGHTS = np.array([0.50, 0.25, 0.25])

CRITERIA = [
    "Accuracy",
    "Response_Time",
    "Perplexity"
]

CRITERIA_TYPES = [
    "benefit",
    "cost",
    "cost"
]

# Strict mode: all three metrics must be available
REQUIRE_ALL_METRICS = True

# Local model paths for perplexity calculation.
# Set these to compatible local Transformers model directories.
#
# GGML/GGUF files are NOT directly compatible with
# AutoModelForCausalLM. They require llama.cpp instead.

LOCAL_MODEL_PATHS = {
    "GPT2_UK": None,
    "Llama2_Medical": None,
    "Medical_ChatBot": None,
    "Psychology_Chatbot": None,
    "Gemini_Chatbot": None
}


# ============================================================
# STEP 2: DEFINE MODELS
# ============================================================

MODELS = {
    "GPT2_UK": {
        "repo": "robinhad/gpt2-uk-conversational",
        "mode": "text"
    },

    "Llama2_Medical": {
        "repo": "ThisIs-Developer/Llama-2-GGML-Medical-Chatbot",
        "mode": "chat"
    },

    "Medical_ChatBot": {
        "repo": "Mohammed-Altaf/Medical-ChatBot",
        "mode": "chat"
    },

    "Psychology_Chatbot": {
        "repo": "Israr-dawar/psychology_chatbot",
        "mode": "chat"
    },

    "Gemini_Chatbot": {
        "repo": "tws-pappu/Gemini_AI_Chatbot",
        "mode": "chat"
    }
}

# Provider IDs can differ from repository IDs.
# Replace these values with actual deployed provider model IDs.
#
# Example:
# PROVIDER_MODELS["GPT2_UK"] = "provider/model-id"

PROVIDER_MODELS = {
    name: config["repo"]
    for name, config in MODELS.items()
}


# ============================================================
# STEP 3: EVALUATION DATASET
# ============================================================

evaluation_data = [

    {
        "question": "What is the capital of France?",
        "reference": "Paris."
    },

    {
        "question": "What is the powerhouse of the cell?",
        "reference": "The mitochondrion is the powerhouse of the cell."
    },

    {
        "question": "What does HTTP stand for?",
        "reference": "Hypertext Transfer Protocol."
    },

    {
        "question": "What is the boiling point of water at sea level?",
        "reference": "100 degrees Celsius."
    },

    {
        "question": "What is photosynthesis?",
        "reference": "Photosynthesis is the process by which plants convert light energy into chemical energy."
    },

    {
        "question": "What is the chemical symbol for oxygen?",
        "reference": "O."
    },

    {
        "question": "Which planet is known as the Red Planet?",
        "reference": "Mars."
    },

    {
        "question": "What is the largest organ in the human body?",
        "reference": "The skin."
    },

    {
        "question": "What is the function of red blood cells?",
        "reference": "Red blood cells transport oxygen throughout the body."
    },

    {
        "question": "What is the main function of the heart?",
        "reference": "The heart pumps blood throughout the body."
    },

    {
        "question": "What is machine learning?",
        "reference": "Machine learning is a field of artificial intelligence in which systems learn patterns from data."
    },

    {
        "question": "What is artificial intelligence?",
        "reference": "Artificial intelligence is the development of systems capable of performing tasks that typically require human intelligence."
    },

    {
        "question": "What is the function of the kidneys?",
        "reference": "The kidneys filter waste and excess fluid from the blood and produce urine."
    },

    {
        "question": "What is DNA?",
        "reference": "DNA is deoxyribonucleic acid, the molecule that carries genetic information."
    },

    {
        "question": "What is the primary function of neurons?",
        "reference": "Neurons transmit information through electrical and chemical signals."
    },

    {
        "question": "What is the SI unit of force?",
        "reference": "The newton."
    },

    {
        "question": "What is the square root of 144?",
        "reference": "12."
    },

    {
        "question": "What is the purpose of the lungs?",
        "reference": "The lungs exchange oxygen and carbon dioxide during breathing."
    },

    {
        "question": "What is the function of insulin?",
        "reference": "Insulin helps regulate blood glucose levels."
    },

    {
        "question": "What is the difference between hardware and software?",
        "reference": "Hardware refers to physical computer components, while software consists of programs and instructions."
    }
]

dataset = pd.DataFrame(evaluation_data)

print("Dataset loaded:", len(dataset), "questions")


# ============================================================
# STEP 4: LOAD SEMANTIC EVALUATION MODEL
# ============================================================

print("\nLoading semantic similarity model...")

similarity_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Semantic model loaded.")


# ============================================================
# STEP 5: MODEL RESPONSE GENERATION
# ============================================================

def generate_response(model_name, question):

    config = MODELS[model_name]

    model_id = PROVIDER_MODELS[model_name]

    client = InferenceClient(
        model=model_id,
        token=HF_TOKEN,
        timeout=120
    )

    start_time = time.perf_counter()

    if config["mode"] == "chat":

        result = client.chat_completion(

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer accurately and directly. "
                        "Keep the response concise."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            max_tokens=MAX_NEW_TOKENS,

            temperature=TEMPERATURE
        )

        response = result.choices[0].message.content

    elif config["mode"] == "text":

        prompt = (
            "Question: " + question +
            "\nAnswer:"
        )

        response = client.text_generation(

            prompt,

            max_new_tokens=MAX_NEW_TOKENS,

            temperature=TEMPERATURE,

            do_sample=False,

            return_full_text=False
        )

    else:
        raise ValueError("Unsupported model mode.")

    elapsed_time = time.perf_counter() - start_time

    return str(response or "").strip(), elapsed_time


# ============================================================
# STEP 6: SEMANTIC ACCURACY
# ============================================================

def calculate_accuracy(reference, response):

    if not response.strip():
        return 0.0

    embeddings = similarity_model.encode(
        [reference, response],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]

    return float(np.clip(similarity, 0, 1))


# ============================================================
# STEP 7: LOCAL PERPLEXITY MODEL LOADING
# ============================================================

perplexity_models = {}


def load_perplexity_model(model_name):

    if model_name in perplexity_models:
        return perplexity_models[model_name]

    path = LOCAL_MODEL_PATHS.get(model_name)

    if not path:
        return None, None

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        token=HF_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        token=HF_TOKEN,
        torch_dtype=(
            torch.float16
            if device == "cuda"
            else torch.float32
        )
    )

    model.to(device)
    model.eval()

    perplexity_models[model_name] = (
        model,
        tokenizer
    )

    return model, tokenizer


# ============================================================
# STEP 8: CALCULATE PERPLEXITY
# ============================================================

def calculate_perplexity(model_name, text):

    loaded = load_perplexity_model(model_name)

    if loaded is None:
        return np.nan

    model, tokenizer = loaded

    if not text.strip():
        return np.nan

    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if inputs["input_ids"].shape[1] < 2:
        return np.nan

    with torch.no_grad():

        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

        loss = outputs.loss

    # Prevent numerical overflow
    perplexity = math.exp(
        min(loss.item(), 20)
    )

    return float(perplexity)


# ============================================================
# STEP 9: RUN MODEL EVALUATION
# ============================================================

results = []

for model_name in MODELS:

    print("\n" + "=" * 60)
    print("Evaluating:", model_name)
    print("=" * 60)

    for index, row in dataset.iterrows():

        question = row["question"]
        reference = row["reference"]

        print(
            f"\nQuestion {index + 1}/{len(dataset)}"
        )

        try:

            response, response_time = generate_response(
                model_name,
                question
            )

            accuracy = calculate_accuracy(
                reference,
                response
            )

            # Perplexity is calculated on the generated response.
            # It requires a compatible local model.

            perplexity = calculate_perplexity(
                model_name,
                response
            )

            results.append({

                "Model": model_name,

                "Question": question,

                "Reference": reference,

                "Response": response,

                "Accuracy": accuracy,

                "Response_Time": response_time,

                "Perplexity": perplexity,

                "Status": "Success"

            })

            print("Response:", response)

            print(f"Accuracy: {accuracy:.4f}")

            print(
                f"Response Time: {response_time:.2f}s"
            )

            print("Perplexity:", perplexity)

        except Exception as e:

            print("ERROR:", str(e))

            results.append({

                "Model": model_name,

                "Question": question,

                "Reference": reference,

                "Response": "",

                "Accuracy": np.nan,

                "Response_Time": np.nan,

                "Perplexity": np.nan,

                "Status": "Failed: " + str(e)[:200]

            })


# ============================================================
# STEP 10: SAVE DETAILED RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "chatbot_evaluation_details.csv",
    index=False
)

print("\nDetailed results saved.")


# ============================================================
# STEP 11: AGGREGATE METRICS
# ============================================================

successful_results = results_df[
    results_df["Status"] == "Success"
].copy()

summary = successful_results.groupby("Model").agg(

    Accuracy=("Accuracy", "mean"),

    Response_Time=("Response_Time", "mean"),

    Perplexity=("Perplexity", "mean"),

    Successful_Responses=("Response", "count")

).reset_index()

summary["Success_Rate"] = (
    summary["Successful_Responses"] / len(dataset)
)

summary.to_csv(
    "model_performance.csv",
    index=False
)

print("\nMODEL PERFORMANCE")

print(summary.to_string(index=False))


# ============================================================
# STEP 12: VALIDATE METRICS FOR TOPSIS
# ============================================================

ranking_data = summary[
    [
        "Model",
        "Accuracy",
        "Response_Time",
        "Perplexity"
    ]
].copy()

if REQUIRE_ALL_METRICS:

    ranking_data = ranking_data.dropna(
        subset=CRITERIA
    )

if len(ranking_data) < 2:

    raise ValueError(
        "\nTOPSIS cannot run with fewer than two models "
        "having valid measurements for all three metrics.\n"
        "Configure local model paths or compatible "
        "perplexity scoring before ranking."
    )


# ============================================================
# STEP 13: TOPSIS FUNCTION
# ============================================================

def topsis(data, weights):

    decision_matrix = data[
        CRITERIA
    ].to_numpy(dtype=float)

    if not np.isfinite(decision_matrix).all():
        raise ValueError("Invalid values in decision matrix.")

    # Vector normalization
    denominator = np.sqrt(
        np.sum(decision_matrix ** 2, axis=0)
    )

    denominator[denominator == 0] = 1

    normalized_matrix = (
        decision_matrix / denominator
    )

    # Normalize weights
    weights = np.asarray(weights, dtype=float)

    weights = weights / weights.sum()

    # Weighted normalized matrix
    weighted_matrix = (
        normalized_matrix * weights
    )

    # Ideal and negative ideal
    ideal_solution = np.zeros(3)

    negative_ideal_solution = np.zeros(3)

    for j, criterion in enumerate(CRITERIA_TYPES):

        if criterion == "benefit":

            ideal_solution[j] = np.max(
                weighted_matrix[:, j]
            )

            negative_ideal_solution[j] = np.min(
                weighted_matrix[:, j]
            )

        elif criterion == "cost":

            ideal_solution[j] = np.min(
                weighted_matrix[:, j]
            )

            negative_ideal_solution[j] = np.max(
                weighted_matrix[:, j]
            )

    # Separation measures
    distance_ideal = np.sqrt(
        np.sum(
            (weighted_matrix - ideal_solution) ** 2,
            axis=1
        )
    )

    distance_negative = np.sqrt(
        np.sum(
            (weighted_matrix - negative_ideal_solution) ** 2,
            axis=1
        )
    )

    denominator = distance_ideal + distance_negative

    scores = np.divide(
        distance_negative,
        denominator,
        out=np.zeros_like(distance_negative),
        where=denominator != 0
    )

    return scores


# ============================================================
# STEP 14: PRIMARY TOPSIS RANKING
# ============================================================

ranking_data["TOPSIS_Score"] = topsis(
    ranking_data,
    WEIGHTS
)

ranking_data["Rank"] = (
    ranking_data["TOPSIS_Score"]
    .rank(ascending=False, method="min")
    .astype(int)
)

ranking_data = ranking_data.sort_values("Rank")

ranking_data.to_csv(
    "topsis_ranking.csv",
    index=False
)

print("\n" + "=" * 70)
print("THREE-METRIC TOPSIS RANKING")
print("=" * 70)

print(
    ranking_data[
        [
            "Model",
            "Accuracy",
            "Response_Time",
            "Perplexity",
            "TOPSIS_Score",
            "Rank"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 15: WEIGHT SENSITIVITY ANALYSIS
# ============================================================

weight_scenarios = {

    "Accuracy_Focused": [0.70, 0.15, 0.15],

    "Balanced": [0.50, 0.25, 0.25],

    "Speed_Focused": [0.30, 0.50, 0.20],

    "Perplexity_Focused": [0.30, 0.20, 0.50],

    "Equal_Weights": [1/3, 1/3, 1/3]

}

sensitivity_results = []

for scenario, weights in weight_scenarios.items():

    temp = ranking_data[
        [
            "Model",
            "Accuracy",
            "Response_Time",
            "Perplexity"
        ]
    ].copy()

    temp["TOPSIS_Score"] = topsis(
        temp,
        np.array(weights)
    )

    temp["Rank"] = (
        temp["TOPSIS_Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    temp["Scenario"] = scenario

    temp["Accuracy_Weight"] = weights[0]

    temp["Response_Time_Weight"] = weights[1]

    temp["Perplexity_Weight"] = weights[2]

    sensitivity_results.append(temp)

sensitivity_df = pd.concat(
    sensitivity_results,
    ignore_index=True
)

sensitivity_df.to_csv(
    "TOPSIS_Sensitivity_Analysis.csv",
    index=False
)

print("\nWEIGHT SENSITIVITY ANALYSIS")

print(
    sensitivity_df[
        [
            "Scenario",
            "Model",
            "TOPSIS_Score",
            "Rank"
        ]
    ].sort_values(
        ["Scenario", "Rank"]
    ).to_string(index=False)
)


# ============================================================
# STEP 16: ACCURACY COMPARISON
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))

sns.barplot(
    data=ranking_data,
    x="Model",
    y="Accuracy"
)

plt.title("Chatbot Accuracy Comparison")

plt.xlabel("Model")

plt.ylabel("Mean Semantic Similarity")

plt.xticks(rotation=40, ha="right")

plt.tight_layout()

plt.savefig(
    "Accuracy_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 17: RESPONSE TIME COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=ranking_data,
    x="Model",
    y="Response_Time"
)

plt.title("Average Response Time Comparison")

plt.xlabel("Model")

plt.ylabel("Response Time (seconds)")

plt.xticks(rotation=40, ha="right")

plt.tight_layout()

plt.savefig(
    "Response_Time_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 18: PERPLEXITY COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=ranking_data,
    x="Model",
    y="Perplexity"
)

plt.title("Model Perplexity Comparison")

plt.xlabel("Model")

plt.ylabel("Perplexity")

plt.xticks(rotation=40, ha="right")

plt.tight_layout()

plt.savefig(
    "Perplexity_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 19: TOPSIS SCORE COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=ranking_data,
    x="Model",
    y="TOPSIS_Score"
)

plt.title("TOPSIS Score Comparison")

plt.xlabel("Model")

plt.ylabel("TOPSIS Score")

plt.xticks(rotation=40, ha="right")

plt.tight_layout()

plt.savefig(
    "TOPSIS_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 20: ACCURACY VS RESPONSE TIME
# ============================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=ranking_data,
    x="Response_Time",
    y="Accuracy",
    hue="Model",
    s=180
)

plt.title("Accuracy vs Response Time")

plt.xlabel("Average Response Time (seconds)")

plt.ylabel("Semantic Accuracy")

plt.tight_layout()

plt.savefig(
    "Accuracy_vs_Response_Time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 21: WEIGHT SENSITIVITY VISUALISATION
# ============================================================

plt.figure(figsize=(12, 7))

sns.lineplot(
    data=sensitivity_df,
    x="Scenario",
    y="Rank",
    hue="Model",
    marker="o",
    linewidth=2
)

plt.title("TOPSIS Rank Sensitivity to Criteria Weights")

plt.xlabel("Weight Scenario")

plt.ylabel("Rank")

plt.gca().invert_yaxis()

plt.xticks(rotation=25, ha="right")

plt.tight_layout()

plt.savefig(
    "TOPSIS_Weight_Sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 22: FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("MODEL WITH HIGHEST TOPSIS SCORE")
print("=" * 70)

best_model = ranking_data.iloc[0]

print("Model:", best_model["Model"])

print(
    "TOPSIS Score:",
    round(best_model["TOPSIS_Score"], 4)
)

print("Rank:", best_model["Rank"])

print(
    "Accuracy:",
    round(best_model["Accuracy"], 4)
)

print(
    "Response Time:",
    round(best_model["Response_Time"], 4),
    "seconds"
)

print(
    "Perplexity:",
    round(best_model["Perplexity"], 4)
)

print("\nAll evaluation steps completed.")


# ============================================================
# GENERATED FILES
# ============================================================

# 1. chatbot_evaluation_details.csv
# 2. model_performance.csv
# 3. topsis_ranking.csv
# 4. TOPSIS_Sensitivity_Analysis.csv
# 5. Accuracy_Comparison.png
# 6. Response_Time_Comparison.png
# 7. Perplexity_Comparison.png
# 8. TOPSIS_Comparison.png
# 9. Accuracy_vs_Response_Time.png
# 10. TOPSIS_Weight_Sensitivity.png

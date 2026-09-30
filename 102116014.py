
# ============================================================
# CHATBOT PERFORMANCE EVALUATION USING TOPSIS
# ============================================================
#
# Models:
# 1. robinhad/gpt2-uk-conversational
# 2. ThisIs-Developer/Llama-2-GGML-Medical-Chatbot
# 3. Mohammed-Altaf/Medical-ChatBot
# 4. Israr-dawar/psychology_chatbot
# 5. tws-pappu/Gemini_AI_Chatbot
#
# Metrics:
# - Semantic Accuracy
# - Average Response Time
# - TOPSIS Score
#
# Outputs:
# - chatbot_evaluation_details.csv
# - model_performance.csv
# - topsis_ranking.csv
# - Accuracy_Comparison.png
# - Response_Time_Comparison.png
# - TOPSIS_Comparison.png
# - Accuracy_vs_Response_Time.png
#
# ============================================================


# ============================================================
# STEP 1: INSTALL REQUIRED LIBRARIES
# ============================================================

# Run separately in terminal / Colab:
#
# pip install pandas numpy matplotlib seaborn
# pip install sentence-transformers huggingface_hub scikit-learn


# ============================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STEP 3: HUGGING FACE TOKEN
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Set your Hugging Face token "
        "as an environment variable."
    )


# ============================================================
# STEP 4: DEFINE MODELS
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


# IMPORTANT:
# If a model is not deployed through a Hugging Face inference
# provider, replace its repo with a supported provider model ID.
#
# The repository ID and provider model ID are not necessarily
# interchangeable.
#
# The Llama GGML repository may require local GGUF/GGML inference.
# Do not assume that its repository can be called through chat API.


# ============================================================
# STEP 5: CONFIGURATION
# ============================================================

MAX_NEW_TOKENS = 150

TEMPERATURE = 0.0

# TOPSIS weights
# Accuracy = 70%
# Response Time = 30%

WEIGHTS = np.array([0.7, 0.3])

# Accuracy = Benefit
# Response Time = Cost

CRITERIA_TYPES = ["benefit", "cost"]


# ============================================================
# STEP 6: EVALUATION DATASET
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
# STEP 7: LOAD SEMANTIC SIMILARITY MODEL
# ============================================================

print("\nLoading semantic similarity model...")

similarity_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Semantic model loaded successfully.")


# ============================================================
# STEP 8: GENERATE MODEL RESPONSES
# ============================================================

def generate_response(model_name, question):

    model_config = MODELS[model_name]

    repo = model_config["repo"]

    mode = model_config["mode"]

    client = InferenceClient(
        model=repo,
        token=HF_TOKEN,
        timeout=120
    )

    start_time = time.perf_counter()

    if mode == "chat":

        result = client.chat_completion(

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question accurately. "
                        "Be concise and directly address the question."
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

    elif mode == "text":

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

        raise ValueError("Unsupported inference mode.")

    elapsed_time = time.perf_counter() - start_time

    if response is None:
        response = ""

    return str(response).strip(), elapsed_time


# ============================================================
# STEP 9: CALCULATE SEMANTIC ACCURACY
# ============================================================

def calculate_accuracy(reference, response):

    if not response or not response.strip():
        return 0.0

    embeddings = similarity_model.encode(
        [reference, response],
        normalize_embeddings=True
    )

    score = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]

    # Convert similarity to a 0-1 range
    score = np.clip(score, 0, 1)

    return float(score)


# ============================================================
# STEP 10: EVALUATE ALL MODELS
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

            results.append({

                "Model": model_name,

                "Question": question,

                "Reference": reference,

                "Response": response,

                "Accuracy": accuracy,

                "Response_Time": response_time,

                "Status": "Success"

            })

            print("Response:", response)

            print(f"Semantic Accuracy: {accuracy:.4f}")

            print(f"Response Time: {response_time:.2f} seconds")

        except Exception as e:

            print("ERROR:", str(e))

            results.append({

                "Model": model_name,

                "Question": question,

                "Reference": reference,

                "Response": "",

                "Accuracy": np.nan,

                "Response_Time": np.nan,

                "Status": "Failed: " + str(e)[:200]

            })


# ============================================================
# STEP 11: SAVE INDIVIDUAL EVALUATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "chatbot_evaluation_details.csv",
    index=False
)

print("\nDetailed evaluation saved.")


# ============================================================
# STEP 12: CALCULATE AGGREGATE METRICS
# ============================================================

successful_results = results_df[
    results_df["Status"] == "Success"
].copy()

if successful_results.empty:
    raise RuntimeError(
        "No models returned successful responses. "
        "Check inference provider access and model compatibility."
    )

summary = successful_results.groupby("Model").agg(

    Accuracy=("Accuracy", "mean"),

    Response_Time=("Response_Time", "mean"),

    Successful_Responses=("Response", "count")

).reset_index()

summary["Success_Rate"] = (
    summary["Successful_Responses"] / len(dataset)
)

summary.to_csv(
    "model_performance.csv",
    index=False
)

print("\nMODEL PERFORMANCE SUMMARY")

print(summary.to_string(index=False))


# ============================================================
# STEP 13: TOPSIS IMPLEMENTATION
# ============================================================

def topsis(data, weights, criteria_types):

    decision_matrix = data[
        ["Accuracy", "Response_Time"]
    ].to_numpy(dtype=float)

    if not np.isfinite(decision_matrix).all():
        raise ValueError(
            "Invalid values found in decision matrix."
        )

    # Vector normalization
    denominator = np.sqrt(
        np.sum(decision_matrix ** 2, axis=0)
    )

    denominator[denominator == 0] = 1

    normalized_matrix = (
        decision_matrix / denominator
    )

    # Normalize weights
    weights = weights / np.sum(weights)

    # Weighted normalized matrix
    weighted_matrix = (
        normalized_matrix * weights
    )

    # Ideal solutions
    ideal_solution = np.zeros(
        weighted_matrix.shape[1]
    )

    negative_ideal_solution = np.zeros(
        weighted_matrix.shape[1]
    )

    for j, criterion in enumerate(criteria_types):

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

    # Distance from ideal solution
    distance_ideal = np.sqrt(
        np.sum(
            (weighted_matrix - ideal_solution) ** 2,
            axis=1
        )
    )

    # Distance from negative ideal solution
    distance_negative = np.sqrt(
        np.sum(
            (weighted_matrix - negative_ideal_solution) ** 2,
            axis=1
        )
    )

    # TOPSIS score
    denominator = distance_ideal + distance_negative

    scores = np.divide(
        distance_negative,
        denominator,
        out=np.zeros_like(distance_negative),
        where=denominator != 0
    )

    return scores


# ============================================================
# STEP 14: RANK MODELS
# ============================================================

ranking_data = summary[
    [
        "Model",
        "Accuracy",
        "Response_Time",
        "Successful_Responses",
        "Success_Rate"
    ]
].copy()

ranking_data["TOPSIS_Score"] = topsis(
    ranking_data,
    WEIGHTS,
    CRITERIA_TYPES
)

ranking_data["Rank"] = (
    ranking_data["TOPSIS_Score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

ranking_data = ranking_data.sort_values(
    "Rank"
)

ranking_data.to_csv(
    "topsis_ranking.csv",
    index=False
)

print("\n" + "=" * 70)

print("FINAL TOPSIS MODEL RANKING")

print("=" * 70)

print(
    ranking_data[
        [
            "Model",
            "Accuracy",
            "Response_Time",
            "TOPSIS_Score",
            "Rank"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 15: ACCURACY COMPARISON
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))

sns.barplot(
    data=summary,
    x="Model",
    y="Accuracy"
)

plt.title("Chatbot Semantic Accuracy Comparison")

plt.xlabel("Model")

plt.ylabel("Average Semantic Similarity")

plt.xticks(rotation=40, ha="right")

plt.tight_layout()

plt.savefig(
    "Accuracy_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 16: RESPONSE TIME COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=summary,
    x="Model",
    y="Response_Time"
)

plt.title("Chatbot Response Time Comparison")

plt.xlabel("Model")

plt.ylabel("Average Response Time (seconds)")

plt.xticks(rotation=40, ha="right")

plt.tight_layout()

plt.savefig(
    "Response_Time_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 17: TOPSIS SCORE COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=ranking_data,
    x="Model",
    y="TOPSIS_Score"
)

plt.title("TOPSIS Model Ranking")

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
# STEP 18: ACCURACY VS RESPONSE TIME
# ============================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=summary,
    x="Response_Time",
    y="Accuracy",
    hue="Model",
    s=180
)

plt.title("Accuracy vs Response Time")

plt.xlabel("Average Response Time (seconds)")

plt.ylabel("Average Semantic Similarity")

plt.tight_layout()

plt.savefig(
    "Accuracy_vs_Response_Time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 19: DISPLAY BEST TOPSIS SCORE
# ============================================================

best_model = ranking_data.iloc[0]

print("\n" + "=" * 70)

print("MODEL WITH HIGHEST TOPSIS SCORE")

print("=" * 70)

print("Model:", best_model["Model"])

print("TOPSIS Score:", round(best_model["TOPSIS_Score"], 4))

print("Rank:", best_model["Rank"])

print("Accuracy:", round(best_model["Accuracy"], 4))

print(
    "Average Response Time:",
    round(best_model["Response_Time"], 4),
    "seconds"
)

print("\nEvaluation completed successfully.")

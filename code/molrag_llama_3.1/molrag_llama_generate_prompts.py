"""
This script generates LLM prompts for receptor-specific activity prediction 
using ToxCast/Tox21 data.
"""

import os
import pickle
import pandas as pd
import numpy as np
import sys


if len(sys.argv) != 8:
    print("Usage:")
    print("python llama_generate_prompts.py "
          "TANIMOTO_PATH SMILES_PATH DESCRIPTORS_DEF_PATH "
          "BASE_SPLITS_DIR BASE_FEATURES_DIR "
          "OUT_NEIGHBORS_DIR OUT_PROMPTS_DIR")
    sys.exit(1)

TANIMOTO_PATH = sys.argv[1]
SMILES_PATH = sys.argv[2]
DESCRIPTORS_DEF_PATH = sys.argv[3]
BASE_SPLITS_DIR = sys.argv[4]
BASE_FEATURES_DIR = sys.argv[5]
OUT_NEIGHBORS_DIR = sys.argv[6]
OUT_PROMPTS_DIR = sys.argv[7]

os.makedirs(OUT_NEIGHBORS_DIR, exist_ok=True)
os.makedirs(OUT_PROMPTS_DIR, exist_ok=True)

os.makedirs(OUT_NEIGHBORS_DIR, exist_ok=True)
os.makedirs(OUT_PROMPTS_DIR, exist_ok=True)

RECEPTOR_MAP = {
    "AhR": "AhR (aryl hydrocarbon receptor)",
    "AR": "AR (androgen receptor)",
    "CAR": "CAR (constitutive androstane receptor)",
    "ERa": "ER (estrogen receptor alpha)",
    "ERb": "ER (estrogen receptor beta)",
    "ESRRA":"ESRRA (estrogen-related receptor alpha)",
    "FXR": "FXR (farnesoid X receptor)",
    "GR": "GR (glucocorticoid receptor)",
    "PPARg": "PPARγ (peroxisome proliferator-activated receptor gamma)",
    "PPARd": "PPARδ (peroxisome proliferator-activated receptor delta)",
    "PR": "PR (progesterone receptor)",
    "PXR": "PXR (pregnane X receptor)",
    "RARA":"RARA (retinoid acid receptor alpha)",
    "RORC":"RORC (RAR related orphan receptor C)",
    "RXRA": "RXR (retinoid X receptor alpha)",
    "THRAB": "THRA/B (thyroid hormone receptor alpha/beta)",    
    "VDR": "VDR (vitamin D receptor)"
}

# BUILD PROMPT COMPONENTS

def build_feature_explanation(top_features_df):
    explanation = "Top predictive molecular descriptors (selected from training data):\n\n"
    for _, row in top_features_df.iterrows():
        desc = row["Descriptor"]
        definition = row["Description"]
        explanation += f"- {desc}: {definition}\n"
    explanation += "\nThese abbreviations will be used below when listing molecular properties.\n\n"
    return explanation


def get_selected_descriptors(smiles, descriptor_df, feature_list):
    row = descriptor_df.loc[descriptor_df["SMILES"] == smiles]
    if row.empty:
        return "Descriptor values unavailable."
    
    values = []
    for feat in feature_list:
        if feat in row.columns:
            val = row.iloc[0][feat]
            values.append(f"{feat}={val:.3f}")
    return ", ".join(values)

# Build the prompt for a single query molecule
def build_prompt_dynamic(
    query_smiles,
    query_id,
    neighbors_df,
    train_descriptor_df,
    val_descriptor_df,
    top_features_df,
    base_rate,
    task_str,
    dataset_str,
    max_neighbors=5
):
    feature_list = top_features_df["Descriptor"].tolist()
    feature_explanation = build_feature_explanation(top_features_df)

    subset = (
        neighbors_df[neighbors_df["test_chem"] == query_id]
        .sort_values("ECFP4", ascending=False)
        .head(max_neighbors)
        .reset_index(drop=True)
    )

    context_lines = []
    for idx, row in enumerate(subset.itertuples(index=False), start=1):
        neighbor_id = row.neighbor
        sim = row.ECFP4
        lbl = row.label
        label_word = "ACTIVE" if lbl == 1 else "INACTIVE"

        neighbor_row = train_descriptor_df.loc[
            train_descriptor_df["dsstox_substance_id"] == neighbor_id
        ]
        if neighbor_row.empty:
            continue

        neighbor_smiles = neighbor_row["SMILES"].iloc[0]
        neighbor_desc = get_selected_descriptors(
            neighbor_smiles, train_descriptor_df, feature_list
        )

        context_lines.append(
            f"{idx}. [{label_word}] similarity={sim:.3f}\n"
            f"   SMILES: {neighbor_smiles}\n"
            f"   Selected descriptors: {neighbor_desc}"
        )

    context = "\n\n".join(context_lines)
    query_desc = get_selected_descriptors(
        query_smiles, val_descriptor_df, feature_list
    )

    # Construct the full prompt
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert computational toxicologist.\n\n"
        f"{task_str}\n\n"
        f"{dataset_str}\n"
        f"Only ~{base_rate:.0%} of compounds are experimentally ACTIVE.\n"
        "Most compounds are INACTIVE. Do not assume activity without strong evidence.\n\n"
        f"{feature_explanation}"
        "Structural similarity is computed using ECFP4 Tanimoto similarity.\n"
        "Higher similarity indicates stronger structural relatedness.\n\n"
        "Use similarity evidence AND descriptor trends to reason carefully.\n\n"
        "Reply with exactly one word: active or inactive.\n"
        "<|eot_id|>\n\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Reference molecules (from training data):\n\n"
        f"{context}\n\n"
        "Query molecule:\n"
        f"SMILES: {query_smiles}\n"
        f"Selected descriptors: {query_desc}\n\n"
        "Is this molecule active or inactive?\n"
        "Reply with one word — active or inactive:\n"
        "<|eot_id|>\n\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


# LOAD GLOBAL FILES 

toxcast_tanimoto = pd.read_csv(TANIMOTO_PATH)
smiles = pd.read_csv(SMILES_PATH, sep='\t', dtype=str, header=None)
smiles.columns = ['dsstox_substance_id', 'SMILES']
padel_descriptors_definitions = pd.read_excel(
    DESCRIPTORS_DEF_PATH, sheet_name='Detailed', dtype=str
)

receptors = [
    f for f in os.listdir(BASE_SPLITS_DIR)
    if os.path.isdir(os.path.join(BASE_SPLITS_DIR, f))
]
# Loop through each receptor and condition, generate prompts, and save outputs
for receptor in receptors:

    receptor_dir = os.path.join(BASE_SPLITS_DIR, receptor)
    full_receptor_name = RECEPTOR_MAP.get(receptor, receptor)

    conditions = [
        f for f in os.listdir(receptor_dir)
        if os.path.isdir(os.path.join(receptor_dir, f))
    ]

    for condition in conditions:

        print("\n=============================================")
        print(f"Processing: Receptor = {receptor} | Condition = {condition}")

        train_csv_path = os.path.join(
            receptor_dir, condition, "Descriptors", "descriptor_train.csv"
        )
        val_csv_path = os.path.join(
            receptor_dir, condition, "Descriptors", "descriptor_val.csv"
        )

        features_txt_path = os.path.join(
            BASE_FEATURES_DIR,
            receptor,
            f"{receptor}_{condition}",
            f"{receptor}_{condition}_desc-only.selected_features.txt"
        )

        if not (
            os.path.exists(train_csv_path)
            and os.path.exists(val_csv_path)
            and os.path.exists(features_txt_path)
        ):
            print(f"Missing required files for {receptor}_{condition}. Skipping...")
            continue

        if condition.lower() == 'combined':
            task_str = f"Task: Predict {full_receptor_name} agonist or antagonist activity."
            dataset_str = (
                "Dataset context:\n"
                f"The data originate from Both Tox21 agonist and antagonist assays for {full_receptor_name}, "
                "where the chemical is active in at least one of these assays."
            )
        else:
            task_str = f"Task: Predict {full_receptor_name} {condition.lower()} activity."
            dataset_str = (
                "Dataset context:\n"
                f"The data originate from Tox21 {full_receptor_name} {condition.lower()} assays "
                "from the ToxCast invitrodb dataset."
            )

        # LOAD DATA

        train_df = pd.read_csv(train_csv_path).rename(
            columns={"chemical_id": "dsstox_substance_id"}
        )
        val_df = pd.read_csv(val_csv_path).rename(
            columns={"chemical_id": "dsstox_substance_id"}
        )

        train_df = train_df.merge(smiles, on="dsstox_substance_id", how="left")
        val_df = val_df.merge(smiles, on="dsstox_substance_id", how="left")

        train_df_trim = train_df[['dsstox_substance_id', 'label', 'SMILES']].copy()
        val_df_trim = val_df[['dsstox_substance_id', 'label', 'SMILES']].copy()

        # LOAD FEATURES 

        with open(features_txt_path, 'r') as f:
            feature_list = [line.strip() for line in f if line.strip()]

        top_5 = pd.DataFrame({'Descriptor': feature_list[:5]})
        top_5 = top_5.merge(
            padel_descriptors_definitions[['Descriptor', 'Description']],
            on='Descriptor',
            how='left'
        )

        # COMPUTE NEIGHBORS 

        print("Computing Tanimoto neighbors...")

        val_ids = set(val_df_trim["dsstox_substance_id"])

        filtered_edges = toxcast_tanimoto[
            toxcast_tanimoto["id1"].isin(val_ids)
            | toxcast_tanimoto["id2"].isin(val_ids)
        ].copy()

        edges_id1 = filtered_edges[
            filtered_edges["id1"].isin(val_ids)
        ][["id1", "id2", "ECFP4"]]
        edges_id1.columns = ["test_chem", "neighbor", "ECFP4"]

        edges_id2 = filtered_edges[
            filtered_edges["id2"].isin(val_ids)
        ][["id2", "id1", "ECFP4"]]
        edges_id2.columns = ["test_chem", "neighbor", "ECFP4"]

        long_df = pd.concat([edges_id1, edges_id2], ignore_index=True)
        long_df = long_df.merge(
            train_df_trim[["dsstox_substance_id", "label"]],
            left_on="neighbor",
            right_on="dsstox_substance_id",
            how="inner"
        )
        long_df = long_df.drop(columns=["dsstox_substance_id"])
        long_df = long_df.sort_values(["test_chem", "ECFP4"], ascending=[True, False])

        top5_overall = long_df.groupby("test_chem", group_keys=False).head(5)[
            ["test_chem", "neighbor", "label", "ECFP4"]
        ]

        out_overall = os.path.join(
            OUT_NEIGHBORS_DIR,
            f"{receptor}_{condition}_top5_overall_all_val_chemicals.csv"
        )
        top5_overall.to_csv(out_overall, index=False)

        # BUILD PROMPTS

        print("Constructing Prompts...")

        BASE_RATE = train_df["label"].mean()
        all_prompts = {}

        for _, row in val_df.iterrows():

            query_id = row["dsstox_substance_id"]
            query_smiles = row["SMILES"]

            prompt = build_prompt_dynamic(
                query_smiles=query_smiles,
                query_id=query_id,
                neighbors_df=top5_overall,
                train_descriptor_df=train_df,
                val_descriptor_df=val_df,
                top_features_df=top_5,
                base_rate=BASE_RATE,
                task_str=task_str,
                dataset_str=dataset_str
            )

            all_prompts[query_id] = prompt

        out_prompt_pkl = os.path.join(
            OUT_PROMPTS_DIR,
            f"{receptor}_{condition}_val_prompts.pkl"
        )

        with open(out_prompt_pkl, "wb") as f:
            pickle.dump(all_prompts, f)

        print(f"Saved {len(all_prompts)} prompts to {out_prompt_pkl}")

print("\nValidation prompt generation completed successfully!")
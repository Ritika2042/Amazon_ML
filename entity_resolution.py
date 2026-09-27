#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

train_s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

train_s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

train_s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("All 4 training files loaded successfully!")


# In[ ]:


print("Source 1 shape:", train_s1.shape)
print("Source 2 shape:", train_s2.shape)
print("Source 3 shape:", train_s3.shape)
print("Ground Truth shape:", ground_truth.shape)


# In[ ]:


print("S1 columns:")
print(train_s1.columns.tolist())

print("\nS2 columns:")
print(train_s2.columns.tolist())

print("\nS3 columns:")
print(train_s3.columns.tolist())

print("\nGround Truth columns:")
print(ground_truth.columns.tolist())


# In[ ]:


print("===== SOURCE 1 =====")
display(train_s1.head(5))

print("===== SOURCE 2 =====")
display(train_s2.head(5))

print("===== SOURCE 3 =====")
display(train_s3.head(5))

print("===== GROUND TRUTH =====")
display(ground_truth.head(10))


# In[ ]:


gt = ground_truth.copy()

# Clean possible missing values
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("").astype(str)

# Count matches
gt["match_count"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if x.strip() == "" else x.count(",") + 1
)

print("Total S1 entities:", len(gt))
print("Total singleton S1 entities:", (gt["match_count"] == 0).sum())
print("S1 entities with at least one match:", (gt["match_count"] > 0).sum())

print("\nDistribution of number of matches:")
print(gt["match_count"].value_counts().sort_index())


# In[ ]:


gt["s2_match_count"] = gt["matched_entity_ids"].str.count("S2-")
gt["s3_match_count"] = gt["matched_entity_ids"].str.count("S3-")

print("Total S2 matches:", gt["s2_match_count"].sum())
print("Total S3 matches:", gt["s3_match_count"].sum())

print("\nS1 entities with S2 matches:")
print((gt["s2_match_count"] > 0).sum())

print("\nS1 entities with S3 matches:")
print((gt["s3_match_count"] > 0).sum())


# In[ ]:


print(
    gt["match_count"]
    .describe()
)


# In[ ]:


print("\nTop match counts:")
print(
    gt["match_count"]
    .value_counts()
    .sort_index()
    .tail(20)
)


# In[ ]:


sample_s1_ids = ground_truth["source1_entity_id"].head(5).tolist()

print(sample_s1_ids)


# In[ ]:


sample_s1 = train_s1[
    train_s1["entity_id"].isin(sample_s1_ids)
]

display(sample_s1)


# In[ ]:


sample_gt = ground_truth[
    ground_truth["source1_entity_id"].isin(sample_s1_ids)
].copy()

sample_gt


# In[ ]:


sample_pairs = (
    sample_gt[
        ["source1_entity_id", "matched_entity_ids"]
    ]
    .assign(
        matched_entity_ids=lambda x: x["matched_entity_ids"].fillna("")
    )
)

sample_pairs["matched_entity_ids"] = sample_pairs[
    "matched_entity_ids"
].str.split(",")

sample_pairs = sample_pairs.explode("matched_entity_ids")

sample_pairs = sample_pairs[
    sample_pairs["matched_entity_ids"].str.strip() != ""
]

display(sample_pairs)


# In[ ]:


sample_s2_ids = sample_pairs[
    sample_pairs["matched_entity_ids"].str.startswith("S2-")
]["matched_entity_ids"].tolist()

sample_s3_ids = sample_pairs[
    sample_pairs["matched_entity_ids"].str.startswith("S3-")
]["matched_entity_ids"].tolist()


# In[ ]:


display(
    train_s2[train_s2["entity_id"].isin(sample_s2_ids)]
)


# In[ ]:


display(
    train_s3[train_s3["entity_id"].isin(sample_s3_ids)]
)


# In[ ]:


# --------------------------------------------------
# Inspect 5 S1 entities and ALL their true matches
# --------------------------------------------------

sample_s1_ids = [
    'S1-965667',
    'S1-55344266',
    'S1-343815751',
    'S1-656753428',
    'S1-102811957'
]

for s1_id in sample_s1_ids:

    print("\n" + "=" * 100)
    print("SOURCE 1:", s1_id)
    print("=" * 100)

    # S1 record
    s1_record = train_s1[
        train_s1["entity_id"] == s1_id
    ]

    display(s1_record)

    # Ground truth
    gt_row = ground_truth[
        ground_truth["source1_entity_id"] == s1_id
    ]

    matched_ids_string = gt_row.iloc[0]["matched_entity_ids"]

    if pd.isna(matched_ids_string) or str(matched_ids_string).strip() == "":
        print("No matches (singleton)")
        continue

    matched_ids = [
        x.strip()
        for x in str(matched_ids_string).split(",")
        if x.strip()
    ]

    # Separate S2 and S3
    s2_ids = [x for x in matched_ids if x.startswith("S2-")]
    s3_ids = [x for x in matched_ids if x.startswith("S3-")]

    print("\nMATCHED S2 RECORDS:")
    if s2_ids:
        display(
            train_s2[
                train_s2["entity_id"].isin(s2_ids)
            ]
        )
    else:
        print("None")

    print("\nMATCHED S3 RECORDS:")
    if s3_ids:
        display(
            train_s3[
                train_s3["entity_id"].isin(s3_ids)
            ]
        )
    else:
        print("None")


# In[ ]:


print("S1 duplicate business names:",
      train_s1["business_name"].duplicated().sum())

print("S2 duplicate business names:",
      train_s2["business_name"].duplicated().sum())

print("S3 duplicate business names:",
      train_s3["business_name"].duplicated().sum())


# In[ ]:


import re
import unicodedata

def normalize_name(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Remove accents:
    # É → E
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    # Replace '&' with 'and'
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# In[ ]:


examples = [
    "Payne Enterprises LLC",
    "Payne Etrepndiels",
    "Payne Énterprises"
]

for x in examples:
    print(x, " --> ", normalize_name(x))


# In[ ]:


def normalize_address(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    # Common address abbreviations
    replacements = {
        r"\bstreet\b": "st",
        r"\broad\b": "rd",
        r"\bavenue\b": "ave",
        r"\broadway\b": "rd",
        r"\bdrive\b": "dr",
        r"\blane\b": "ln",
        r"\bboulevard\b": "blvd",
        r"\bplace\b": "pl",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# In[ ]:


address_examples = [
    "3315 Fremont St, Peoria, Illinois",
    "3315 Fremont Street, Peoria, Illinois",
    "Fremont St, Peoria, Illinois"
]

for x in address_examples:
    print(x, " --> ", normalize_address(x))


# In[ ]:


# ============================================
# Measure how normalization behaves on
# REAL TRUE MATCHES
# ============================================

# Take a manageable random sample of S1 entities
sample_gt = ground_truth.sample(
    n=10000,
    random_state=42
).copy()

# Split the comma-separated match list
sample_pairs = (
    sample_gt[
        ["source1_entity_id", "matched_entity_ids"]
    ]
    .assign(
        matched_entity_ids=lambda df:
        df["matched_entity_ids"]
        .fillna("")
        .astype(str)
        .str.split(",")
    )
    .explode("matched_entity_ids")
)

# Remove empty match lists (singletons)
sample_pairs = sample_pairs[
    sample_pairs["matched_entity_ids"].str.strip() != ""
].copy()

sample_pairs["matched_entity_ids"] = (
    sample_pairs["matched_entity_ids"].str.strip()
)

print("Sample S1 entities:", len(sample_gt))
print("True matched pairs in sample:", len(sample_pairs))


# In[ ]:


# IDs we actually need
sample_s1_ids = sample_pairs["source1_entity_id"].unique()

sample_s2_ids = sample_pairs.loc[
    sample_pairs["matched_entity_ids"].str.startswith("S2-"),
    "matched_entity_ids"
].unique()

sample_s3_ids = sample_pairs.loc[
    sample_pairs["matched_entity_ids"].str.startswith("S3-"),
    "matched_entity_ids"
].unique()

# Pull only those records from the huge datasets
sample_s1 = train_s1[
    train_s1["entity_id"].isin(sample_s1_ids)
].copy()

sample_s2 = train_s2[
    train_s2["entity_id"].isin(sample_s2_ids)
].copy()

sample_s3 = train_s3[
    train_s3["entity_id"].isin(sample_s3_ids)
].copy()

print("S1 records loaded:", len(sample_s1))
print("S2 matched records loaded:", len(sample_s2))
print("S3 matched records loaded:", len(sample_s3))


# In[ ]:


pairs["name_exact"] = (
    (pairs["name_norm_s1"] != "") &
    (pairs["name_norm_candidate"] != "") &
    (pairs["name_norm_s1"] == pairs["name_norm_candidate"])
)

pairs["address_exact"] = (
    (pairs["address_norm_s1"] != "") &
    (pairs["address_norm_candidate"] != "") &
    (pairs["address_norm_s1"] == pairs["address_norm_candidate"])
)

pairs["both_exact"] = (
    pairs["name_exact"] &
    pairs["address_exact"]
)

pairs["country_exact"] = (
    pairs["country_s1"] == pairs["country_candidate"]
)

print("===================================")
print("TRUE MATCH ANALYSIS")
print("===================================")

print(
    f"Exact normalized NAME : "
    f"{pairs['name_exact'].mean():.2%}"
)

print(
    f"Exact normalized ADDRESS: "
    f"{pairs['address_exact'].mean():.2%}"
)

print(
    f"Both exact             : "
    f"{pairs['both_exact'].mean():.2%}"
)

print(
    f"Same COUNTRY           : "
    f"{pairs['country_exact'].mean():.2%}"
)


# In[ ]:


# ============================================
# REBUILD SAMPLE PAIRS
# ============================================

# 1. Take 10,000 S1 entities from ground truth
sample_gt = ground_truth.sample(
    n=10000,
    random_state=42
).copy()

# 2. Split matched IDs into individual rows
sample_pairs = (
    sample_gt[
        ["source1_entity_id", "matched_entity_ids"]
    ]
    .assign(
        matched_entity_ids=lambda df:
        df["matched_entity_ids"]
        .fillna("")
        .astype(str)
        .str.split(",")
    )
    .explode("matched_entity_ids")
)

# Remove singleton rows
sample_pairs = sample_pairs[
    sample_pairs["matched_entity_ids"].str.strip() != ""
].copy()

sample_pairs["matched_entity_ids"] = (
    sample_pairs["matched_entity_ids"].str.strip()
)

print("Sample S1 entities:", len(sample_gt))
print("True matched pairs:", len(sample_pairs))


# 3. Get the IDs we actually need
sample_s1_ids = sample_pairs["source1_entity_id"].unique()

sample_s2_ids = sample_pairs.loc[
    sample_pairs["matched_entity_ids"].str.startswith("S2-"),
    "matched_entity_ids"
].unique()

sample_s3_ids = sample_pairs.loc[
    sample_pairs["matched_entity_ids"].str.startswith("S3-"),
    "matched_entity_ids"
].unique()


# 4. Retrieve only those records
sample_s1 = train_s1[
    train_s1["entity_id"].isin(sample_s1_ids)
].copy()

sample_s2 = train_s2[
    train_s2["entity_id"].isin(sample_s2_ids)
].copy()

sample_s3 = train_s3[
    train_s3["entity_id"].isin(sample_s3_ids)
].copy()


# 5. Normalize names and addresses
sample_s1["name_norm"] = sample_s1["business_name"].apply(normalize_name)
sample_s1["address_norm"] = sample_s1["business_address"].apply(normalize_address)

sample_s2["name_norm"] = sample_s2["business_name"].apply(normalize_name)
sample_s2["address_norm"] = sample_s2["business_address"].apply(normalize_address)

sample_s3["name_norm"] = sample_s3["business_name"].apply(normalize_name)
sample_s3["address_norm"] = sample_s3["business_address"].apply(normalize_address)


# 6. Combine S2 and S3
sample_candidates = pd.concat(
    [
        sample_s2.assign(source="S2"),
        sample_s3.assign(source="S3")
    ],
    ignore_index=True
)


# 7. Build the actual positive pair table
pairs = sample_pairs.merge(
    sample_s1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm"
        ]
    ],
    left_on="source1_entity_id",
    right_on="entity_id",
    how="left"
)

pairs = pairs.merge(
    sample_candidates[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm",
            "source"
        ]
    ],
    left_on="matched_entity_ids",
    right_on="entity_id",
    how="left",
    suffixes=("_s1", "_candidate")
)

print("Positive pairs ready:", len(pairs))


# In[ ]:


pairs["name_exact"] = (
    (pairs["name_norm_s1"] != "") &
    (pairs["name_norm_candidate"] != "") &
    (pairs["name_norm_s1"] == pairs["name_norm_candidate"])
)

pairs["address_exact"] = (
    (pairs["address_norm_s1"] != "") &
    (pairs["address_norm_candidate"] != "") &
    (pairs["address_norm_s1"] == pairs["address_norm_candidate"])
)

pairs["both_exact"] = (
    pairs["name_exact"] &
    pairs["address_exact"]
)

pairs["country_exact"] = (
    pairs["country_s1"] == pairs["country_candidate"]
)

print("===================================")
print("TRUE MATCH ANALYSIS")
print("===================================")

print(f"Exact normalized NAME : {pairs['name_exact'].mean():.2%}")
print(f"Exact normalized ADDRESS: {pairs['address_exact'].mean():.2%}")
print(f"Both exact             : {pairs['both_exact'].mean():.2%}")
print(f"Same COUNTRY           : {pairs['country_exact'].mean():.2%}")


# In[ ]:


get_ipython().system('pip install rapidfuzz -q')


# In[ ]:


from rapidfuzz import fuzz

print("RapidFuzz loaded successfully!")


# In[ ]:


pairs["name_ratio"] = pairs.apply(
    lambda row: fuzz.ratio(
        row["name_norm_s1"],
        row["name_norm_candidate"]
    ) / 100,
    axis=1
)

pairs["name_token_ratio"] = pairs.apply(
    lambda row: fuzz.token_set_ratio(
        row["name_norm_s1"],
        row["name_norm_candidate"]
    ) / 100,
    axis=1
)

pairs["address_ratio"] = pairs.apply(
    lambda row: fuzz.ratio(
        row["address_norm_s1"],
        row["address_norm_candidate"]
    ) / 100,
    axis=1
)

pairs["address_token_ratio"] = pairs.apply(
    lambda row: fuzz.token_set_ratio(
        row["address_norm_s1"],
        row["address_norm_candidate"]
    ) / 100,
    axis=1
)

print("Fuzzy similarity features created!")


# In[ ]:


print(pairs["name_ratio"].describe())
print(pairs["name_token_ratio"].describe())
print(pairs["address_ratio"].describe())
print(pairs["address_token_ratio"].describe())


# In[ ]:


# ============================================
# HARDEST TRUE MATCHES
# ============================================

print("===== LOWEST NAME SIMILARITY =====")

display(
    pairs[
        [
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_token_ratio",
            "business_address_s1",
            "business_address_candidate",
            "address_ratio",
            "address_token_ratio"
        ]
    ]
    .sort_values("name_ratio")
    .head(15)
)


# In[ ]:


print("===== LOWEST ADDRESS SIMILARITY =====")

display(
    pairs[
        [
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_token_ratio",
            "business_address_s1",
            "business_address_candidate",
            "address_ratio",
            "address_token_ratio"
        ]
    ]
    .sort_values("address_ratio")
    .head(15)
)


# In[ ]:


print("===== TOKEN SIMILARITY HELPING =====")

display(
    pairs[
        [
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_token_ratio"
        ]
    ]
    .query("name_ratio < 0.75 and name_token_ratio > 0.90")
    .head(15)
)


# In[ ]:


get_ipython().system('pip install Unidecode -q')


# In[ ]:


from unidecode import unidecode

def transliterate_name(text):
    if pd.isna(text):
        return ""

    text = unidecode(str(text))
    return normalize_name(text)

print(transliterate_name("शिवम सॉल्यूशंस"))
print(transliterate_name("অল ইনফ্রাস্ট্রাকচার প্রাইভেট লিমিটেড"))
print(transliterate_name("ईस्ट लॉजिस्टिक्स प्रा. लि."))


# In[ ]:


pairs["name_translit_s1"] = pairs["business_name_s1"].apply(
    transliterate_name
)

pairs["name_translit_candidate"] = pairs["business_name_candidate"].apply(
    transliterate_name
)

pairs["name_translit_ratio"] = pairs.apply(
    lambda row: fuzz.ratio(
        row["name_translit_s1"],
        row["name_translit_candidate"]
    ) / 100,
    axis=1
)

pairs["name_translit_token_ratio"] = pairs.apply(
    lambda row: fuzz.token_set_ratio(
        row["name_translit_s1"],
        row["name_translit_candidate"]
    ) / 100,
    axis=1
)

print("Transliteration features created.")


# In[ ]:


print("===== TRANSLITERATED NAME RATIO =====")
print(pairs["name_translit_ratio"].describe())

print("\n===== TRANSLITERATED TOKEN RATIO =====")
print(pairs["name_translit_token_ratio"].describe())


# In[ ]:


display(
    pairs[
        [
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_token_ratio",
            "name_translit_ratio",
            "name_translit_token_ratio"
        ]
    ]
    .sort_values("name_ratio")
    .head(20)
)


# In[ ]:


display(
    pairs[
        [
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_token_ratio",
            "name_translit_ratio",
            "name_translit_token_ratio"
        ]
    ]
    .sort_values("name_ratio")
    .head(20)
)


# In[ ]:


# ============================================
# Create fast ground-truth lookup
# ============================================

ground_truth_lookup = {}

for _, row in sample_gt.iterrows():
    s1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]) or str(row["matched_entity_ids"]).strip() == "":
        ground_truth_lookup[s1_id] = set()
    else:
        ground_truth_lookup[s1_id] = set(
            x.strip()
            for x in str(row["matched_entity_ids"]).split(",")
            if x.strip()
        )

print("Ground truth lookup created.")
print("Number of S1 entities:", len(ground_truth_lookup))


# In[ ]:


print("S2 duplicate names in sample:",
      sample_s2["business_name"].duplicated().sum())

print("S3 duplicate names in sample:",
      sample_s3["business_name"].duplicated().sum())


# In[ ]:


name_counts = pd.concat(
    [
        sample_s2["business_name"],
        sample_s3["business_name"]
    ]
).value_counts()

print("Names occurring more than once:")
display(name_counts[name_counts > 1].head(20))


# In[ ]:


# ============================================
# HARD NEGATIVES: SAME NAME + SAME COUNTRY
# S1 vs S2
# ============================================

# Only the columns we need from S1
s1_keys = sample_s1[
    ["entity_id", "business_name", "country"]
].rename(
    columns={"entity_id": "source1_entity_id"}
)

# Only the columns we need from S2
s2_keys = train_s2[
    ["entity_id", "business_name", "country"]
].rename(
    columns={"entity_id": "candidate_entity_id"}
)

# Remove missing names
s1_keys = s1_keys[
    s1_keys["business_name"].notna()
    & s1_keys["country"].notna()
]

s2_keys = s2_keys[
    s2_keys["business_name"].notna()
    & s2_keys["country"].notna()
]

# Find records with EXACT same name AND country
hard_neg_s2 = s1_keys.merge(
    s2_keys,
    on=["business_name", "country"],
    how="inner"
)

print("Same-name/same-country S2 candidates:",
      len(hard_neg_s2))


# In[ ]:


def is_true_match(row):
    true_ids = ground_truth_lookup.get(
        row["source1_entity_id"],
        set()
    )
    return row["candidate_entity_id"] in true_ids


hard_neg_s2["is_true"] = hard_neg_s2.apply(
    is_true_match,
    axis=1
)

hard_neg_s2 = hard_neg_s2[
    ~hard_neg_s2["is_true"]
].copy()

print(
    "Hard negative S2 pairs:",
    len(hard_neg_s2)
)


# In[ ]:


# ============================================
# HARD NEGATIVES: SAME NAME + SAME COUNTRY
# S1 vs S3
# ============================================

s3_keys = train_s3[
    ["entity_id", "business_name", "country"]
].rename(
    columns={"entity_id": "candidate_entity_id"}
)

s3_keys = s3_keys[
    s3_keys["business_name"].notna()
    & s3_keys["country"].notna()
]

hard_neg_s3 = s1_keys.merge(
    s3_keys,
    on=["business_name", "country"],
    how="inner"
)

print("Same-name/same-country S3 candidates:",
      len(hard_neg_s3))


# In[ ]:


hard_neg_s3["is_true"] = hard_neg_s3.apply(
    is_true_match,
    axis=1
)

hard_neg_s3 = hard_neg_s3[
    ~hard_neg_s3["is_true"]
].copy()

print(
    "Hard negative S3 pairs:",
    len(hard_neg_s3)
)


# In[ ]:


hard_neg_s2 = (
    hard_neg_s2
    .groupby("source1_entity_id", sort=False)
    .head(3)
    .reset_index(drop=True)
)

hard_neg_s3 = (
    hard_neg_s3
    .groupby("source1_entity_id", sort=False)
    .head(3)
    .reset_index(drop=True)
)

print("Final hard negatives S2:", len(hard_neg_s2))
print("Final hard negatives S3:", len(hard_neg_s3))


# In[ ]:


display(
    hard_neg_s2.head(20)
)


# In[ ]:


display(
    hard_neg_s3.head(20)
)


# In[ ]:


# ============================================
# BUILD NEGATIVE PAIR TABLE
# ============================================

# Combine S2 and S3 negatives
negative_pairs_raw = pd.concat(
    [
        hard_neg_s2[
            ["source1_entity_id", "candidate_entity_id"]
        ].assign(source="S2"),

        hard_neg_s3[
            ["source1_entity_id", "candidate_entity_id"]
        ].assign(source="S3")
    ],
    ignore_index=True
)

print("Negative pairs:", len(negative_pairs_raw))


# In[ ]:


# ============================================
# GET RECORDS FOR NEGATIVE PAIRS
# ============================================

neg_s1_ids = negative_pairs_raw[
    "source1_entity_id"
].unique()

neg_s2_ids = negative_pairs_raw.loc[
    negative_pairs_raw["source"] == "S2",
    "candidate_entity_id"
].unique()

neg_s3_ids = negative_pairs_raw.loc[
    negative_pairs_raw["source"] == "S3",
    "candidate_entity_id"
].unique()

neg_s1 = train_s1[
    train_s1["entity_id"].isin(neg_s1_ids)
].copy()

neg_s2 = train_s2[
    train_s2["entity_id"].isin(neg_s2_ids)
].copy()

neg_s3 = train_s3[
    train_s3["entity_id"].isin(neg_s3_ids)
].copy()

print("Negative S1 records:", len(neg_s1))
print("Negative S2 records:", len(neg_s2))
print("Negative S3 records:", len(neg_s3))


# In[ ]:


neg_s1["name_norm"] = neg_s1["business_name"].apply(normalize_name)
neg_s1["address_norm"] = neg_s1["business_address"].apply(normalize_address)

neg_s2["name_norm"] = neg_s2["business_name"].apply(normalize_name)
neg_s2["address_norm"] = neg_s2["business_address"].apply(normalize_address)

neg_s3["name_norm"] = neg_s3["business_name"].apply(normalize_name)
neg_s3["address_norm"] = neg_s3["business_address"].apply(normalize_address)


# In[ ]:


neg_s1["name_translit"] = neg_s1["business_name"].apply(transliterate_name)

neg_s2["name_translit"] = neg_s2["business_name"].apply(transliterate_name)

neg_s3["name_translit"] = neg_s3["business_name"].apply(transliterate_name)


# In[ ]:


neg_candidates = pd.concat(
    [
        neg_s2.assign(source="S2"),
        neg_s3.assign(source="S3")
    ],
    ignore_index=True
)


# In[ ]:


negative_pairs = negative_pairs_raw.merge(
    neg_s1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm",
            "name_translit"
        ]
    ],
    left_on="source1_entity_id",
    right_on="entity_id",
    how="left"
)

negative_pairs = negative_pairs.merge(
    neg_candidates[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm",
            "name_translit",
            "source"
        ]
    ],
    left_on="candidate_entity_id",
    right_on="entity_id",
    how="left",
    suffixes=("_s1", "_candidate")
)

negative_pairs["label"] = 0

print("Negative pair table:", negative_pairs.shape)


# In[ ]:


# ============================================
# NEGATIVE PAIR FEATURES
# ============================================

negative_pairs["name_ratio"] = negative_pairs.apply(
    lambda row: fuzz.ratio(
        row["name_norm_s1"],
        row["name_norm_candidate"]
    ) / 100,
    axis=1
)

negative_pairs["name_token_ratio"] = negative_pairs.apply(
    lambda row: fuzz.token_set_ratio(
        row["name_norm_s1"],
        row["name_norm_candidate"]
    ) / 100,
    axis=1
)

negative_pairs["address_ratio"] = negative_pairs.apply(
    lambda row: fuzz.ratio(
        row["address_norm_s1"],
        row["address_norm_candidate"]
    ) / 100,
    axis=1
)

negative_pairs["address_token_ratio"] = negative_pairs.apply(
    lambda row: fuzz.token_set_ratio(
        row["address_norm_s1"],
        row["address_norm_candidate"]
    ) / 100,
    axis=1
)

negative_pairs["name_translit_ratio"] = negative_pairs.apply(
    lambda row: fuzz.ratio(
        row["name_translit_s1"],
        row["name_translit_candidate"]
    ) / 100,
    axis=1
)

negative_pairs["name_translit_token_ratio"] = negative_pairs.apply(
    lambda row: fuzz.token_set_ratio(
        row["name_translit_s1"],
        row["name_translit_candidate"]
    ) / 100,
    axis=1
)

negative_pairs["country_exact"] = (
    negative_pairs["country_s1"] ==
    negative_pairs["country_candidate"]
)

print("Negative features created!")


# In[ ]:


# ============================================
# COMPARE POSITIVE VS NEGATIVE
# ============================================

feature_columns = [
    "name_ratio",
    "name_token_ratio",
    "name_translit_ratio",
    "name_translit_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "country_exact"
]

print("===== POSITIVE PAIRS =====")
display(
    pairs[feature_columns].describe().T
)

print("\n===== NEGATIVE PAIRS =====")
display(
    negative_pairs[feature_columns].describe().T
)


# In[ ]:


comparison = pd.DataFrame({
    "Positive": pairs[
        [
            "name_ratio",
            "name_token_ratio",
            "name_translit_ratio",
            "name_translit_token_ratio",
            "address_ratio",
            "address_token_ratio"
        ]
    ].mean(),

    "Negative": negative_pairs[
        [
            "name_ratio",
            "name_token_ratio",
            "name_translit_ratio",
            "name_translit_token_ratio",
            "address_ratio",
            "address_token_ratio"
        ]
    ].mean()
})

display(comparison)


# In[ ]:


import re

def extract_numbers(text):
    if pd.isna(text):
        return set()

    return set(
        re.findall(r"\d+", str(text))
    )

def number_overlap(addr1, addr2):
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)

    if not nums1 or not nums2:
        return 0.0

    return len(nums1 & nums2) / len(nums1 | nums2)


# In[ ]:


pairs["number_overlap"] = pairs.apply(
    lambda row: number_overlap(
        row["business_address_s1"],
        row["business_address_candidate"]
    ),
    axis=1
)


# In[ ]:


pairs["number_overlap"] = pairs.apply(
    lambda row: number_overlap(
        row["business_address_s1"],
        row["business_address_candidate"]
    ),
    axis=1
)


# In[ ]:


negative_pairs["number_overlap"] = negative_pairs.apply(
    lambda row: number_overlap(
        row["business_address_s1"],
        row["business_address_candidate"]
    ),
    axis=1
)


# In[ ]:


print("TRUE MATCHES")
print(pairs["number_overlap"].describe())

print("\nFALSE MATCHES")
print(negative_pairs["number_overlap"].describe())


# In[ ]:


# ============================================
# NEGATIVE TYPE 2
# SAME RAW ADDRESS + SAME COUNTRY
# DIFFERENT BUSINESS
# ============================================

# Addresses belonging to our sampled S1 records
sample_s1_address_keys = sample_s1[
    ["entity_id", "business_address", "business_name", "country"]
].rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_address": "s1_address",
        "business_name": "s1_business_name",
        "country": "s1_country"
    }
)

# S2 records with non-missing addresses
s2_address_keys = train_s2[
    ["entity_id", "business_address", "business_name", "country"]
].rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_address": "candidate_address",
        "business_name": "candidate_business_name",
        "country": "candidate_country"
    }
)

# Match on EXACT RAW ADDRESS + COUNTRY
address_neg_s2 = sample_s1_address_keys.merge(
    s2_address_keys,
    left_on=["s1_address", "s1_country"],
    right_on=["candidate_address", "candidate_country"],
    how="inner"
)

print("Same-address S2 candidates:", len(address_neg_s2))


# In[ ]:


def keep_address_negative(row):
    true_ids = ground_truth_lookup.get(
        row["source1_entity_id"],
        set()
    )

    # Remove true matches
    if row["candidate_entity_id"] in true_ids:
        return False

    # Remove same-name cases for this experiment
    if (
        str(row["s1_business_name"]).strip().lower()
        ==
        str(row["candidate_business_name"]).strip().lower()
    ):
        return False

    return True


address_neg_s2["keep"] = address_neg_s2.apply(
    keep_address_negative,
    axis=1
)

address_neg_s2 = address_neg_s2[
    address_neg_s2["keep"]
].copy()

print(
    "Valid same-address/different-name S2 negatives:",
    len(address_neg_s2)
)


# In[ ]:


# ============================================
# SAME RAW ADDRESS + SAME COUNTRY
# S1 vs S3
# ============================================

s3_address_keys = train_s3[
    ["entity_id", "business_address", "business_name", "country"]
].rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_address": "candidate_address",
        "business_name": "candidate_business_name",
        "country": "candidate_country"
    }
)

address_neg_s3 = sample_s1_address_keys.merge(
    s3_address_keys,
    left_on=["s1_address", "s1_country"],
    right_on=["candidate_address", "candidate_country"],
    how="inner"
)

print("Same-address S3 candidates:", len(address_neg_s3))


# In[ ]:


address_neg_s3["keep"] = address_neg_s3.apply(
    keep_address_negative,
    axis=1
)

address_neg_s3 = address_neg_s3[
    address_neg_s3["keep"]
].copy()

print(
    "Valid same-address/different-name S3 negatives:",
    len(address_neg_s3)
)


# In[ ]:


address_neg_s2 = (
    address_neg_s2
    .groupby("source1_entity_id", sort=False)
    .head(3)
    .reset_index(drop=True)
)

address_neg_s3 = (
    address_neg_s3
    .groupby("source1_entity_id", sort=False)
    .head(3)
    .reset_index(drop=True)
)

print("Final address negatives S2:", len(address_neg_s2))
print("Final address negatives S3:", len(address_neg_s3))


# In[ ]:


print(address_neg_s2.columns.tolist())
print(address_neg_s2.head())


# In[ ]:


print(address_neg_s2.columns.tolist())


# In[ ]:


# ============================================
# REBUILD ADDRESS-BASED NEGATIVES — S2
# ============================================

# S1 records from our 10k sample
s1_addr = sample_s1[
    ["entity_id", "business_address", "business_name", "country"]
].copy()

s1_addr = s1_addr.rename(columns={
    "entity_id": "source1_entity_id",
    "business_address": "s1_address",
    "business_name": "s1_business_name",
    "country": "s1_country"
})

# S2 records
s2_addr = train_s2[
    ["entity_id", "business_address", "business_name", "country"]
].copy()

s2_addr = s2_addr.rename(columns={
    "entity_id": "candidate_entity_id",
    "business_address": "candidate_address",
    "business_name": "candidate_business_name",
    "country": "candidate_country"
})

# Remove missing values
s1_addr = s1_addr.dropna(
    subset=["s1_address", "s1_country"]
)

s2_addr = s2_addr.dropna(
    subset=["candidate_address", "candidate_country"]
)

print("S1 records available:", len(s1_addr))
print("S2 records available:", len(s2_addr))

# Exact raw address + country match
address_neg_s2 = s1_addr.merge(
    s2_addr,
    left_on=["s1_address", "s1_country"],
    right_on=["candidate_address", "candidate_country"],
    how="inner"
)

print("\nExact same-address candidates:", len(address_neg_s2))

# Remove true matches
if len(address_neg_s2) > 0:

    address_neg_s2["is_true"] = address_neg_s2.apply(
        lambda row:
        row["candidate_entity_id"]
        in ground_truth_lookup.get(
            row["source1_entity_id"],
            set()
        ),
        axis=1
    )

    # Also require different business names
    address_neg_s2 = address_neg_s2[
        (~address_neg_s2["is_true"]) &
        (
            address_neg_s2["s1_business_name"].fillna("").str.strip().str.lower()
            !=
            address_neg_s2["candidate_business_name"].fillna("").str.strip().str.lower()
        )
    ].copy()

print("Valid same-address/different-name negatives:", len(address_neg_s2))

if len(address_neg_s2) > 0:
    display(address_neg_s2.head(20))
else:
    print("\nNo exact raw-address negatives were found.")


# In[ ]:


# ============================================
# NEGATIVE TYPE 2
# SAME NORMALIZED ADDRESS + SAME COUNTRY
# DIFFERENT NORMALIZED NAME
# S1 vs FULL S2
# ============================================

# Sample S1 records
s1_lookup = sample_s1[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "address_norm"
    ]
].copy()

s1_lookup = s1_lookup.rename(
    columns={"entity_id": "source1_entity_id"}
)

# Remove rows with empty normalized addresses
s1_lookup = s1_lookup[
    (s1_lookup["address_norm"].notna()) &
    (s1_lookup["address_norm"] != "") &
    (s1_lookup["country"].notna())
].copy()

# Build a lookup set of normalized address + country
lookup_keys = set(
    zip(
        s1_lookup["address_norm"],
        s1_lookup["country"]
    )
)

print("Unique S1 address-country keys:",
      len(lookup_keys))

# Search S2 in chunks
address_neg_s2_parts = []

for chunk_no, chunk in enumerate(
    pd.read_csv(
        "dataset/train/train_source2.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=500_000
    )
):

    # Normalize address
    chunk["address_norm"] = (
        chunk["business_address"]
        .fillna("")
        .apply(normalize_address)
    )

    # Normalize name
    chunk["name_norm"] = (
        chunk["business_name"]
        .fillna("")
        .apply(normalize_name)
    )

    # Keep only rows whose normalized address-country
    # exists among our sampled S1 records
    mask = [
        (addr, country) in lookup_keys
        for addr, country in zip(
            chunk["address_norm"],
            chunk["country"]
        )
    ]

    matched_chunk = chunk.loc[mask].copy()

    if len(matched_chunk) > 0:

        # Attach matching S1 records
        matched_chunk = s1_lookup.merge(
            matched_chunk,
            left_on=["address_norm", "country"],
            right_on=["address_norm", "country"],
            how="inner",
            suffixes=("_s1", "_candidate")
        )

        # Different normalized business names
        matched_chunk = matched_chunk[
            matched_chunk["name_norm_s1"]
            !=
            matched_chunk["name_norm_candidate"]
        ]

        if len(matched_chunk) > 0:
            address_neg_s2_parts.append(
                matched_chunk
            )

    print(
        f"Processed S2 chunk {chunk_no + 1} | "
        f"candidates collected: "
        f"{sum(len(x) for x in address_neg_s2_parts)}"
    )

# Combine
if address_neg_s2_parts:

    address_neg_s2 = pd.concat(
        address_neg_s2_parts,
        ignore_index=True
    )

else:

    address_neg_s2 = pd.DataFrame()

print("\nRaw normalized-address negatives:",
      len(address_neg_s2))


# In[ ]:


if len(address_neg_s2) > 0:

    address_neg_s2["is_true"] = address_neg_s2.apply(
        lambda row:
        row["candidate_entity_id"]
        in ground_truth_lookup.get(
            row["source1_entity_id"],
            set()
        ),
        axis=1
    )

    address_neg_s2 = address_neg_s2[
        ~address_neg_s2["is_true"]
    ].copy()

    print(
        "After removing true matches:",
        len(address_neg_s2)
    )

else:
    print("No normalized-address candidates found.")


# In[ ]:


if len(address_neg_s2) > 0:

    address_neg_s2["is_true"] = address_neg_s2.apply(
        lambda row:
        row["candidate_entity_id"]
        in ground_truth_lookup.get(
            row["source1_entity_id"],
            set()
        ),
        axis=1
    )

    address_neg_s2 = address_neg_s2[
        ~address_neg_s2["is_true"]
    ].copy()

    print(
        "After removing true matches:",
        len(address_neg_s2)
    )

else:
    print("No normalized-address candidates found.")


# In[ ]:


print(address_neg_s2.columns.tolist())


# In[ ]:


# Rename the S2 entity ID
address_neg_s2 = address_neg_s2.rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_name": "candidate_business_name",
        "business_address": "candidate_address",
        "country": "candidate_country"
    }
)

print(address_neg_s2.columns.tolist())


# In[ ]:


# ============================================
# Create fast ground-truth lookup
# ============================================

ground_truth_lookup = {}

for _, row in sample_gt.iterrows():
    s1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]) or str(row["matched_entity_ids"]).strip() == "":
        ground_truth_lookup[s1_id] = set()
    else:
        ground_truth_lookup[s1_id] = set(
            x.strip()
            for x in str(row["matched_entity_ids"]).split(",")
            if x.strip()
        )

print("Ground truth lookup created.")
print("Number of S1 entities:", len(ground_truth_lookup))


# In[ ]:


print("Rows:", len(address_neg_s2))
display(address_neg_s2.head(10))


# In[ ]:


address_neg_s2["is_true"] = address_neg_s2.apply(
    lambda row:
    row["candidate_entity_id"]
    in ground_truth_lookup.get(
        row["source1_entity_id"],
        set()
    ),
    axis=1
)

address_neg_s2 = address_neg_s2[
    ~address_neg_s2["is_true"]
].copy()

print(
    "After removing true matches:",
    len(address_neg_s2)
)


# In[ ]:


address_neg_s2 = (
    address_neg_s2
    .groupby("source1_entity_id", sort=False)
    .head(3)
    .reset_index(drop=True)
)

print(
    "Final normalized-address negatives S2:",
    len(address_neg_s2)
)


# In[ ]:


address_neg_s2 = (
    address_neg_s2
    .groupby("source1_entity_id", sort=False)
    .head(3)
    .reset_index(drop=True)
)

print(
    "Final normalized-address negatives S2:",
    len(address_neg_s2)
)


# In[ ]:


display(
    address_neg_s2[
        [
            "source1_entity_id",
            "s1_business_name",
            "s1_address",
            "candidate_entity_id",
            "candidate_business_name",
            "candidate_address",
            "s1_country"
        ]
    ].head(20)
)


# In[ ]:


display(
    address_neg_s2[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_address_s1",
            "candidate_entity_id",
            "business_name_candidate",
            "business_address_candidate",
            "country"
        ]
    ].head(20)
)


# In[ ]:


display(
    address_neg_s2[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_address_s1",
            "candidate_entity_id",
            "business_name_candidate",
            "business_address_candidate",
            "candidate_country"
        ]
    ].head(20)
)


# In[ ]:


print("===== CURRENT DATAFRAME COLUMNS =====")

dataframes = {
    "train_s1": train_s1,
    "train_s2": train_s2,
    "train_s3": train_s3,
    "ground_truth": ground_truth,
    "sample_s1": sample_s1,
    "sample_s2": sample_s2,
    "sample_s3": sample_s3,
    "sample_candidates": sample_candidates,
    "pairs": pairs,
    "hard_neg_s2": hard_neg_s2,
    "hard_neg_s3": hard_neg_s3,
    "negative_pairs": negative_pairs,
    "address_neg_s2": address_neg_s2,
}

for name, df in dataframes.items():
    print(f"\n{name}:")
    print(df.columns.tolist())


# In[ ]:


# ============================================================
# FEATURES FOR NORMALIZED-ADDRESS NEGATIVES
# ============================================================

address_neg_s2["name_translit_s1"] = (
    address_neg_s2["business_name_s1"]
    .apply(transliterate_name)
)

address_neg_s2["name_translit_candidate"] = (
    address_neg_s2["business_name_candidate"]
    .apply(transliterate_name)
)

address_neg_s2["name_ratio"] = address_neg_s2.apply(
    lambda row: fuzz.ratio(
        row["name_norm_s1"],
        row["name_norm_candidate"]
    ) / 100,
    axis=1
)

address_neg_s2["name_token_ratio"] = address_neg_s2.apply(
    lambda row: fuzz.token_set_ratio(
        row["name_norm_s1"],
        row["name_norm_candidate"]
    ) / 100,
    axis=1
)

address_neg_s2["name_translit_ratio"] = address_neg_s2.apply(
    lambda row: fuzz.ratio(
        row["name_translit_s1"],
        row["name_translit_candidate"]
    ) / 100,
    axis=1
)

address_neg_s2["name_translit_token_ratio"] = address_neg_s2.apply(
    lambda row: fuzz.token_set_ratio(
        row["name_translit_s1"],
        row["name_translit_candidate"]
    ) / 100,
    axis=1
)

address_neg_s2["address_ratio"] = address_neg_s2.apply(
    lambda row: fuzz.ratio(
        row["business_address_s1"],
        row["business_address_candidate"]
    ) / 100,
    axis=1
)

address_neg_s2["address_token_ratio"] = address_neg_s2.apply(
    lambda row: fuzz.token_set_ratio(
        row["business_address_s1"],
        row["business_address_candidate"]
    ) / 100,
    axis=1
)

address_neg_s2["number_overlap"] = address_neg_s2.apply(
    lambda row: number_overlap(
        row["business_address_s1"],
        row["business_address_candidate"]
    ),
    axis=1
)

# Same country by construction
address_neg_s2["country_exact"] = True

# This is definitely a negative pair
address_neg_s2["label"] = 0

print("Features added to address_neg_s2.")
print(address_neg_s2.shape)


# In[ ]:


# ============================================================
# STANDARDIZE POSITIVE PAIRS
# ============================================================

positive_train = pairs[
    [
        "source1_entity_id",
        "entity_id_candidate",
        "source",
        "business_name_s1",
        "business_name_candidate",
        "business_address_s1",
        "business_address_candidate",
        "country_s1",
        "country_candidate",
        "name_ratio",
        "name_token_ratio",
        "name_translit_ratio",
        "name_translit_token_ratio",
        "address_ratio",
        "address_token_ratio",
        "country_exact",
        "number_overlap"
    ]
].copy()

positive_train = positive_train.rename(
    columns={
        "entity_id_candidate": "candidate_entity_id"
    }
)

positive_train["label"] = 1


# ============================================================
# STANDARDIZE SAME-NAME NEGATIVES
# ============================================================

negative_train_1 = negative_pairs[
    [
        "source1_entity_id",
        "entity_id_candidate",
        "source_candidate",
        "business_name_s1",
        "business_name_candidate",
        "business_address_s1",
        "business_address_candidate",
        "country_s1",
        "country_candidate",
        "name_ratio",
        "name_token_ratio",
        "name_translit_ratio",
        "name_translit_token_ratio",
        "address_ratio",
        "address_token_ratio",
        "country_exact",
        "number_overlap"
    ]
].copy()

negative_train_1 = negative_train_1.rename(
    columns={
        "entity_id_candidate": "candidate_entity_id",
        "source_candidate": "source"
    }
)

negative_train_1["label"] = 0


# ============================================================
# STANDARDIZE SAME-ADDRESS NEGATIVES
# ============================================================

negative_train_2 = address_neg_s2[
    [
        "source1_entity_id",
        "candidate_entity_id",
        "business_name_s1",
        "business_name_candidate",
        "business_address_s1",
        "business_address_candidate",
        "candidate_country",
        "name_ratio",
        "name_token_ratio",
        "name_translit_ratio",
        "name_translit_token_ratio",
        "address_ratio",
        "address_token_ratio",
        "country_exact",
        "number_overlap",
        "label"
    ]
].copy()

negative_train_2 = negative_train_2.rename(
    columns={
        "candidate_country": "country_candidate"
    }
)

# Because the address-negative was generated using
# same address + same country:
negative_train_2["country_s1"] = (
    negative_train_2["country_candidate"]
)

negative_train_2["source"] = "S2"

# Reorder to exactly match the other tables
negative_train_2 = negative_train_2[
    positive_train.columns
]


print("Positive:", positive_train.shape)
print("Negative type 1:", negative_train_1.shape)
print("Negative type 2:", negative_train_2.shape)


# In[ ]:


# ============================================================
# FINAL EXPERIMENTAL TRAINING TABLE
# ============================================================

training_pairs = pd.concat(
    [
        positive_train,
        negative_train_1,
        negative_train_2
    ],
    ignore_index=True
)

print("Training pairs:", training_pairs.shape)

print("\nLabel distribution:")
print(training_pairs["label"].value_counts())

print("\nMissing values:")
print(
    training_pairs.isnull().sum()
)


# In[ ]:


display(
    training_pairs[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "source",
            "name_ratio",
            "name_token_ratio",
            "name_translit_ratio",
            "address_ratio",
            "address_token_ratio",
            "number_overlap",
            "country_exact",
            "label"
        ]
    ].sample(
        20,
        random_state=42
    )
)


# In[ ]:


print("Missing candidate addresses by label:")
print(
    training_pairs
    .assign(
        candidate_address_missing=
        training_pairs["business_address_candidate"].isna()
        |
        (training_pairs["business_address_candidate"].fillna("").str.strip() == "")
    )
    .groupby("label")["candidate_address_missing"]
    .sum()
)


# In[ ]:


print("Missing S1 addresses:")
print(
    training_pairs["business_address_s1"].isna().sum()
)

print("Missing candidate addresses:")
print(
    training_pairs["business_address_candidate"].isna().sum()
)


# In[ ]:


feature_columns = [
    "name_ratio",
    "name_token_ratio",
    "name_translit_ratio",
    "name_translit_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "number_overlap",
    "country_exact"
]

print(
    training_pairs[feature_columns]
    .isnull()
    .sum()
)


# In[ ]:


training_pairs["s1_address_missing"] = (
    training_pairs["business_address_s1"]
    .fillna("")
    .str.strip()
    .eq("")
    .astype(int)
)

training_pairs["candidate_address_missing"] = (
    training_pairs["business_address_candidate"]
    .fillna("")
    .str.strip()
    .eq("")
    .astype(int)
)

print(
    training_pairs[
        [
            "s1_address_missing",
            "candidate_address_missing"
        ]
    ].sum()
)


# In[ ]:


# ============================================================
# TYPE 2 NEGATIVES — S1 vs S3
# SAME NORMALIZED ADDRESS + SAME COUNTRY
# DIFFERENT NORMALIZED BUSINESS NAME
# ============================================================

# ------------------------------------------------------------
# 1. Prepare the sampled S1 records
# ------------------------------------------------------------

s1_addr_lookup = sample_s1[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "address_norm"
    ]
].copy()

s1_addr_lookup = s1_addr_lookup.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "business_name_s1",
        "business_address": "business_address_s1",
        "name_norm": "name_norm_s1"
    }
)

# We only want usable addresses
s1_addr_lookup = s1_addr_lookup[
    (s1_addr_lookup["address_norm"].notna()) &
    (s1_addr_lookup["address_norm"] != "") &
    (s1_addr_lookup["country"].notna())
].copy()

# Fast lookup keys
lookup_keys = set(
    zip(
        s1_addr_lookup["address_norm"],
        s1_addr_lookup["country"]
    )
)

print("Unique S1 address-country keys:", len(lookup_keys))


# ------------------------------------------------------------
# 2. Scan the FULL S3 in chunks
# ------------------------------------------------------------

address_neg_s3_parts = []

for chunk_no, chunk in enumerate(
    pd.read_csv(
        "dataset/train/train_source3.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=500_000
    )
):

    # Normalize S3
    chunk["candidate_address_norm"] = (
        chunk["business_address"]
        .fillna("")
        .apply(normalize_address)
    )

    chunk["candidate_name_norm"] = (
        chunk["business_name"]
        .fillna("")
        .apply(normalize_name)
    )

    # Keep only S3 rows whose normalized
    # address + country occurs in sampled S1
    mask = [
        (addr, country) in lookup_keys
        for addr, country in zip(
            chunk["candidate_address_norm"],
            chunk["country"]
        )
    ]

    matched_chunk = chunk.loc[mask].copy()

    if len(matched_chunk) > 0:

        # Match S1 and S3 on normalized address + country
        matched_chunk = s1_addr_lookup.merge(
            matched_chunk,
            left_on=["address_norm", "country"],
            right_on=[
                "candidate_address_norm",
                "country"
            ],
            how="inner"
        )

        # Different normalized business names
        matched_chunk = matched_chunk[
            matched_chunk["name_norm_s1"]
            !=
            matched_chunk["candidate_name_norm"]
        ].copy()

        if len(matched_chunk) > 0:
            address_neg_s3_parts.append(
                matched_chunk
            )

    print(
        f"Processed S3 chunk {chunk_no + 1} | "
        f"candidates collected: "
        f"{sum(len(x) for x in address_neg_s3_parts)}"
    )


# ------------------------------------------------------------
# 3. Combine all chunks
# ------------------------------------------------------------

if address_neg_s3_parts:

    address_neg_s3 = pd.concat(
        address_neg_s3_parts,
        ignore_index=True
    )

else:

    address_neg_s3 = pd.DataFrame()


print(
    "\nRaw normalized-address S3 candidates:",
    len(address_neg_s3)
)


# In[2]:


print("address_neg_s3 exists:", "address_neg_s3" in globals())

if "address_neg_s3" in globals():
    print("Rows:", len(address_neg_s3))
    print("Columns:", address_neg_s3.columns.tolist())


# In[3]:


print("train_s1:", "train_s1" in globals())
print("train_s2:", "train_s2" in globals())
print("train_s3:", "train_s3" in globals())
print("ground_truth:", "ground_truth" in globals())
print("sample_s1:", "sample_s1" in globals())
print("pairs:", "pairs" in globals())
print("ground_truth_lookup:", "ground_truth_lookup" in globals())
print("address_neg_s2:", "address_neg_s2" in globals())


# In[4]:


import pandas as pd
import numpy as np
import re
import unicodedata

print("Core libraries loaded.")


# In[5]:


train_s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

train_s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

train_s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Training files loaded.")
print("S1:", train_s1.shape)
print("S2:", train_s2.shape)
print("S3:", train_s3.shape)
print("Ground truth:", ground_truth.shape)


# In[6]:


import os

print("Current Jupyter folder:")
print(os.getcwd())

print("\nFolders/files here:")
print(os.listdir())


# In[7]:


print(os.listdir("dataset"))
print(os.listdir("dataset/train"))


# In[8]:


TRAIN_DIR = r"E:\some_folder\student_resource\dataset\train"

train_s1 = pd.read_csv(
    TRAIN_DIR + r"\train_source1.tsv",
    sep="\t"
)

train_s2 = pd.read_csv(
    TRAIN_DIR + r"\train_source2.tsv",
    sep="\t"
)

train_s3 = pd.read_csv(
    TRAIN_DIR + r"\train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    TRAIN_DIR + r"\train_ground_truth.tsv",
    sep="\t"
)


# In[9]:


TRAIN_DIR = r"E:\AmazonML\student_resource\dataset\train"


# In[10]:


train_s1 = pd.read_csv(
    TRAIN_DIR + r"\train_source1.tsv",
    sep="\t"
)

train_s2 = pd.read_csv(
    TRAIN_DIR + r"\train_source2.tsv",
    sep="\t"
)

train_s3 = pd.read_csv(
    TRAIN_DIR + r"\train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    TRAIN_DIR + r"\train_ground_truth.tsv",
    sep="\t"
)

print("All training files loaded!")


# In[11]:


import os

TRAIN_DIR = r"E:\Hackathons_Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

print("Folder exists:", os.path.exists(TRAIN_DIR))
print("\nFiles in training folder:")
for file in os.listdir(TRAIN_DIR):
    print(file)


# In[12]:


import pandas as pd

train_s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t")
train_s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t")
train_s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t")
ground_truth = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")

print("S1:", train_s1.shape)
print("S2:", train_s2.shape)
print("S3:", train_s3.shape)
print("Ground Truth:", ground_truth.shape)


# In[13]:


print(train_s1.columns.tolist())
print(ground_truth.columns.tolist())


# In[14]:


import os
import re
import unicodedata
import pandas as pd
import numpy as np

from rapidfuzz.fuzz import ratio, token_set_ratio
from unidecode import unidecode

# -----------------------------
# Normalize business names
# -----------------------------
def normalize_name(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Handle accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # Treat & as "and"
    text = text.replace("&", " and ")

    # Keep only letters/numbers/spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Normalize addresses
# -----------------------------
def normalize_address(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    replacements = {
        r"\bstreet\b": "st",
        r"\broad\b": "rd",
        r"\bavenue\b": "ave",
        r"\broadway\b": "rd",
        r"\bdrive\b": "dr",
        r"\blane\b": "ln",
        r"\bboulevard\b": "blvd",
        r"\bplace\b": "pl",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Transliteration
# -----------------------------
def transliterate_name(text):
    if pd.isna(text):
        return ""

    return normalize_name(unidecode(str(text)))


# -----------------------------
# Extract numbers from address
# -----------------------------
def extract_numbers(text):
    if pd.isna(text):
        return set()

    return set(re.findall(r"\d+", str(text)))


def number_overlap(addr1, addr2):
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)

    if not nums1 or not nums2:
        return 0.0

    return len(nums1 & nums2) / len(nums1 | nums2)


print("Normalization functions ready.")


# In[15]:


print(normalize_name("Payne Énterprises LLC"))
print(normalize_address("3315 Fremont Street, Peoria, Illinois"))
print(transliterate_name("शिवम सोल्यूशन्स"))
print(number_overlap("14 Mountainbrook Road, Asheville", 
                      "14 Mountainbrook Rd, Asheville"))


# In[16]:


# ============================================
# STEP 4: CREATE POSITIVE TRAINING PAIRS
# ============================================

# Take a reproducible sample of S1 businesses
sample_gt = ground_truth.sample(
    n=10000,
    random_state=42
).copy()

print("Sampled S1 records:", len(sample_gt))

# ------------------------------------------------
# Convert comma-separated matched IDs into rows
# ------------------------------------------------
sample_matches = sample_gt.copy()

sample_matches["matched_entity_ids"] = (
    sample_matches["matched_entity_ids"]
    .fillna("")
    .astype(str)
)

sample_matches["matched_entity_ids"] = sample_matches[
    "matched_entity_ids"
].apply(
    lambda x: [i.strip() for i in x.split(",") if i.strip()]
)

sample_matches = sample_matches.explode(
    "matched_entity_ids"
).rename(
    columns={"matched_entity_ids": "candidate_entity_id"}
)

# Remove singleton rows with no match
sample_matches = sample_matches[
    sample_matches["candidate_entity_id"].notna()
].copy()

print("Positive pairs from sampled S1:", len(sample_matches))

# ------------------------------------------------
# Separate S2 and S3 candidates
# ------------------------------------------------
s2_ids = set(train_s2["entity_id"])
s3_ids = set(train_s3["entity_id"])

positive_s2 = sample_matches[
    sample_matches["candidate_entity_id"].isin(s2_ids)
].copy()

positive_s3 = sample_matches[
    sample_matches["candidate_entity_id"].isin(s3_ids)
].copy()

print("Positive S2 pairs:", len(positive_s2))
print("Positive S3 pairs:", len(positive_s3))

# ------------------------------------------------
# Get the actual S1 records
# ------------------------------------------------
sample_s1_ids = sample_gt["source1_entity_id"]

sample_s1 = train_s1[
    train_s1["entity_id"].isin(sample_s1_ids)
].copy()

# ------------------------------------------------
# Get only the S2/S3 records that are actually
# needed for these positive pairs
# ------------------------------------------------
sample_s2 = train_s2[
    train_s2["entity_id"].isin(
        positive_s2["candidate_entity_id"]
    )
].copy()

sample_s3 = train_s3[
    train_s3["entity_id"].isin(
        positive_s3["candidate_entity_id"]
    )
].copy()

print("Unique S1 records:", len(sample_s1))
print("Unique matched S2 records:", len(sample_s2))
print("Unique matched S3 records:", len(sample_s3))

# ------------------------------------------------
# Add normalization only to these smaller tables
# ------------------------------------------------
for df in [sample_s1, sample_s2, sample_s3]:

    df["name_norm"] = df["business_name"].apply(
        normalize_name
    )

    df["address_norm"] = df["business_address"].apply(
        normalize_address
    )

    df["name_translit"] = df["business_name"].apply(
        transliterate_name
    )

# ------------------------------------------------
# Build positive S2 training pairs
# ------------------------------------------------
positive_s2 = positive_s2.merge(
    sample_s1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm",
            "name_translit"
        ]
    ],
    left_on="source1_entity_id",
    right_on="entity_id",
    how="left"
)

positive_s2 = positive_s2.merge(
    sample_s2[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm",
            "name_translit"
        ]
    ],
    left_on="candidate_entity_id",
    right_on="entity_id",
    how="left",
    suffixes=("_s1", "_candidate")
)

positive_s2["source"] = "S2"

# ------------------------------------------------
# Build positive S3 training pairs
# ------------------------------------------------
positive_s3 = positive_s3.merge(
    sample_s1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm",
            "name_translit"
        ]
    ],
    left_on="source1_entity_id",
    right_on="entity_id",
    how="left"
)

positive_s3 = positive_s3.merge(
    sample_s3[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm",
            "name_translit"
        ]
    ],
    left_on="candidate_entity_id",
    right_on="entity_id",
    how="left",
    suffixes=("_s1", "_candidate")
)

positive_s3["source"] = "S3"

# ------------------------------------------------
# Combine S2 + S3
# ------------------------------------------------
positive_train_raw = pd.concat(
    [positive_s2, positive_s3],
    ignore_index=True
)

print("\nTotal positive training pairs:",
      len(positive_train_raw))

print("\nColumns:")
print(positive_train_raw.columns.tolist())

# ------------------------------------------------
# Save immediately so a kernel restart won't
# destroy this work
# ------------------------------------------------
positive_train_raw.to_pickle(
    "positive_train_raw.pkl"
)

print("\nSaved: positive_train_raw.pkl")


# In[17]:


print(positive_train_raw.shape)

print(
    positive_train_raw[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "business_address_s1",
            "business_address_candidate",
            "country_s1",
            "country_candidate",
            "source"
        ]
    ].head(10).to_string(index=False)
)


# In[18]:


positive_train_raw.to_pickle("positive_train_raw.pkl")


# In[19]:


import os

PROJECT_DIR = r"C:\Users\badig\amazon_ml_entity_resolution"

os.makedirs(PROJECT_DIR, exist_ok=True)

print("Project folder:", PROJECT_DIR)


# In[20]:


positive_train_raw.to_pickle(
    os.path.join(PROJECT_DIR, "positive_train_raw.pkl")
)

print("Saved to:")
print(os.path.join(PROJECT_DIR, "positive_train_raw.pkl"))


# In[21]:


# ============================================
# STEP 5: TYPE-1 HARD NEGATIVES
# Same name + same country, but NOT a true match
# ============================================

# Create a lookup of true matches for each sampled S1
true_match_map = (
    sample_gt
    .set_index("source1_entity_id")["matched_entity_ids"]
    .apply(
        lambda x: set(x) if isinstance(x, list)
        else set()
    )
    .to_dict()
)

# ------------------------------------------------
# S2: same raw business name + same country
# ------------------------------------------------

hard_neg_s2 = sample_s1[
    [
        "entity_id",
        "business_name",
        "country"
    ]
].merge(
    train_s2[
        [
            "entity_id",
            "business_name",
            "country"
        ]
    ],
    on=["business_name", "country"],
    how="inner",
    suffixes=("_s1", "_candidate")
)

# Rename S1 ID
hard_neg_s2 = hard_neg_s2.rename(
    columns={"entity_id_s1": "source1_entity_id"}
)

# Remove the actual S1 record itself if IDs ever overlap
hard_neg_s2 = hard_neg_s2[
    hard_neg_s2["source1_entity_id"] !=
    hard_neg_s2["candidate_entity_id"]
].copy()

# Remove candidates that are TRUE matches
hard_neg_s2 = hard_neg_s2[
    ~hard_neg_s2.apply(
        lambda row:
        row["candidate_entity_id"]
        in true_match_map.get(row["source1_entity_id"], set()),
        axis=1
    )
].copy()

hard_neg_s2["is_true"] = 0

print("S2 Type-1 negatives before cap:", len(hard_neg_s2))


# ------------------------------------------------
# S3: same raw business name + same country
# ------------------------------------------------

hard_neg_s3 = sample_s1[
    [
        "entity_id",
        "business_name",
        "country"
    ]
].merge(
    train_s3[
        [
            "entity_id",
            "business_name",
            "country"
        ]
    ],
    on=["business_name", "country"],
    how="inner",
    suffixes=("_s1", "_candidate")
)

hard_neg_s3 = hard_neg_s3.rename(
    columns={"entity_id_s1": "source1_entity_id"}
)

hard_neg_s3 = hard_neg_s3[
    hard_neg_s3["source1_entity_id"] !=
    hard_neg_s3["candidate_entity_id"]
].copy()

hard_neg_s3 = hard_neg_s3[
    ~hard_neg_s3.apply(
        lambda row:
        row["candidate_entity_id"]
        in true_match_map.get(row["source1_entity_id"], set()),
        axis=1
    )
].copy()

hard_neg_s3["is_true"] = 0

print("S3 Type-1 negatives before cap:", len(hard_neg_s3))


# ------------------------------------------------
# Keep at most 3 hard negatives per S1
# ------------------------------------------------

hard_neg_s2 = (
    hard_neg_s2
    .groupby("source1_entity_id", group_keys=False)
    .head(3)
    .reset_index(drop=True)
)

hard_neg_s3 = (
    hard_neg_s3
    .groupby("source1_entity_id", group_keys=False)
    .head(3)
    .reset_index(drop=True)
)

print("\nAfter maximum-3-per-S1 cap:")
print("S2:", len(hard_neg_s2))
print("S3:", len(hard_neg_s3))

# ------------------------------------------------
# Add source labels
# ------------------------------------------------

hard_neg_s2["source"] = "S2"
hard_neg_s3["source"] = "S3"

# ------------------------------------------------
# Combine
# ------------------------------------------------

negative_train_type1 = pd.concat(
    [hard_neg_s2, hard_neg_s3],
    ignore_index=True
)

print("\nTotal Type-1 negatives:",
      len(negative_train_type1))

# ------------------------------------------------
# Save immediately
# ------------------------------------------------

negative_train_type1.to_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type1.pkl"
    )
)

print("\nSaved:")
print(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type1.pkl"
    )
)


# In[22]:


# ============================================
# STEP 5: TYPE-1 HARD NEGATIVES
# Same business name + same country
# but NOT a true match
# ============================================

# --------------------------------------------
# 1. Build correct true-match lookup
# --------------------------------------------

true_match_map = {}

for _, row in sample_gt.iterrows():

    s1_id = row["source1_entity_id"]

    matched = str(row["matched_entity_ids"]).strip()

    if matched == "" or matched.lower() == "nan":
        true_match_map[s1_id] = set()
    else:
        true_match_map[s1_id] = {
            x.strip()
            for x in matched.split(",")
            if x.strip()
        }

print("True-match lookup created.")


# In[23]:


# --------------------------------------------
# 2. S2 hard negatives
# --------------------------------------------

s1_for_neg = sample_s1[
    [
        "entity_id",
        "business_name",
        "country"
    ]
].copy()

s2_for_neg = train_s2[
    [
        "entity_id",
        "business_name",
        "country"
    ]
].copy()

# Rename BEFORE merging so column names are unambiguous
s1_for_neg = s1_for_neg.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "business_name_s1"
    }
)

s2_for_neg = s2_for_neg.rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_name": "business_name_candidate"
    }
)

hard_neg_s2 = s1_for_neg.merge(
    s2_for_neg,
    left_on=["business_name_s1", "country"],
    right_on=["business_name_candidate", "country"],
    how="inner"
)

# Remove same ID if present
hard_neg_s2 = hard_neg_s2[
    hard_neg_s2["source1_entity_id"] !=
    hard_neg_s2["candidate_entity_id"]
].copy()

# Remove TRUE matches
hard_neg_s2 = hard_neg_s2[
    ~hard_neg_s2.apply(
        lambda row:
        row["candidate_entity_id"]
        in true_match_map.get(
            row["source1_entity_id"],
            set()
        ),
        axis=1
    )
].copy()

hard_neg_s2["is_true"] = 0
hard_neg_s2["source"] = "S2"

print("S2 Type-1 negatives before cap:", len(hard_neg_s2))


# In[24]:


# --------------------------------------------
# 3. S3 hard negatives
# --------------------------------------------

s3_for_neg = train_s3[
    [
        "entity_id",
        "business_name",
        "country"
    ]
].copy()

s3_for_neg = s3_for_neg.rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_name": "business_name_candidate"
    }
)

hard_neg_s3 = s1_for_neg.merge(
    s3_for_neg,
    left_on=["business_name_s1", "country"],
    right_on=["business_name_candidate", "country"],
    how="inner"
)

hard_neg_s3 = hard_neg_s3[
    hard_neg_s3["source1_entity_id"] !=
    hard_neg_s3["candidate_entity_id"]
].copy()

hard_neg_s3 = hard_neg_s3[
    ~hard_neg_s3.apply(
        lambda row:
        row["candidate_entity_id"]
        in true_match_map.get(
            row["source1_entity_id"],
            set()
        ),
        axis=1
    )
].copy()

hard_neg_s3["is_true"] = 0
hard_neg_s3["source"] = "S3"

print("S3 Type-1 negatives before cap:", len(hard_neg_s3))


# In[25]:


# --------------------------------------------
# 4. Keep maximum 3 negatives per S1
# --------------------------------------------

hard_neg_s2 = (
    hard_neg_s2
    .groupby("source1_entity_id", group_keys=False)
    .head(3)
    .reset_index(drop=True)
)

hard_neg_s3 = (
    hard_neg_s3
    .groupby("source1_entity_id", group_keys=False)
    .head(3)
    .reset_index(drop=True)
)

negative_train_type1 = pd.concat(
    [hard_neg_s2, hard_neg_s3],
    ignore_index=True
)

print("\nAfter cap:")
print("S2:", len(hard_neg_s2))
print("S3:", len(hard_neg_s3))
print("Total:", len(negative_train_type1))

# --------------------------------------------
# 5. Save locally
# --------------------------------------------

type1_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type1.pkl"
)

negative_train_type1.to_pickle(type1_path)

print("\nSaved:")
print(type1_path)


# In[26]:


print(negative_train_type1.shape)

print(
    negative_train_type1[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "country",
            "candidate_entity_id",
            "is_true",
            "source"
        ]
    ].head(15).to_string(index=False)
)


# In[27]:


# ============================================
# STEP 6A: TYPE-2 HARD NEGATIVES FROM S2
# Same normalized address + same country
# Different business name + NOT a true match
# ============================================

# We only need S1 address/name information
s1_address_base = sample_s1[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "address_norm"
    ]
].copy()

s1_address_base = s1_address_base.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "business_name_s1",
        "business_address": "business_address_s1",
        "name_norm": "name_norm_s1",
        "address_norm": "address_norm_s1"
    }
)

# Ignore empty addresses — otherwise many unrelated
# records with missing addresses could match.
s1_address_base = s1_address_base[
    s1_address_base["address_norm_s1"].str.len() > 0
].copy()

print("S1 records with usable addresses:",
      len(s1_address_base))

# ------------------------------------------------
# Create lookup key:
# country + normalized address
# ------------------------------------------------

s1_address_base["match_key"] = (
    s1_address_base["country"].fillna("").astype(str)
    + "||"
    + s1_address_base["address_norm_s1"]
)

# Keep only what is necessary for merging
s1_lookup = s1_address_base[
    [
        "source1_entity_id",
        "business_name_s1",
        "business_address_s1",
        "country",
        "name_norm_s1",
        "address_norm_s1",
        "match_key"
    ]
].copy()

print("S1 address lookup ready.")


# In[28]:


# ------------------------------------------------
# Scan S2 in chunks
# ------------------------------------------------

results_s2 = []

chunk_size = 500_000

for start in range(0, len(train_s2), chunk_size):

    end = min(start + chunk_size, len(train_s2))

    chunk = train_s2.iloc[start:end][
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    # Normalize address only
    chunk["address_norm"] = chunk[
        "business_address"
    ].apply(normalize_address)

    # Ignore missing/empty addresses
    chunk = chunk[
        chunk["address_norm"].str.len() > 0
    ].copy()

    # Build country + address key
    chunk["match_key"] = (
        chunk["country"].fillna("").astype(str)
        + "||"
        + chunk["address_norm"]
    )

    # Find only S2 rows whose address occurs
    # among our sampled S1 businesses
    matched = chunk.merge(
        s1_lookup,
        on=["match_key", "country"],
        how="inner"
    )

    if len(matched) > 0:

        # Normalize names only for these candidates
        matched["name_norm_candidate"] = (
            matched["business_name"]
            .apply(normalize_name)
        )

        # We want DIFFERENT normalized names
        matched = matched[
            matched["name_norm_candidate"]
            != matched["name_norm_s1"]
        ].copy()

        # Remove identical entity IDs
        matched = matched[
            matched["entity_id"]
            != matched["source1_entity_id"]
        ].copy()

        # Remove true matches
        matched = matched[
            ~matched.apply(
                lambda row:
                row["entity_id"]
                in true_match_map.get(
                    row["source1_entity_id"],
                    set()
                ),
                axis=1
            )
        ].copy()

        if len(matched) > 0:

            matched["candidate_entity_id"] = (
                matched["entity_id"]
            )

            matched["business_name_candidate"] = (
                matched["business_name"]
            )

            matched["business_address_candidate"] = (
                matched["business_address"]
            )

            matched["candidate_country"] = (
                matched["country"]
            )

            matched["is_true"] = 0

            results_s2.append(
                matched[
                    [
                        "source1_entity_id",
                        "business_name_s1",
                        "business_address_s1",
                        "candidate_country",
                        "name_norm_s1",
                        "address_norm",
                        "candidate_entity_id",
                        "business_name_candidate",
                        "business_address_candidate",
                        "name_norm_candidate",
                        "is_true"
                    ]
                ]
            )

    print(
        f"Processed {end:,} / {len(train_s2):,} "
        f"({end / len(train_s2) * 100:.1f}%)"
    )

# Combine all chunks
address_neg_s2 = pd.concat(
    results_s2,
    ignore_index=True
)

print("\nRaw Type-2 S2 candidates:",
      len(address_neg_s2))


# In[29]:


# ============================================
# Limit to maximum 3 Type-2 negatives per S1
# ============================================

address_neg_s2 = (
    address_neg_s2
    .groupby(
        "source1_entity_id",
        group_keys=False
    )
    .head(3)
    .reset_index(drop=True)
)

print("Final Type-2 S2 negatives:",
      len(address_neg_s2))

print("\nLabel distribution:")
print(address_neg_s2["is_true"].value_counts())

# --------------------------------------------
# Save immediately
# --------------------------------------------

type2_s2_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type2_s2.pkl"
)

address_neg_s2.to_pickle(type2_s2_path)

print("\nSaved:")
print(type2_s2_path)


# In[30]:


print(
    address_neg_s2[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "business_address_s1",
            "business_address_candidate",
            "candidate_country",
            "candidate_entity_id",
            "is_true"
        ]
    ].head(15).to_string(index=False)
)


# In[31]:


# ============================================
# STEP 6B: TYPE-2 HARD NEGATIVES FROM S3
# Same normalized address + same country
# Different business name + NOT a true match
# ============================================

results_s3 = []

chunk_size = 500_000

for start in range(0, len(train_s3), chunk_size):

    end = min(start + chunk_size, len(train_s3))

    chunk = train_s3.iloc[start:end][
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    # Normalize addresses
    chunk["address_norm"] = (
        chunk["business_address"]
        .apply(normalize_address)
    )

    # Ignore empty addresses
    chunk = chunk[
        chunk["address_norm"].str.len() > 0
    ].copy()

    # Build same key as S1
    chunk["match_key"] = (
        chunk["country"].fillna("").astype(str)
        + "||"
        + chunk["address_norm"]
    )

    # Match only against addresses occurring in our sampled S1
    matched = chunk.merge(
        s1_lookup,
        on=["match_key", "country"],
        how="inner"
    )

    if len(matched) > 0:

        # Normalize candidate names
        matched["name_norm_candidate"] = (
            matched["business_name"]
            .apply(normalize_name)
        )

        # Different business names
        matched = matched[
            matched["name_norm_candidate"]
            != matched["name_norm_s1"]
        ].copy()

        # Different entity IDs
        matched = matched[
            matched["entity_id"]
            != matched["source1_entity_id"]
        ].copy()

        # Remove actual ground-truth matches
        matched = matched[
            ~matched.apply(
                lambda row:
                row["entity_id"]
                in true_match_map.get(
                    row["source1_entity_id"],
                    set()
                ),
                axis=1
            )
        ].copy()

        if len(matched) > 0:

            matched["candidate_entity_id"] = (
                matched["entity_id"]
            )

            matched["business_name_candidate"] = (
                matched["business_name"]
            )

            matched["business_address_candidate"] = (
                matched["business_address"]
            )

            matched["candidate_country"] = (
                matched["country"]
            )

            matched["is_true"] = 0

            results_s3.append(
                matched[
                    [
                        "source1_entity_id",
                        "business_name_s1",
                        "business_address_s1",
                        "candidate_country",
                        "name_norm_s1",
                        "address_norm",
                        "candidate_entity_id",
                        "business_name_candidate",
                        "business_address_candidate",
                        "name_norm_candidate",
                        "is_true"
                    ]
                ]
            )

    print(
        f"Processed {end:,} / {len(train_s3):,} "
        f"({end / len(train_s3) * 100:.1f}%)"
    )

# Combine chunk results
address_neg_s3 = pd.concat(
    results_s3,
    ignore_index=True
)

print("\nRaw Type-2 S3 candidates:",
      len(address_neg_s3))


# In[32]:


# ============================================
# LIMIT + SAVE TYPE-2 S3 NEGATIVES
# ============================================

address_neg_s3 = (
    address_neg_s3
    .groupby(
        "source1_entity_id",
        group_keys=False
    )
    .head(3)
    .reset_index(drop=True)
)

print("Final Type-2 S3 negatives:",
      len(address_neg_s3))

print("\nLabel distribution:")
print(address_neg_s3["is_true"].value_counts())

# Save immediately
type2_s3_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type2_s3.pkl"
)

address_neg_s3.to_pickle(type2_s3_path)

print("\nSaved:")
print(type2_s3_path)


# In[33]:


print(
    address_neg_s3[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "business_address_s1",
            "business_address_candidate",
            "candidate_country",
            "candidate_entity_id",
            "is_true"
        ]
    ].head(15).to_string(index=False)
)


# In[34]:


# ============================================
# STEP 7: COMBINE POSITIVES + ALL NEGATIVES
# ============================================

# Load from disk as a safety measure
positive_train_raw = pd.read_pickle(
    os.path.join(PROJECT_DIR, "positive_train_raw.pkl")
)

negative_train_type1 = pd.read_pickle(
    os.path.join(PROJECT_DIR, "negative_train_type1.pkl")
)

address_neg_s2 = pd.read_pickle(
    os.path.join(PROJECT_DIR, "negative_train_type2_s2.pkl")
)

address_neg_s3 = pd.read_pickle(
    os.path.join(PROJECT_DIR, "negative_train_type2_s3.pkl")
)

print("Loaded:")
print("Positive:", len(positive_train_raw))
print("Type-1:", len(negative_train_type1))
print("Type-2 S2:", len(address_neg_s2))
print("Type-2 S3:", len(address_neg_s3))


# --------------------------------------------
# Helper: construct standardized pairs
# --------------------------------------------

def build_negative_pairs(neg_df, source):
    return pd.DataFrame({
        "source1_entity_id":
            neg_df["source1_entity_id"].values,

        "candidate_entity_id":
            neg_df["candidate_entity_id"].values,

        "source":
            source,

        "label":
            0
    })


# --------------------------------------------
# Positive pairs
# --------------------------------------------

positive_pairs = pd.DataFrame({
    "source1_entity_id":
        positive_train_raw["source1_entity_id"].values,

    "candidate_entity_id":
        positive_train_raw["candidate_entity_id"].values,

    "source":
        positive_train_raw["source"].values,

    "label":
        1
})


# --------------------------------------------
# Negative pairs
# --------------------------------------------

negative_pairs_1 = build_negative_pairs(
    negative_train_type1,
    "S2"  # temporary; will correct below
)

# Type-1 contains both S2 and S3, so retain its
# original source column
negative_pairs_1["source"] = (
    negative_train_type1["source"].values
)

negative_pairs_2_s2 = build_negative_pairs(
    address_neg_s2,
    "S2"
)

negative_pairs_2_s3 = build_negative_pairs(
    address_neg_s3,
    "S3"
)


# --------------------------------------------
# Combine
# --------------------------------------------

training_pairs_ids = pd.concat(
    [
        positive_pairs,
        negative_pairs_1,
        negative_pairs_2_s2,
        negative_pairs_2_s3
    ],
    ignore_index=True
)

# Remove accidental duplicates
training_pairs_ids = (
    training_pairs_ids
    .drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "source"
        ]
    )
    .reset_index(drop=True)
)

print("\nCombined training pairs:",
      len(training_pairs_ids))

print("\nLabels:")
print(training_pairs_ids["label"].value_counts())

print("\nSources:")
print(training_pairs_ids["source"].value_counts())


# In[35]:


# ============================================
# STEP 8: ATTACH S1 + CANDIDATE RECORDS
# ============================================

# S1 lookup
s1_lookup_full = train_s1[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
].rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "business_name_s1",
        "business_address": "business_address_s1",
        "country": "country_s1"
    }
)


# S2 lookup
s2_lookup_full = train_s2[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
].rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_name": "business_name_candidate",
        "business_address": "business_address_candidate",
        "country": "country_candidate"
    }
)


# S3 lookup
s3_lookup_full = train_s3[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
].rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_name": "business_name_candidate",
        "business_address": "business_address_candidate",
        "country": "country_candidate"
    }
)


# Split S2 and S3 so we use the correct lookup
training_s2 = training_pairs_ids[
    training_pairs_ids["source"] == "S2"
].copy()

training_s3 = training_pairs_ids[
    training_pairs_ids["source"] == "S3"
].copy()


# Attach S1 information
training_s2 = training_s2.merge(
    s1_lookup_full,
    on="source1_entity_id",
    how="left"
)

training_s3 = training_s3.merge(
    s1_lookup_full,
    on="source1_entity_id",
    how="left"
)


# Attach candidate information
training_s2 = training_s2.merge(
    s2_lookup_full,
    on="candidate_entity_id",
    how="left"
)

training_s3 = training_s3.merge(
    s3_lookup_full,
    on="candidate_entity_id",
    how="left"
)


# Combine again
training_pairs = pd.concat(
    [training_s2, training_s3],
    ignore_index=True
)

print("Training table shape:",
      training_pairs.shape)

print("\nColumns:")
print(training_pairs.columns.tolist())


# In[36]:


# ============================================
# STEP 9: CREATE ML FEATURES
# ============================================

print("Calculating similarity features...")


# --------------------------------------------
# Normalized text
# --------------------------------------------

training_pairs["name_norm_s1"] = (
    training_pairs["business_name_s1"]
    .apply(normalize_name)
)

training_pairs["name_norm_candidate"] = (
    training_pairs["business_name_candidate"]
    .apply(normalize_name)
)

training_pairs["address_norm_s1"] = (
    training_pairs["business_address_s1"]
    .apply(normalize_address)
)

training_pairs["address_norm_candidate"] = (
    training_pairs["business_address_candidate"]
    .apply(normalize_address)
)

training_pairs["name_translit_s1"] = (
    training_pairs["business_name_s1"]
    .apply(transliterate_name)
)

training_pairs["name_translit_candidate"] = (
    training_pairs["business_name_candidate"]
    .apply(transliterate_name)
)


# --------------------------------------------
# Exact matching features
# --------------------------------------------

training_pairs["name_exact"] = (
    training_pairs["name_norm_s1"]
    ==
    training_pairs["name_norm_candidate"]
).astype(int)

training_pairs["address_exact"] = (
    training_pairs["address_norm_s1"]
    ==
    training_pairs["address_norm_candidate"]
).astype(int)

training_pairs["both_exact"] = (
    (
        training_pairs["name_exact"] == 1
    )
    &
    (
        training_pairs["address_exact"] == 1
    )
).astype(int)

training_pairs["country_exact"] = (
    training_pairs["country_s1"].fillna("").str.lower()
    ==
    training_pairs["country_candidate"].fillna("").str.lower()
).astype(int)


# --------------------------------------------
# Fuzzy name features
# --------------------------------------------

training_pairs["name_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        training_pairs["name_norm_s1"],
        training_pairs["name_norm_candidate"]
    )
]

training_pairs["name_token_ratio"] = [
    token_set_ratio(a, b) / 100
    for a, b in zip(
        training_pairs["name_norm_s1"],
        training_pairs["name_norm_candidate"]
    )
]


# --------------------------------------------
# Fuzzy address features
# --------------------------------------------

training_pairs["address_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        training_pairs["address_norm_s1"],
        training_pairs["address_norm_candidate"]
    )
]

training_pairs["address_token_ratio"] = [
    token_set_ratio(a, b) / 100
    for a, b in zip(
        training_pairs["address_norm_s1"],
        training_pairs["address_norm_candidate"]
    )
]


# --------------------------------------------
# Transliteration features
# --------------------------------------------

training_pairs["name_translit_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        training_pairs["name_translit_s1"],
        training_pairs["name_translit_candidate"]
    )
]

training_pairs["name_translit_token_ratio"] = [
    token_set_ratio(a, b) / 100
    for a, b in zip(
        training_pairs["name_translit_s1"],
        training_pairs["name_translit_candidate"]
    )
]


# --------------------------------------------
# Address number overlap
# --------------------------------------------

training_pairs["number_overlap"] = [
    number_overlap(a, b)
    for a, b in zip(
        training_pairs["business_address_s1"],
        training_pairs["business_address_candidate"]
    )
]


# --------------------------------------------
# Missing-address indicators
# --------------------------------------------

training_pairs["s1_address_missing"] = (
    training_pairs["business_address_s1"].isna()
).astype(int)

training_pairs["candidate_address_missing"] = (
    training_pairs["business_address_candidate"].isna()
).astype(int)


# --------------------------------------------
# Source indicator
# --------------------------------------------

training_pairs["source_is_s3"] = (
    training_pairs["source"] == "S3"
).astype(int)


print("Feature calculation complete.")


# In[37]:


# ============================================
# STEP 10: SAVE TRAINING TABLE
# ============================================

training_pairs_path = os.path.join(
    PROJECT_DIR,
    "training_pairs.pkl"
)

training_pairs.to_pickle(
    training_pairs_path
)

print("Saved:")
print(training_pairs_path)

print("\nShape:")
print(training_pairs.shape)


# In[38]:


# ============================================
# STEP 11: FEATURE CHECK
# ============================================

feature_columns = [
    "name_ratio",
    "name_token_ratio",
    "name_translit_ratio",
    "name_translit_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "number_overlap",
    "country_exact",
    "name_exact",
    "address_exact",
    "both_exact",
    "s1_address_missing",
    "candidate_address_missing",
    "source_is_s3"
]

print(
    training_pairs[
        feature_columns + ["label"]
    ].describe()
)

print("\nMissing values:")
print(
    training_pairs[
        feature_columns
    ].isna().sum()
)

print("\nLabels:")
print(
    training_pairs["label"].value_counts()
)


# In[39]:


# ============================================
# STEP 12A: GROUPED TRAIN / VALIDATION SPLIT
# ============================================

from sklearn.model_selection import GroupShuffleSplit

feature_columns = [
    "name_ratio",
    "name_token_ratio",
    "name_translit_ratio",
    "name_translit_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "number_overlap",
    "name_exact",
    "address_exact",
    "both_exact",
    "candidate_address_missing",
    "source_is_s3"
]

X = training_pairs[feature_columns].copy()
y = training_pairs["label"].copy()

groups = training_pairs["source1_entity_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

groups_train = groups.iloc[train_idx]
groups_val = groups.iloc[val_idx]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

print("\nTraining S1 groups:",
      groups_train.nunique())

print("Validation S1 groups:",
      groups_val.nunique())

print("\nTraining labels:")
print(y_train.value_counts())

print("\nValidation labels:")
print(y_val.value_counts())


# In[40]:


overlap = set(groups_train) & set(groups_val)

print("S1 overlap:", len(overlap))


# In[41]:


# ============================================
# STEP 13: CHECK XGBOOST
# ============================================

import xgboost as xgb

print("XGBoost version:", xgb.__version__)


# In[42]:


get_ipython().run_line_magic('pip', 'install xgboost-cpu')


# In[43]:


import xgboost as xgb

print("XGBoost version:", xgb.__version__)


# In[97]:


# ============================================
# STEP 13A: NAME BLOCKING FOR TYPE-3
# ============================================

def normalize_name_core(text):
    """
    Remove common business/legal suffixes so that
    similar business names can fall into the same block.
    """

    name = normalize_name(text)

    tokens = name.split()

    legal_suffixes = {
        "llc",
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "corp",
        "corporation",
        "company",
        "co",
        "pvt",
        "private",
        "llp",
        "pllc",
        "lp",
        "pc",
        "pte",
        "gmbh",
        "sarl",
        "sa",
        "spa",
        "ag",
        "bv"
    }

    tokens = [
        token
        for token in tokens
        if token not in legal_suffixes
    ]

    return " ".join(tokens)


def make_name_block(text):
    name_core = normalize_name_core(text)

    tokens = name_core.split()

    if len(tokens) >= 2:
        return " ".join(tokens[:2])

    if len(tokens) == 1:
        return tokens[0]

    return ""


print(normalize_name_core("Bishop, Melton and Schmidt Foods Inc."))
print(normalize_name_core("Bishop Melton and Schmidt Foods LLC"))
print(make_name_block(
    normalize_name_core(
        "Bishop, Melton and Schmidt Foods Inc."
    )
))


# In[45]:


# ============================================
# STEP 13B: PREPARE S1 TYPE-3 LOOKUP
# ============================================

# Recreate the exact 10,000-S1 sample
sample_gt = ground_truth.sample(
    n=10000,
    random_state=42
).copy()

sample_s1_type3 = train_s1[
    train_s1["entity_id"].isin(
        sample_gt["source1_entity_id"]
    )
].copy()

sample_s1_type3["name_norm"] = (
    sample_s1_type3["business_name"]
    .apply(normalize_name)
)

sample_s1_type3["name_core"] = (
    sample_s1_type3["business_name"]
    .apply(normalize_name_core)
)

sample_s1_type3["name_block"] = (
    sample_s1_type3["name_core"]
    .apply(make_name_block)
)

sample_s1_type3["address_norm"] = (
    sample_s1_type3["business_address"]
    .apply(normalize_address)
)

# Ignore empty blocks
sample_s1_type3 = sample_s1_type3[
    sample_s1_type3["name_block"].str.len() > 0
].copy()

# Create country + name-block key
sample_s1_type3["match_key"] = (
    sample_s1_type3["country"].fillna("").astype(str)
    + "||"
    + sample_s1_type3["name_block"]
)

s1_type3_lookup = sample_s1_type3[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "name_core",
        "name_block",
        "address_norm",
        "match_key"
    ]
].rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "business_name_s1",
        "business_address": "business_address_s1",
        "name_norm": "name_norm_s1",
        "name_core": "name_core_s1",
        "address_norm": "address_norm_s1"
    }
)

print(
    "S1 records available for Type-3:",
    len(s1_type3_lookup)
)


# In[46]:


# ============================================
# STEP 13C: TYPE-3 NEGATIVES FROM S2
# Similar name + different address
# ============================================

results_type3_s2 = []

chunk_size = 500_000

for start in range(0, len(train_s2), chunk_size):

    end = min(start + chunk_size, len(train_s2))

    chunk = train_s2.iloc[start:end][
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    # Name normalization
    chunk["name_norm"] = (
        chunk["business_name"]
        .apply(normalize_name)
    )

    chunk["name_core"] = (
        chunk["business_name"]
        .apply(normalize_name_core)
    )

    chunk["name_block"] = (
        chunk["name_core"]
        .apply(make_name_block)
    )

    # Remove empty blocks
    chunk = chunk[
        chunk["name_block"].str.len() > 0
    ].copy()

    # Blocking key
    chunk["match_key"] = (
        chunk["country"].fillna("").astype(str)
        + "||"
        + chunk["name_block"]
    )

    # Merge only potential name-block matches
    matched = chunk.merge(
        s1_type3_lookup,
        on=["match_key", "country"],
        how="inner"
    )

    if len(matched) > 0:

        # Different entity
        matched = matched[
            matched["entity_id"]
            != matched["source1_entity_id"]
        ].copy()

        # Different normalized name
        matched = matched[
            matched["name_norm"]
            != matched["name_norm_s1"]
        ].copy()

        # Remove true matches
        matched = matched[
            ~matched.apply(
                lambda row:
                row["entity_id"]
                in true_match_map.get(
                    row["source1_entity_id"],
                    set()
                ),
                axis=1
            )
        ].copy()

        if len(matched) > 0:

            # Fuzzy name similarity
            matched["name_ratio"] = [
                ratio(a, b) / 100
                for a, b in zip(
                    matched["name_norm_s1"],
                    matched["name_norm"]
                )
            ]

            # Different-address requirement
            matched["address_norm_candidate"] = (
                matched["business_address"]
                .apply(normalize_address)
            )

            matched["address_ratio"] = [
                ratio(a, b) / 100
                for a, b in zip(
                    matched["address_norm_s1"],
                    matched["address_norm_candidate"]
                )
            ]

            # Hard-negative conditions
            matched = matched[
                (matched["name_ratio"] >= 0.75)
                &
                (matched["address_ratio"] <= 0.65)
            ].copy()

            if len(matched) > 0:

                matched["candidate_entity_id"] = (
                    matched["entity_id"]
                )

                matched["business_name_candidate"] = (
                    matched["business_name"]
                )

                matched["business_address_candidate"] = (
                    matched["business_address"]
                )

                matched["candidate_country"] = (
                    matched["country"]
                )

                matched["is_true"] = 0

                results_type3_s2.append(
                    matched[
                        [
                            "source1_entity_id",
                            "business_name_s1",
                            "business_address_s1",
                            "candidate_country",
                            "name_norm_s1",
                            "address_norm_s1",
                            "candidate_entity_id",
                            "business_name_candidate",
                            "business_address_candidate",
                            "name_norm",
                            "name_ratio",
                            "address_ratio",
                            "is_true"
                        ]
                    ].rename(
                        columns={
                            "name_norm":
                                "name_norm_candidate"
                        }
                    )
                )

    print(
        f"Processed {end:,} / {len(train_s2):,} "
        f"({end / len(train_s2) * 100:.1f}%)"
    )

address_type3_s2 = pd.concat(
    results_type3_s2,
    ignore_index=True
)

print(
    "\nRaw Type-3 S2 candidates:",
    len(address_type3_s2)
)


# In[47]:


# ============================================
# STEP 13D: CAP + SAVE TYPE-3 S2
# ============================================

type3_s2 = (
    address_type3_s2
    .sort_values(
        [
            "source1_entity_id",
            "name_ratio",
            "address_ratio"
        ],
        ascending=[True, False, True]
    )
    .groupby(
        "source1_entity_id",
        group_keys=False
    )
    .head(3)
    .reset_index(drop=True)
)

print(
    "Final Type-3 S2 negatives:",
    len(type3_s2)
)

print("\nLabels:")
print(type3_s2["is_true"].value_counts())

type3_s2_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type3_s2.pkl"
)

type3_s2.to_pickle(type3_s2_path)

print("\nSaved:")
print(type3_s2_path)


# In[48]:


print(
    type3_s2[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "business_address_s1",
            "business_address_candidate",
            "name_ratio",
            "address_ratio",
            "candidate_entity_id",
            "is_true"
        ]
    ].head(15).to_string(index=False)
)


# In[49]:


# ============================================
# STEP 13E: TYPE-3 NEGATIVES FROM S3
# Similar name + different address
# ============================================

results_type3_s3 = []

chunk_size = 500_000

for start in range(0, len(train_s3), chunk_size):

    end = min(start + chunk_size, len(train_s3))

    chunk = train_s3.iloc[start:end][
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    # -----------------------------
    # Normalize candidate names
    # -----------------------------
    chunk["name_norm"] = (
        chunk["business_name"]
        .apply(normalize_name)
    )

    chunk["name_core"] = (
        chunk["business_name"]
        .apply(normalize_name_core)
    )

    chunk["name_block"] = (
        chunk["name_core"]
        .apply(make_name_block)
    )

    # Ignore empty blocks
    chunk = chunk[
        chunk["name_block"].str.len() > 0
    ].copy()

    # -----------------------------
    # Blocking key
    # -----------------------------
    chunk["match_key"] = (
        chunk["country"].fillna("").astype(str)
        + "||"
        + chunk["name_block"]
    )

    # -----------------------------
    # Candidate generation
    # -----------------------------
    matched = chunk.merge(
        s1_type3_lookup,
        on=["match_key", "country"],
        how="inner"
    )

    if len(matched) > 0:

        # Different entity
        matched = matched[
            matched["entity_id"]
            != matched["source1_entity_id"]
        ].copy()

        # Different normalized name
        matched = matched[
            matched["name_norm"]
            != matched["name_norm_s1"]
        ].copy()

        # Remove ground-truth matches
        matched = matched[
            ~matched.apply(
                lambda row:
                row["entity_id"]
                in true_match_map.get(
                    row["source1_entity_id"],
                    set()
                ),
                axis=1
            )
        ].copy()

        if len(matched) > 0:

            # -----------------------------
            # Name similarity
            # -----------------------------
            matched["name_ratio"] = [
                ratio(a, b) / 100
                for a, b in zip(
                    matched["name_norm_s1"],
                    matched["name_norm"]
                )
            ]

            # -----------------------------
            # Address similarity
            # -----------------------------
            matched["address_norm_candidate"] = (
                matched["business_address"]
                .apply(normalize_address)
            )

            matched["address_ratio"] = [
                ratio(a, b) / 100
                for a, b in zip(
                    matched["address_norm_s1"],
                    matched["address_norm_candidate"]
                )
            ]

            # -----------------------------
            # Type-3 conditions
            # -----------------------------
            matched = matched[
                (matched["name_ratio"] >= 0.75)
                &
                (matched["address_ratio"] <= 0.65)
            ].copy()

            if len(matched) > 0:

                matched["candidate_entity_id"] = (
                    matched["entity_id"]
                )

                matched["business_name_candidate"] = (
                    matched["business_name"]
                )

                matched["business_address_candidate"] = (
                    matched["business_address"]
                )

                matched["candidate_country"] = (
                    matched["country"]
                )

                matched["is_true"] = 0

                results_type3_s3.append(
                    matched[
                        [
                            "source1_entity_id",
                            "business_name_s1",
                            "business_address_s1",
                            "candidate_country",
                            "name_norm_s1",
                            "address_norm_s1",
                            "candidate_entity_id",
                            "business_name_candidate",
                            "business_address_candidate",
                            "name_norm",
                            "name_ratio",
                            "address_ratio",
                            "is_true"
                        ]
                    ].rename(
                        columns={
                            "name_norm":
                                "name_norm_candidate"
                        }
                    )
                )

    print(
        f"Processed {end:,} / {len(train_s3):,} "
        f"({end / len(train_s3) * 100:.1f}%)"
    )


# Combine all chunks
address_type3_s3 = pd.concat(
    results_type3_s3,
    ignore_index=True
)

print(
    "\nRaw Type-3 S3 candidates:",
    len(address_type3_s3)
)


# In[50]:


# ============================================
# STEP 13F: CAP + SAVE TYPE-3 S3
# ============================================

type3_s3 = (
    address_type3_s3
    .sort_values(
        [
            "source1_entity_id",
            "name_ratio",
            "address_ratio"
        ],
        ascending=[True, False, True]
    )
    .groupby(
        "source1_entity_id",
        group_keys=False
    )
    .head(3)
    .reset_index(drop=True)
)

print(
    "Final Type-3 S3 negatives:",
    len(type3_s3)
)

print("\nLabel distribution:")
print(type3_s3["is_true"].value_counts())


# Save immediately
type3_s3_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type3_s3.pkl"
)

type3_s3.to_pickle(type3_s3_path)

print("\nSaved:")
print(type3_s3_path)


# In[51]:


print(
    type3_s3[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "business_address_s1",
            "business_address_candidate",
            "name_ratio",
            "address_ratio",
            "candidate_entity_id",
            "is_true"
        ]
    ].head(15).to_string(index=False)
)


# In[52]:


import os

type3_s3_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type3_s3.pkl"
)

print("Exists:", os.path.exists(type3_s3_path))
print("Path:", type3_s3_path)

if os.path.exists(type3_s3_path):
    print(
        "Rows:",
        len(pd.read_pickle(type3_s3_path))
    )


# In[53]:


# ============================================
# STEP 14A: COUNTRY-BASED CANDIDATE POOLS
# ============================================

# Maximum random candidates retained per country
POOL_SIZE = 30

# S2 reservoir: up to 30 random businesses per country
country_pool_s2 = (
    train_s2
    .groupby("country", group_keys=False)
    .apply(
        lambda x: x.sample(
            n=min(len(x), POOL_SIZE),
            random_state=42
        )
    )
    .reset_index(drop=True)
)

# S3 reservoir
country_pool_s3 = (
    train_s3
    .groupby("country", group_keys=False)
    .apply(
        lambda x: x.sample(
            n=min(len(x), POOL_SIZE),
            random_state=43
        )
    )
    .reset_index(drop=True)
)

print("S2 country-pool rows:", len(country_pool_s2))
print("S3 country-pool rows:", len(country_pool_s3))

print("\nCountries in S2 pool:",
      country_pool_s2["country"].nunique())

print("Countries in S3 pool:",
      country_pool_s3["country"].nunique())


# In[54]:


# ============================================
# STEP 14A — FIXED COUNTRY CANDIDATE POOLS
# ============================================

POOL_SIZE = 30

def make_country_pool(df, pool_size, random_state):
    pieces = []

    for country_value, group in df.groupby(
        "country",
        dropna=False,
        sort=False
    ):
        n = min(len(group), pool_size)

        sampled = group.sample(
            n=n,
            random_state=random_state
        )

        pieces.append(sampled)

    return pd.concat(
        pieces,
        ignore_index=True
    )


# S2 pool
country_pool_s2 = make_country_pool(
    train_s2,
    POOL_SIZE,
    42
)

# S3 pool
country_pool_s3 = make_country_pool(
    train_s3,
    POOL_SIZE,
    43
)

print("S2 country-pool rows:", len(country_pool_s2))
print("S3 country-pool rows:", len(country_pool_s3))

print(
    "\nS2 countries:",
    country_pool_s2["country"].nunique()
)

print(
    "S3 countries:",
    country_pool_s3["country"].nunique()
)

print("\nS2 pool columns:")
print(country_pool_s2.columns.tolist())

print("\nS3 pool columns:")
print(country_pool_s3.columns.tolist())


# In[55]:


type4_s2 = create_type4_negatives(
    sample_s1,
    country_pool_s2,
    "S2",
    max_candidates_per_s1=1
)

print("\nFinal Type-4 S2:", len(type4_s2))

print(
    type4_s2[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "country",
            "candidate_entity_id",
            "name_ratio",
            "address_ratio",
            "is_true"
        ]
    ].head(15).to_string(index=False)
)


# In[56]:


def create_type4_negatives(


# In[58]:


# ============================================
# STEP 14B: TYPE-4 NEGATIVE GENERATOR
# ============================================

def create_type4_negatives(
    s1_df,
    candidate_pool,
    source_name,
    max_candidates_per_s1=1
):
    """
    Create same-country but unrelated-business
    negative pairs.
    """

    s1_base = s1_df[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "address_norm"
        ]
    ].copy()

    s1_base = s1_base.rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "business_name_s1",
            "business_address": "business_address_s1",
            "name_norm": "name_norm_s1",
            "address_norm": "address_norm_s1"
        }
    )

    candidate_base = candidate_pool[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    candidate_base = candidate_base.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "business_name_candidate",
            "business_address": "business_address_candidate",
            "country": "candidate_country"
        }
    )

    # Same-country candidate generation
    candidates = s1_base.merge(
        candidate_base,
        left_on="country",
        right_on="candidate_country",
        how="inner"
    )

    print(
        source_name,
        "country-matched candidate pairs:",
        len(candidates)
    )

    # Remove accidental same IDs
    candidates = candidates[
        candidates["source1_entity_id"]
        != candidates["candidate_entity_id"]
    ].copy()

    # Remove TRUE matches
    candidates = candidates[
        ~candidates.apply(
            lambda row:
            row["candidate_entity_id"]
            in true_match_map.get(
                row["source1_entity_id"],
                set()
            ),
            axis=1
        )
    ].copy()

    # Normalize candidate names/addresses
    candidates["name_norm_candidate"] = (
        candidates["business_name_candidate"]
        .apply(normalize_name)
    )

    candidates["address_norm_candidate"] = (
        candidates["business_address_candidate"]
        .apply(normalize_address)
    )

    # Exclude same normalized names.
    # Those belong more naturally to Type 1.
    candidates = candidates[
        candidates["name_norm_candidate"]
        != candidates["name_norm_s1"]
    ].copy()

    # Fuzzy similarities
    candidates["name_ratio"] = [
        ratio(a, b) / 100
        for a, b in zip(
            candidates["name_norm_s1"],
            candidates["name_norm_candidate"]
        )
    ]

    candidates["address_ratio"] = [
        ratio(a, b) / 100
        for a, b in zip(
            candidates["address_norm_s1"],
            candidates["address_norm_candidate"]
        )
    ]

    # Prefer genuinely unrelated records
    unrelated = candidates[
        (candidates["name_ratio"] <= 0.50)
        &
        (candidates["address_ratio"] <= 0.60)
    ].copy()

    print(
        source_name,
        "strong unrelated candidates:",
        len(unrelated)
    )

    # If a country has too few suitable candidates,
    # retain the lowest-similarity candidate instead.
    candidates["combined_similarity"] = (
        0.6 * candidates["name_ratio"]
        + 0.4 * candidates["address_ratio"]
    )

    unrelated = (
        unrelated
        .sort_values(
            [
                "source1_entity_id",
                "name_ratio",
                "address_ratio"
            ],
            ascending=[True, True, True]
        )
        .groupby(
            "source1_entity_id",
            group_keys=False
        )
        .head(max_candidates_per_s1)
    )

    selected_ids = set(unrelated.index)

    fallback = candidates[
        ~candidates.index.isin(selected_ids)
    ].copy()

    # S1s already represented do not need fallback
    represented_s1 = set(
        unrelated["source1_entity_id"]
    )

    fallback = fallback[
        ~fallback["source1_entity_id"].isin(
            represented_s1
        )
    ].copy()

    fallback = (
        fallback
        .sort_values(
            [
                "source1_entity_id",
                "combined_similarity"
            ],
            ascending=[True, True]
        )
        .groupby(
            "source1_entity_id",
            group_keys=False
        )
        .head(max_candidates_per_s1)
    )

    result = pd.concat(
        [unrelated, fallback],
        ignore_index=True
    )

    # Final safety check
    result = result[
        ~result.apply(
            lambda row:
            row["candidate_entity_id"]
            in true_match_map.get(
                row["source1_entity_id"],
                set()
            ),
            axis=1
        )
    ].copy()

    result["is_true"] = 0
    result["source"] = source_name

    return result[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_address_s1",
            "country",
            "candidate_entity_id",
            "business_name_candidate",
            "business_address_candidate",
            "candidate_country",
            "name_norm_s1",
            "name_norm_candidate",
            "address_norm_s1",
            "address_norm_candidate",
            "name_ratio",
            "address_ratio",
            "is_true",
            "source"
        ]
    ]


# In[59]:


# ============================================
# STEP 14C: TYPE-4 S2
# ============================================

type4_s2 = create_type4_negatives(
    sample_s1,
    country_pool_s2,
    "S2",
    max_candidates_per_s1=1
)

print("\nFinal Type-4 S2:", len(type4_s2))

print(
    type4_s2[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "country",
            "candidate_entity_id",
            "name_ratio",
            "address_ratio",
            "is_true"
        ]
    ].head(15).to_string(index=False)
)


# In[60]:


type4_s2_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type4_s2.pkl"
)

type4_s2.to_pickle(type4_s2_path)

print("\nSaved:")
print(type4_s2_path)


# In[61]:


# ============================================
# STEP 14E: TYPE-4 S3
# ============================================

type4_s3 = create_type4_negatives(
    sample_s1,
    country_pool_s3,
    "S3",
    max_candidates_per_s1=1
)

print("\nFinal Type-4 S3:", len(type4_s3))

print(
    type4_s3[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "country",
            "candidate_entity_id",
            "name_ratio",
            "address_ratio",
            "is_true"
        ]
    ].head(15).to_string(index=False)
)


# In[62]:


type4_s3_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type4_s3.pkl"
)

type4_s3.to_pickle(type4_s3_path)

print("\nSaved:")
print(type4_s3_path)


# In[63]:


# ============================================
# STEP 15A: PREPARE TYPE-5 S1 LOOKUP
# ============================================

def has_non_ascii(text):
    if pd.isna(text):
        return False

    return any(ord(ch) > 127 for ch in str(text))


def make_type5_block(text):
    """
    Use the first two transliterated words.
    Require at least two words to avoid extremely
    common one-word blocks.
    """

    core = normalize_name_core(text)
    tokens = core.split()

    if len(tokens) >= 2:
        return " ".join(tokens[:2])

    return ""


sample_s1_type5 = sample_s1[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
].copy()

sample_s1_type5["name_norm"] = (
    sample_s1_type5["business_name"]
    .apply(normalize_name)
)

sample_s1_type5["name_translit"] = (
    sample_s1_type5["business_name"]
    .apply(transliterate_name)
)

sample_s1_type5["name_block"] = (
    sample_s1_type5["name_translit"]
    .apply(make_type5_block)
)

sample_s1_type5["address_norm"] = (
    sample_s1_type5["business_address"]
    .apply(normalize_address)
)

sample_s1_type5["has_non_ascii"] = (
    sample_s1_type5["business_name"]
    .apply(has_non_ascii)
)

sample_s1_type5["match_key"] = (
    sample_s1_type5["country"].fillna("").astype(str)
    + "||"
    + sample_s1_type5["name_block"]
)

# Only rows with a proper 2-word transliteration block
sample_s1_type5 = sample_s1_type5[
    sample_s1_type5["name_block"].str.len() > 0
].copy()

# Split by script
s1_ascii = sample_s1_type5[
    ~sample_s1_type5["has_non_ascii"]
].copy()

s1_nonascii = sample_s1_type5[
    sample_s1_type5["has_non_ascii"]
].copy()

print("Total S1 Type-5 candidates:",
      len(sample_s1_type5))

print("ASCII S1:", len(s1_ascii))
print("Non-ASCII S1:", len(s1_nonascii))


# In[64]:


# ============================================
# STEP 15B: TYPE-5 NEGATIVES FROM S2
# ============================================

results_type5_s2 = []

chunk_size = 500_000

for start in range(0, len(train_s2), chunk_size):

    end = min(start + chunk_size, len(train_s2))

    chunk = train_s2.iloc[start:end][
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()

    # ----------------------------------------
    # Detect scripts
    # ----------------------------------------

    chunk["has_non_ascii"] = (
        chunk["business_name"]
        .astype(str)
        .str.contains(
            r"[^\x00-\x7F]",
            regex=True,
            na=False
        )
    )

    # ----------------------------------------
    # Candidate ASCII records
    # These can conflict with NON-ASCII S1
    # ----------------------------------------

    candidate_ascii = chunk[
        ~chunk["has_non_ascii"]
    ].copy()

    if len(candidate_ascii) > 0:

        candidate_ascii["name_norm"] = (
            candidate_ascii["business_name"]
            .apply(normalize_name)
        )

        # ASCII text = its own transliteration
        candidate_ascii["name_translit"] = (
            candidate_ascii["name_norm"]
        )

        candidate_ascii["name_block"] = (
            candidate_ascii["name_translit"]
            .apply(make_type5_block)
        )

        candidate_ascii["address_norm_candidate"] = (
            candidate_ascii["business_address"]
            .apply(normalize_address)
        )

        candidate_ascii["match_key"] = (
            candidate_ascii["country"]
            .fillna("")
            .astype(str)
            + "||"
            + candidate_ascii["name_block"]
        )

        matched = candidate_ascii.merge(
            s1_nonascii[
                [
                    "entity_id",
                    "business_name",
                    "business_address",
                    "country",
                    "name_norm",
                    "name_translit",
                    "address_norm",
                    "match_key"
                ]
            ],
            on=["match_key", "country"],
            how="inner",
            suffixes=("_candidate", "_s1")
        )

        if len(matched) > 0:

            matched = matched[
                matched["entity_id"]
                != matched["entity_id_s1"]
            ].copy()

            # Remove true matches
            matched = matched[
                ~matched.apply(
                    lambda row:
                    row["entity_id"]
                    in true_match_map.get(
                        row["entity_id_s1"],
                        set()
                    ),
                    axis=1
                )
            ].copy()

            if len(matched) > 0:

                # ----------------------------
                # Similarity features
                # ----------------------------

                matched["name_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_norm_s1"],
                        matched["name_norm_candidate"]
                    )
                ]

                matched["name_translit_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_translit_s1"],
                        matched["name_translit_candidate"]
                    )
                ]

                matched["name_translit_token_ratio"] = [
                    token_set_ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_translit_s1"],
                        matched["name_translit_candidate"]
                    )
                ]

                matched["address_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["address_norm_s1"],
                        matched["address_norm_candidate"]
                    )
                ]

                # ----------------------------
                # Type-5 conditions
                # ----------------------------

                matched = matched[
                    (matched["name_translit_ratio"] >= 0.75)
                    &
                    (matched["name_translit_token_ratio"] >= 0.80)
                    &
                    (matched["name_ratio"] <= 0.50)
                    &
                    (matched["address_ratio"] <= 0.70)
                ].copy()

                if len(matched) > 0:

                    matched["candidate_entity_id"] = (
                        matched["entity_id"]
                    )

                    matched["source1_entity_id"] = (
                        matched["entity_id_s1"]
                    )

                    matched["business_name_s1"] = (
                        matched["business_name_s1"]
                    )

                    matched["business_name_candidate"] = (
                        matched["business_name_candidate"]
                    )

                    matched["business_address_s1"] = (
                        matched["business_address_s1"]
                    )

                    matched["business_address_candidate"] = (
                        matched["business_address_candidate"]
                    )

                    matched["candidate_country"] = (
                        matched["country"]
                    )

                    matched["is_true"] = 0
                    matched["source"] = "S2"

                    results_type5_s2.append(
                        matched[
                            [
                                "source1_entity_id",
                                "business_name_s1",
                                "business_address_s1",
                                "candidate_entity_id",
                                "business_name_candidate",
                                "business_address_candidate",
                                "candidate_country",
                                "name_norm_s1",
                                "name_norm_candidate",
                                "name_translit_s1",
                                "name_translit_candidate",
                                "name_ratio",
                                "name_translit_ratio",
                                "name_translit_token_ratio",
                                "address_ratio",
                                "is_true",
                                "source"
                            ]
                        ]
                    )

    # ----------------------------------------
    # Candidate NON-ASCII records
    # These conflict with ASCII S1
    # ----------------------------------------

    candidate_nonascii = chunk[
        chunk["has_non_ascii"]
    ].copy()

    if len(candidate_nonascii) > 0:

        candidate_nonascii["name_norm"] = (
            candidate_nonascii["business_name"]
            .apply(normalize_name)
        )

        candidate_nonascii["name_translit"] = (
            candidate_nonascii["business_name"]
            .apply(transliterate_name)
        )

        candidate_nonascii["name_block"] = (
            candidate_nonascii["name_translit"]
            .apply(make_type5_block)
        )

        candidate_nonascii["address_norm_candidate"] = (
            candidate_nonascii["business_address"]
            .apply(normalize_address)
        )

        candidate_nonascii["match_key"] = (
            candidate_nonascii["country"]
            .fillna("")
            .astype(str)
            + "||"
            + candidate_nonascii["name_block"]
        )

        matched = candidate_nonascii.merge(
            s1_ascii[
                [
                    "entity_id",
                    "business_name",
                    "business_address",
                    "country",
                    "name_norm",
                    "name_translit",
                    "address_norm",
                    "match_key"
                ]
            ],
            on=["match_key", "country"],
            how="inner",
            suffixes=("_candidate", "_s1")
        )

        if len(matched) > 0:

            matched = matched[
                matched["entity_id"]
                != matched["entity_id_s1"]
            ].copy()

            matched = matched[
                ~matched.apply(
                    lambda row:
                    row["entity_id"]
                    in true_match_map.get(
                        row["entity_id_s1"],
                        set()
                    ),
                    axis=1
                )
            ].copy()

            if len(matched) > 0:

                matched["name_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_norm_s1"],
                        matched["name_norm_candidate"]
                    )
                ]

                matched["name_translit_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_translit_s1"],
                        matched["name_translit_candidate"]
                    )
                ]

                matched["name_translit_token_ratio"] = [
                    token_set_ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_translit_s1"],
                        matched["name_translit_candidate"]
                    )
                ]

                matched["address_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["address_norm_s1"],
                        matched["address_norm_candidate"]
                    )
                ]

                matched = matched[
                    (matched["name_translit_ratio"] >= 0.75)
                    &
                    (matched["name_translit_token_ratio"] >= 0.80)
                    &
                    (matched["name_ratio"] <= 0.50)
                    &
                    (matched["address_ratio"] <= 0.70)
                ].copy()

                if len(matched) > 0:

                    matched["candidate_entity_id"] = (
                        matched["entity_id"]
                    )

                    matched["source1_entity_id"] = (
                        matched["entity_id_s1"]
                    )

                    matched["is_true"] = 0
                    matched["source"] = "S2"

                    results_type5_s2.append(
                        matched[
                            [
                                "source1_entity_id",
                                "business_name_s1",
                                "business_address_s1",
                                "candidate_entity_id",
                                "business_name_candidate",
                                "business_address_candidate",
                                "candidate_country",
                                "name_norm_s1",
                                "name_norm_candidate",
                                "name_translit_s1",
                                "name_translit_candidate",
                                "name_ratio",
                                "name_translit_ratio",
                                "name_translit_token_ratio",
                                "address_ratio",
                                "is_true",
                                "source"
                            ]
                        ]
                    )

    print(
        f"Processed {end:,} / {len(train_s2):,} "
        f"({end / len(train_s2) * 100:.1f}%)"
    )


# --------------------------------------------
# Combine results
# --------------------------------------------

type5_s2_raw = pd.concat(
    results_type5_s2,
    ignore_index=True
) if results_type5_s2 else pd.DataFrame()

print(
    "\nRaw Type-5 S2 candidates:",
    len(type5_s2_raw)
)


# In[65]:


# ============================================
# STEP 15B — FIXED TYPE-5 S2 SCAN
# Multilingual / transliteration hard negatives
# ============================================

results_type5_s2 = []

chunk_size = 500_000


# --------------------------------------------
# Prepare S1 lookups with explicit column names
# --------------------------------------------

s1_nonascii_lookup = s1_nonascii[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "name_translit",
        "address_norm",
        "match_key"
    ]
].rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "business_name_s1",
        "business_address": "business_address_s1",
        "name_norm": "name_norm_s1",
        "name_translit": "name_translit_s1",
        "address_norm": "address_norm_s1"
    }
)


s1_ascii_lookup = s1_ascii[
    [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "name_translit",
        "address_norm",
        "match_key"
    ]
].rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "business_name_s1",
        "business_address": "business_address_s1",
        "name_norm": "name_norm_s1",
        "name_translit": "name_translit_s1",
        "address_norm": "address_norm_s1"
    }
)


# ============================================
# Scan S2
# ============================================

for start in range(0, len(train_s2), chunk_size):

    end = min(start + chunk_size, len(train_s2))

    chunk = train_s2.iloc[start:end][
        [
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ].copy()


    # ----------------------------------------
    # Detect script
    # ----------------------------------------

    chunk["has_non_ascii"] = (
        chunk["business_name"]
        .fillna("")
        .astype(str)
        .str.contains(
            r"[^\x00-\x7F]",
            regex=True,
            na=False
        )
    )


    # ========================================
    # CASE 1:
    # ASCII candidate vs non-ASCII S1
    # ========================================

    candidate_ascii = chunk[
        ~chunk["has_non_ascii"]
    ].copy()

    if len(candidate_ascii) > 0:

        candidate_ascii = candidate_ascii.rename(
            columns={
                "entity_id": "candidate_entity_id",
                "business_name": "business_name_candidate",
                "business_address": "business_address_candidate"
            }
        )

        candidate_ascii["name_norm_candidate"] = (
            candidate_ascii["business_name_candidate"]
            .apply(normalize_name)
        )

        candidate_ascii["name_translit_candidate"] = (
            candidate_ascii["name_norm_candidate"]
        )

        candidate_ascii["address_norm_candidate"] = (
            candidate_ascii["business_address_candidate"]
            .apply(normalize_address)
        )

        candidate_ascii["name_block"] = (
            candidate_ascii["name_translit_candidate"]
            .apply(make_type5_block)
        )

        candidate_ascii["match_key"] = (
            candidate_ascii["country"]
            .fillna("")
            .astype(str)
            + "||"
            + candidate_ascii["name_block"]
        )

        matched = candidate_ascii.merge(
            s1_nonascii_lookup,
            on=["match_key", "country"],
            how="inner"
        )

        if len(matched) > 0:

            # Different entity
            matched = matched[
                matched["candidate_entity_id"]
                != matched["source1_entity_id"]
            ].copy()

            # Remove true matches
            matched = matched[
                ~matched.apply(
                    lambda row:
                    row["candidate_entity_id"]
                    in true_match_map.get(
                        row["source1_entity_id"],
                        set()
                    ),
                    axis=1
                )
            ].copy()

            if len(matched) > 0:

                matched["name_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_norm_s1"],
                        matched["name_norm_candidate"]
                    )
                ]

                matched["name_translit_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_translit_s1"],
                        matched["name_translit_candidate"]
                    )
                ]

                matched["name_translit_token_ratio"] = [
                    token_set_ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_translit_s1"],
                        matched["name_translit_candidate"]
                    )
                ]

                matched["address_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["address_norm_s1"],
                        matched["address_norm_candidate"]
                    )
                ]

                # Type-5 conditions
                matched = matched[
                    (matched["name_translit_ratio"] >= 0.75)
                    &
                    (matched["name_translit_token_ratio"] >= 0.80)
                    &
                    (matched["name_ratio"] <= 0.50)
                    &
                    (matched["address_ratio"] <= 0.70)
                ].copy()

                if len(matched) > 0:

                    matched["is_true"] = 0
                    matched["source"] = "S2"

                    results_type5_s2.append(
                        matched[
                            [
                                "source1_entity_id",
                                "business_name_s1",
                                "business_address_s1",
                                "candidate_entity_id",
                                "business_name_candidate",
                                "business_address_candidate",
                                "country",
                                "name_norm_s1",
                                "name_norm_candidate",
                                "name_translit_s1",
                                "name_translit_candidate",
                                "name_ratio",
                                "name_translit_ratio",
                                "name_translit_token_ratio",
                                "address_ratio",
                                "is_true",
                                "source"
                            ]
                        ]
                    )


    # ========================================
    # CASE 2:
    # Non-ASCII candidate vs ASCII S1
    # ========================================

    candidate_nonascii = chunk[
        chunk["has_non_ascii"]
    ].copy()

    if len(candidate_nonascii) > 0:

        candidate_nonascii = candidate_nonascii.rename(
            columns={
                "entity_id": "candidate_entity_id",
                "business_name": "business_name_candidate",
                "business_address": "business_address_candidate"
            }
        )

        candidate_nonascii["name_norm_candidate"] = (
            candidate_nonascii["business_name_candidate"]
            .apply(normalize_name)
        )

        candidate_nonascii["name_translit_candidate"] = (
            candidate_nonascii["business_name_candidate"]
            .apply(transliterate_name)
        )

        candidate_nonascii["address_norm_candidate"] = (
            candidate_nonascii["business_address_candidate"]
            .apply(normalize_address)
        )

        candidate_nonascii["name_block"] = (
            candidate_nonascii["name_translit_candidate"]
            .apply(make_type5_block)
        )

        candidate_nonascii["match_key"] = (
            candidate_nonascii["country"]
            .fillna("")
            .astype(str)
            + "||"
            + candidate_nonascii["name_block"]
        )

        matched = candidate_nonascii.merge(
            s1_ascii_lookup,
            on=["match_key", "country"],
            how="inner"
        )

        if len(matched) > 0:

            # Different entity
            matched = matched[
                matched["candidate_entity_id"]
                != matched["source1_entity_id"]
            ].copy()

            # Remove true matches
            matched = matched[
                ~matched.apply(
                    lambda row:
                    row["candidate_entity_id"]
                    in true_match_map.get(
                        row["source1_entity_id"],
                        set()
                    ),
                    axis=1
                )
            ].copy()

            if len(matched) > 0:

                matched["name_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_norm_s1"],
                        matched["name_norm_candidate"]
                    )
                ]

                matched["name_translit_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_translit_s1"],
                        matched["name_translit_candidate"]
                    )
                ]

                matched["name_translit_token_ratio"] = [
                    token_set_ratio(a, b) / 100
                    for a, b in zip(
                        matched["name_translit_s1"],
                        matched["name_translit_candidate"]
                    )
                ]

                matched["address_ratio"] = [
                    ratio(a, b) / 100
                    for a, b in zip(
                        matched["address_norm_s1"],
                        matched["address_norm_candidate"]
                    )
                ]

                # Type-5 conditions
                matched = matched[
                    (matched["name_translit_ratio"] >= 0.75)
                    &
                    (matched["name_translit_token_ratio"] >= 0.80)
                    &
                    (matched["name_ratio"] <= 0.50)
                    &
                    (matched["address_ratio"] <= 0.70)
                ].copy()

                if len(matched) > 0:

                    matched["is_true"] = 0
                    matched["source"] = "S2"

                    results_type5_s2.append(
                        matched[
                            [
                                "source1_entity_id",
                                "business_name_s1",
                                "business_address_s1",
                                "candidate_entity_id",
                                "business_name_candidate",
                                "business_address_candidate",
                                "country",
                                "name_norm_s1",
                                "name_norm_candidate",
                                "name_translit_s1",
                                "name_translit_candidate",
                                "name_ratio",
                                "name_translit_ratio",
                                "name_translit_token_ratio",
                                "address_ratio",
                                "is_true",
                                "source"
                            ]
                        ]
                    )


    print(
        f"Processed {end:,} / {len(train_s2):,} "
        f"({end / len(train_s2) * 100:.1f}%)"
    )


# ============================================
# Combine
# ============================================

if results_type5_s2:

    type5_s2_raw = pd.concat(
        results_type5_s2,
        ignore_index=True
    )

else:

    type5_s2_raw = pd.DataFrame()


print(
    "\nRaw Type-5 S2 candidates:",
    len(type5_s2_raw)
)


# In[66]:


# ============================================
# STEP 15C: INSPECT TYPE-5 S2 CANDIDATES
# ============================================

print("Number of raw Type-5 S2 candidates:",
      len(type5_s2_raw))

print(
    type5_s2_raw[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_translit_ratio",
            "name_translit_token_ratio",
            "business_address_s1",
            "business_address_candidate",
            "address_ratio",
            "candidate_entity_id",
            "is_true"
        ]
    ].to_string(index=False)
)


# In[67]:


# ============================================
# STEP 15D: CAP + SAVE TYPE-5 S2
# ============================================

type5_s2 = (
    type5_s2_raw
    .sort_values(
        [
            "source1_entity_id",
            "name_translit_ratio",
            "name_translit_token_ratio"
        ],
        ascending=[True, False, False]
    )
    .groupby(
        "source1_entity_id",
        group_keys=False
    )
    .head(3)
    .reset_index(drop=True)
)

print("Final Type-5 S2 negatives:", len(type5_s2))

print("\nLabels:")
print(type5_s2["is_true"].value_counts())

type5_s2_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type5_s2.pkl"
)

type5_s2.to_pickle(type5_s2_path)

print("\nSaved:")
print(type5_s2_path)


# In[69]:


# ============================================
# STEP 15E: TYPE-5 SCANNER
# Works for S2 or S3
# ============================================

def scan_type5_source(
    candidate_df,
    source_name,
    chunk_size=500_000
):

    results = []

    # ----------------------------------------
    # Explicit S1 lookup tables
    # ----------------------------------------

    s1_nonascii_lookup = s1_nonascii[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "name_translit",
            "address_norm",
            "match_key"
        ]
    ].rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "business_name_s1",
            "business_address": "business_address_s1",
            "name_norm": "name_norm_s1",
            "name_translit": "name_translit_s1",
            "address_norm": "address_norm_s1"
        }
    )

    s1_ascii_lookup = s1_ascii[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "name_translit",
            "address_norm",
            "match_key"
        ]
    ].rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "business_name_s1",
            "business_address": "business_address_s1",
            "name_norm": "name_norm_s1",
            "name_translit": "name_translit_s1",
            "address_norm": "address_norm_s1"
        }
    )

    # ----------------------------------------
    # Process candidate source in chunks
    # ----------------------------------------

    for start in range(
        0,
        len(candidate_df),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(candidate_df)
        )

        chunk = candidate_df.iloc[
            start:end
        ][
            [
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ].copy()

        # Detect non-ASCII business names
        chunk["has_nonascii"] = (
            chunk["business_name"]
            .fillna("")
            .astype(str)
            .str.contains(
                r"[^\x00-\x7F]",
                regex=True,
                na=False
            )
        )

        # ==================================================
        # CASE A:
        # ASCII candidate vs non-ASCII S1
        # ==================================================

        candidate_ascii = chunk[
            ~chunk["has_nonascii"]
        ].copy()

        if len(candidate_ascii) > 0:

            candidate_ascii = candidate_ascii.rename(
                columns={
                    "entity_id":
                        "candidate_entity_id",
                    "business_name":
                        "business_name_candidate",
                    "business_address":
                        "business_address_candidate"
                }
            )

            candidate_ascii[
                "name_norm_candidate"
            ] = (
                candidate_ascii[
                    "business_name_candidate"
                ].apply(normalize_name)
            )

            candidate_ascii[
                "name_translit_candidate"
            ] = candidate_ascii[
                "name_norm_candidate"
            ]

            candidate_ascii[
                "address_norm_candidate"
            ] = (
                candidate_ascii[
                    "business_address_candidate"
                ].apply(normalize_address)
            )

            candidate_ascii["name_block"] = (
                candidate_ascii[
                    "name_translit_candidate"
                ].apply(make_type5_block)
            )

            candidate_ascii["match_key"] = (
                candidate_ascii["country"]
                .fillna("")
                .astype(str)
                + "||"
                + candidate_ascii["name_block"]
            )

            matched = candidate_ascii.merge(
                s1_nonascii_lookup,
                on=["match_key", "country"],
                how="inner"
            )

            if len(matched) > 0:

                matched = matched[
                    matched["candidate_entity_id"]
                    !=
                    matched["source1_entity_id"]
                ].copy()

                # Remove genuine matches
                matched = matched[
                    ~matched.apply(
                        lambda row:
                        row["candidate_entity_id"]
                        in true_match_map.get(
                            row["source1_entity_id"],
                            set()
                        ),
                        axis=1
                    )
                ].copy()

                if len(matched) > 0:

                    matched["name_ratio"] = [
                        ratio(a, b) / 100
                        for a, b in zip(
                            matched["name_norm_s1"],
                            matched["name_norm_candidate"]
                        )
                    ]

                    matched["name_translit_ratio"] = [
                        ratio(a, b) / 100
                        for a, b in zip(
                            matched["name_translit_s1"],
                            matched["name_translit_candidate"]
                        )
                    ]

                    matched[
                        "name_translit_token_ratio"
                    ] = [
                        token_set_ratio(a, b) / 100
                        for a, b in zip(
                            matched["name_translit_s1"],
                            matched["name_translit_candidate"]
                        )
                    ]

                    matched["address_ratio"] = [
                        ratio(a, b) / 100
                        for a, b in zip(
                            matched["address_norm_s1"],
                            matched["address_norm_candidate"]
                        )
                    ]

                    matched = matched[
                        (matched["name_translit_ratio"] >= 0.75)
                        &
                        (matched["name_translit_token_ratio"] >= 0.80)
                        &
                        (matched["name_ratio"] <= 0.50)
                        &
                        (matched["address_ratio"] <= 0.70)
                    ].copy()

                    if len(matched) > 0:

                        matched["is_true"] = 0
                        matched["source"] = source_name

                        results.append(
                            matched[
                                [
                                    "source1_entity_id",
                                    "business_name_s1",
                                    "business_address_s1",
                                    "candidate_entity_id",
                                    "business_name_candidate",
                                    "business_address_candidate",
                                    "country",
                                    "name_norm_s1",
                                    "name_norm_candidate",
                                    "name_translit_s1",
                                    "name_translit_candidate",
                                    "name_ratio",
                                    "name_translit_ratio",
                                    "name_translit_token_ratio",
                                    "address_ratio",
                                    "is_true",
                                    "source"
                                ]
                            ]
                        )

        # ==================================================
        # CASE B:
        # Non-ASCII candidate vs ASCII S1
        # ==================================================

        candidate_nonascii = chunk[
            chunk["has_nonascii"]
        ].copy()

        if len(candidate_nonascii) > 0:

            candidate_nonascii = candidate_nonascii.rename(
                columns={
                    "entity_id":
                        "candidate_entity_id",
                    "business_name":
                        "business_name_candidate",
                    "business_address":
                        "business_address_candidate"
                }
            )

            candidate_nonascii[
                "name_norm_candidate"
            ] = (
                candidate_nonascii[
                    "business_name_candidate"
                ].apply(normalize_name)
            )

            candidate_nonascii[
                "name_translit_candidate"
            ] = (
                candidate_nonascii[
                    "business_name_candidate"
                ].apply(transliterate_name)
            )

            candidate_nonascii[
                "address_norm_candidate"
            ] = (
                candidate_nonascii[
                    "business_address_candidate"
                ].apply(normalize_address)
            )

            candidate_nonascii["name_block"] = (
                candidate_nonascii[
                    "name_translit_candidate"
                ].apply(make_type5_block)
            )

            candidate_nonascii["match_key"] = (
                candidate_nonascii["country"]
                .fillna("")
                .astype(str)
                + "||"
                + candidate_nonascii["name_block"]
            )

            matched = candidate_nonascii.merge(
                s1_ascii_lookup,
                on=["match_key", "country"],
                how="inner"
            )

            if len(matched) > 0:

                matched = matched[
                    matched["candidate_entity_id"]
                    !=
                    matched["source1_entity_id"]
                ].copy()

                # Remove genuine matches
                matched = matched[
                    ~matched.apply(
                        lambda row:
                        row["candidate_entity_id"]
                        in true_match_map.get(
                            row["source1_entity_id"],
                            set()
                        ),
                        axis=1
                    )
                ].copy()

                if len(matched) > 0:

                    matched["name_ratio"] = [
                        ratio(a, b) / 100
                        for a, b in zip(
                            matched["name_norm_s1"],
                            matched["name_norm_candidate"]
                        )
                    ]

                    matched["name_translit_ratio"] = [
                        ratio(a, b) / 100
                        for a, b in zip(
                            matched["name_translit_s1"],
                            matched["name_translit_candidate"]
                        )
                    ]

                    matched[
                        "name_translit_token_ratio"
                    ] = [
                        token_set_ratio(a, b) / 100
                        for a, b in zip(
                            matched["name_translit_s1"],
                            matched["name_translit_candidate"]
                        )
                    ]

                    matched["address_ratio"] = [
                        ratio(a, b) / 100
                        for a, b in zip(
                            matched["address_norm_s1"],
                            matched["address_norm_candidate"]
                        )
                    ]

                    matched = matched[
                        (matched["name_translit_ratio"] >= 0.75)
                        &
                        (matched["name_translit_token_ratio"] >= 0.80)
                        &
                        (matched["name_ratio"] <= 0.50)
                        &
                        (matched["address_ratio"] <= 0.70)
                    ].copy()

                    if len(matched) > 0:

                        matched["is_true"] = 0
                        matched["source"] = source_name

                        results.append(
                            matched[
                                [
                                    "source1_entity_id",
                                    "business_name_s1",
                                    "business_address_s1",
                                    "candidate_entity_id",
                                    "business_name_candidate",
                                    "business_address_candidate",
                                    "country",
                                    "name_norm_s1",
                                    "name_norm_candidate",
                                    "name_translit_s1",
                                    "name_translit_candidate",
                                    "name_ratio",
                                    "name_translit_ratio",
                                    "name_translit_token_ratio",
                                    "address_ratio",
                                    "is_true",
                                    "source"
                                ]
                            ]
                        )

        print(
            f"{source_name}: "
            f"Processed {end:,} / {len(candidate_df):,} "
            f"({end / len(candidate_df) * 100:.1f}%)"
        )

    if results:
        return pd.concat(
            results,
            ignore_index=True
        )

    return pd.DataFrame()


# In[70]:


# ============================================
# STEP 15F: TYPE-5 S3 SCAN
# ============================================

type5_s3_raw = scan_type5_source(
    train_s3,
    "S3"
)

print(
    "\nRaw Type-5 S3 candidates:",
    len(type5_s3_raw)
)


# In[71]:


# ============================================
# STEP 15G: INSPECT TYPE-5 S3 CANDIDATES
# ============================================

print(
    "Number of raw Type-5 S3 candidates:",
    len(type5_s3_raw)
)

print(
    type5_s3_raw[
        [
            "source1_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_translit_ratio",
            "name_translit_token_ratio",
            "business_address_s1",
            "business_address_candidate",
            "address_ratio",
            "candidate_entity_id",
            "is_true"
        ]
    ].to_string(index=False)
)


# In[72]:


# ============================================
# STEP 15H: CAP + SAVE TYPE-5 S3
# ============================================

type5_s3 = (
    type5_s3_raw
    .sort_values(
        [
            "source1_entity_id",
            "name_translit_ratio",
            "name_translit_token_ratio"
        ],
        ascending=[True, False, False]
    )
    .groupby(
        "source1_entity_id",
        group_keys=False
    )
    .head(3)
    .reset_index(drop=True)
)

print(
    "Final Type-5 S3 negatives:",
    len(type5_s3)
)

print("\nLabels:")
print(type5_s3["is_true"].value_counts())


type5_s3_path = os.path.join(
    PROJECT_DIR,
    "negative_train_type5_s3.pkl"
)

type5_s3.to_pickle(type5_s3_path)

print("\nSaved:")
print(type5_s3_path)


# In[73]:


print(
    "Exists:",
    os.path.exists(type5_s3_path)
)

if os.path.exists(type5_s3_path):
    print(
        "Rows:",
        len(pd.read_pickle(type5_s3_path))
    )


# In[74]:


# ============================================
# STEP 16: LOAD ALL SAVED TRAINING COMPONENTS
# ============================================

import os
import pandas as pd
import numpy as np

# --------------------------------------------
# Load saved files
# --------------------------------------------

positive_train_raw = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "positive_train_raw.pkl"
    )
)

negative_train_type1 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type1.pkl"
    )
)

type2_s2 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type2_s2.pkl"
    )
)

type2_s3 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type2_s3.pkl"
    )
)

type3_s2 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type3_s2.pkl"
    )
)

type3_s3 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type3_s3.pkl"
    )
)

type4_s2 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type4_s2.pkl"
    )
)

type4_s3 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type4_s3.pkl"
    )
)

type5_s2 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type5_s2.pkl"
    )
)

type5_s3 = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "negative_train_type5_s3.pkl"
    )
)


# --------------------------------------------
# Print sizes
# --------------------------------------------

print("POSITIVES")
print("Positive:", len(positive_train_raw))

print("\nNEGATIVE TYPES")
print("Type 1:", len(negative_train_type1))
print("Type 2 S2:", len(type2_s2))
print("Type 2 S3:", len(type2_s3))
print("Type 3 S2:", len(type3_s2))
print("Type 3 S3:", len(type3_s3))
print("Type 4 S2:", len(type4_s2))
print("Type 4 S3:", len(type4_s3))
print("Type 5 S2:", len(type5_s2))
print("Type 5 S3:", len(type5_s3))


# In[75]:


# ============================================
# STEP 17: BUILD UNIFIED TRAINING PAIRS
# ============================================

# --------------------------------------------
# Positive pairs
# --------------------------------------------

positive_pairs = pd.DataFrame({
    "source1_entity_id":
        positive_train_raw["source1_entity_id"].values,

    "candidate_entity_id":
        positive_train_raw["candidate_entity_id"].values,

    "source":
        positive_train_raw["source"].values,

    "label":
        1,

    "negative_type":
        "positive"
})


# --------------------------------------------
# Generic negative converter
# --------------------------------------------

def make_negative_pairs(
    df,
    source,
    negative_type
):
    return pd.DataFrame({
        "source1_entity_id":
            df["source1_entity_id"].values,

        "candidate_entity_id":
            df["candidate_entity_id"].values,

        "source":
            source,

        "label":
            0,

        "negative_type":
            negative_type
    })


# --------------------------------------------
# Each negative class
# --------------------------------------------

neg_type1 = make_negative_pairs(
    negative_train_type1,
    negative_train_type1["source"].iloc[0]
    if "source" not in negative_train_type1.columns
    else "mixed",
    "type1"
)

# Type-1 actually contains both S2 and S3,
# so preserve its original source column.
neg_type1["source"] = (
    negative_train_type1["source"].values
)


neg_type2_s2 = make_negative_pairs(
    type2_s2,
    "S2",
    "type2"
)

neg_type2_s3 = make_negative_pairs(
    type2_s3,
    "S3",
    "type2"
)

neg_type3_s2 = make_negative_pairs(
    type3_s2,
    "S2",
    "type3"
)

neg_type3_s3 = make_negative_pairs(
    type3_s3,
    "S3",
    "type3"
)

neg_type4_s2 = make_negative_pairs(
    type4_s2,
    "S2",
    "type4"
)

neg_type4_s3 = make_negative_pairs(
    type4_s3,
    "S3",
    "type4"
)

neg_type5_s2 = make_negative_pairs(
    type5_s2,
    "S2",
    "type5"
)

neg_type5_s3 = make_negative_pairs(
    type5_s3,
    "S3",
    "type5"
)


# --------------------------------------------
# Combine everything
# --------------------------------------------

training_pair_ids = pd.concat(
    [
        positive_pairs,
        neg_type1,
        neg_type2_s2,
        neg_type2_s3,
        neg_type3_s2,
        neg_type3_s3,
        neg_type4_s2,
        neg_type4_s3,
        neg_type5_s2,
        neg_type5_s3
    ],
    ignore_index=True
)


# --------------------------------------------
# Remove accidental duplicate pairs
# --------------------------------------------

training_pair_ids = (
    training_pair_ids
    .drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "source"
        ]
    )
    .reset_index(drop=True)
)


print("TOTAL PAIRS:",
      len(training_pair_ids))

print("\nLABEL COUNTS:")
print(
    training_pair_ids["label"]
    .value_counts()
)

print("\nNEGATIVE TYPE COUNTS:")
print(
    training_pair_ids[
        training_pair_ids["label"] == 0
    ]["negative_type"]
    .value_counts()
)

print("\nSOURCE COUNTS:")
print(
    training_pair_ids["source"]
    .value_counts()
)


# In[76]:


# ============================================
# STEP 18: SAVE COMBINED PAIRS
# ============================================

combined_ids_path = os.path.join(
    PROJECT_DIR,
    "training_pair_ids_all_types.pkl"
)

training_pair_ids.to_pickle(
    combined_ids_path
)

print("Saved:")
print(combined_ids_path)


# In[77]:


# ============================================
# STEP 19: ATTACH ACTUAL BUSINESS RECORDS
# ============================================

# Load our saved combined pair IDs
training_pair_ids = pd.read_pickle(
    os.path.join(
        PROJECT_DIR,
        "training_pair_ids_all_types.pkl"
    )
)

print("Training pair IDs:", len(training_pair_ids))


# ============================================
# 1. S1 records actually needed
# ============================================

needed_s1_ids = training_pair_ids[
    "source1_entity_id"
].unique()

needed_s1 = train_s1[
    train_s1["entity_id"].isin(needed_s1_ids)
][
    [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
].copy()

needed_s1 = needed_s1.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "business_name_s1",
        "business_address": "business_address_s1",
        "country": "country_s1"
    }
)

print("Needed S1 records:", len(needed_s1))


# ============================================
# 2. S2 records actually needed
# ============================================

needed_s2_ids = training_pair_ids.loc[
    training_pair_ids["source"] == "S2",
    "candidate_entity_id"
].unique()

needed_s2 = train_s2[
    train_s2["entity_id"].isin(needed_s2_ids)
][
    [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
].copy()

needed_s2 = needed_s2.rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_name": "business_name_candidate",
        "business_address": "business_address_candidate",
        "country": "country_candidate"
    }
)

print("Needed S2 records:", len(needed_s2))


# ============================================
# 3. S3 records actually needed
# ============================================

needed_s3_ids = training_pair_ids.loc[
    training_pair_ids["source"] == "S3",
    "candidate_entity_id"
].unique()

needed_s3 = train_s3[
    train_s3["entity_id"].isin(needed_s3_ids)
][
    [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
].copy()

needed_s3 = needed_s3.rename(
    columns={
        "entity_id": "candidate_entity_id",
        "business_name": "business_name_candidate",
        "business_address": "business_address_candidate",
        "country": "country_candidate"
    }
)

print("Needed S3 records:", len(needed_s3))


# ============================================
# 4. Split training pairs by source
# ============================================

training_s2 = training_pair_ids[
    training_pair_ids["source"] == "S2"
].copy()

training_s3 = training_pair_ids[
    training_pair_ids["source"] == "S3"
].copy()


# ============================================
# 5. Attach S1 + S2
# ============================================

training_s2 = training_s2.merge(
    needed_s1,
    on="source1_entity_id",
    how="left"
)

training_s2 = training_s2.merge(
    needed_s2,
    on="candidate_entity_id",
    how="left"
)


# ============================================
# 6. Attach S1 + S3
# ============================================

training_s3 = training_s3.merge(
    needed_s1,
    on="source1_entity_id",
    how="left"
)

training_s3 = training_s3.merge(
    needed_s3,
    on="candidate_entity_id",
    how="left"
)


# ============================================
# 7. Combine
# ============================================

training_pairs = pd.concat(
    [
        training_s2,
        training_s3
    ],
    ignore_index=True
)


print("\nFinal training table shape:")
print(training_pairs.shape)

print("\nColumns:")
print(training_pairs.columns.tolist())

print("\nMissing business names:")
print(
    training_pairs[
        [
            "business_name_s1",
            "business_name_candidate"
        ]
    ].isna().sum()
)

print("\nMissing addresses:")
print(
    training_pairs[
        [
            "business_address_s1",
            "business_address_candidate"
        ]
    ].isna().sum()
)


# In[78]:


# ============================================
# STEP 20: CALCULATE ML FEATURES
# ============================================

print("Creating normalized text...")

# --------------------------------------------
# Normalized names
# --------------------------------------------

training_pairs["name_norm_s1"] = (
    training_pairs["business_name_s1"]
    .apply(normalize_name)
)

training_pairs["name_norm_candidate"] = (
    training_pairs["business_name_candidate"]
    .apply(normalize_name)
)


# --------------------------------------------
# Normalized addresses
# --------------------------------------------

training_pairs["address_norm_s1"] = (
    training_pairs["business_address_s1"]
    .apply(normalize_address)
)

training_pairs["address_norm_candidate"] = (
    training_pairs["business_address_candidate"]
    .apply(normalize_address)
)


# --------------------------------------------
# Transliteration
# --------------------------------------------

print("Creating transliterated names...")

training_pairs["name_translit_s1"] = (
    training_pairs["business_name_s1"]
    .apply(transliterate_name)
)

training_pairs["name_translit_candidate"] = (
    training_pairs["business_name_candidate"]
    .apply(transliterate_name)
)


# ============================================
# Exact features
# ============================================

print("Creating exact-match features...")

training_pairs["name_exact"] = (
    training_pairs["name_norm_s1"]
    ==
    training_pairs["name_norm_candidate"]
).astype(int)

training_pairs["address_exact"] = (
    training_pairs["address_norm_s1"]
    ==
    training_pairs["address_norm_candidate"]
).astype(int)

training_pairs["both_exact"] = (
    (training_pairs["name_exact"] == 1)
    &
    (training_pairs["address_exact"] == 1)
).astype(int)

training_pairs["country_exact"] = (
    training_pairs["country_s1"]
    .fillna("")
    .astype(str)
    .str.lower()
    ==
    training_pairs["country_candidate"]
    .fillna("")
    .astype(str)
    .str.lower()
).astype(int)


# ============================================
# Fuzzy name features
# ============================================

print("Calculating name similarity...")

training_pairs["name_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        training_pairs["name_norm_s1"],
        training_pairs["name_norm_candidate"]
    )
]

training_pairs["name_token_ratio"] = [
    token_set_ratio(a, b) / 100
    for a, b in zip(
        training_pairs["name_norm_s1"],
        training_pairs["name_norm_candidate"]
    )
]


# ============================================
# Fuzzy address features
# ============================================

print("Calculating address similarity...")

training_pairs["address_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        training_pairs["address_norm_s1"],
        training_pairs["address_norm_candidate"]
    )
]

training_pairs["address_token_ratio"] = [
    token_set_ratio(a, b) / 100
    for a, b in zip(
        training_pairs["address_norm_s1"],
        training_pairs["address_norm_candidate"]
    )
]


# ============================================
# Transliteration similarity
# ============================================

print("Calculating transliteration similarity...")

training_pairs["name_translit_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        training_pairs["name_translit_s1"],
        training_pairs["name_translit_candidate"]
    )
]

training_pairs["name_translit_token_ratio"] = [
    token_set_ratio(a, b) / 100
    for a, b in zip(
        training_pairs["name_translit_s1"],
        training_pairs["name_translit_candidate"]
    )
]


# ============================================
# Address-number overlap
# ============================================

print("Calculating number overlap...")

training_pairs["number_overlap"] = [
    number_overlap(a, b)
    for a, b in zip(
        training_pairs["business_address_s1"],
        training_pairs["business_address_candidate"]
    )
]


# ============================================
# Missing-address indicators
# ============================================

training_pairs["s1_address_missing"] = (
    training_pairs["business_address_s1"]
    .isna()
).astype(int)

training_pairs["candidate_address_missing"] = (
    training_pairs["business_address_candidate"]
    .isna()
).astype(int)


# ============================================
# Source indicator
# ============================================

training_pairs["source_is_s3"] = (
    training_pairs["source"] == "S3"
).astype(int)


print("\nFeature creation complete.")


# In[79]:


# ============================================
# STEP 22: FINAL SANITY CHECK
# ============================================

feature_columns = [
    "name_ratio",
    "name_token_ratio",
    "name_translit_ratio",
    "name_translit_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "number_overlap",
    "name_exact",
    "address_exact",
    "both_exact",
    "candidate_address_missing",
    "source_is_s3"
]

print("FEATURE SUMMARY")
print(
    training_pairs[
        feature_columns
    ].describe().T
)

print("\nMISSING VALUES")
print(
    training_pairs[
        feature_columns
    ].isna().sum()
)

print("\nLABEL DISTRIBUTION")
print(
    training_pairs["label"].value_counts()
)

print("\nNEGATIVE TYPES")
print(
    training_pairs[
        training_pairs["label"] == 0
    ]["negative_type"].value_counts()
)


# In[80]:


# ============================================
# STEP 23: FINAL GROUPED TRAIN / VALIDATION SPLIT
# ============================================

from sklearn.model_selection import GroupShuffleSplit

feature_columns = [
    "name_ratio",
    "name_token_ratio",
    "name_translit_ratio",
    "name_translit_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "number_overlap",
    "name_exact",
    "address_exact",
    "both_exact",
    "candidate_address_missing",
    "source_is_s3"
]

X = training_pairs[feature_columns].copy()
y = training_pairs["label"].astype(int).copy()
groups = training_pairs["source1_entity_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_val = X.iloc[val_idx].copy()

y_train = y.iloc[train_idx].copy()
y_val = y.iloc[val_idx].copy()

groups_train = groups.iloc[train_idx].copy()
groups_val = groups.iloc[val_idx].copy()

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

print("\nTraining S1 groups:",
      groups_train.nunique())

print("Validation S1 groups:",
      groups_val.nunique())

print("\nTraining labels:")
print(y_train.value_counts())

print("\nValidation labels:")
print(y_val.value_counts())

overlap = (
    set(groups_train)
    &
    set(groups_val)
)

print("\nS1 overlap:", len(overlap))


# In[81]:


# ============================================
# STEP 24: TRAIN XGBOOST
# ============================================

import xgboost as xgb
from xgboost import XGBClassifier

print("XGBoost version:", xgb.__version__)

model = XGBClassifier(
    n_estimators=700,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.90,
    colsample_bytree=0.90,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Training completed.")


# In[82]:


# ============================================
# STEP 25: SAVE MODEL
# ============================================

import joblib

model_path = os.path.join(
    PROJECT_DIR,
    "xgboost_entity_resolution_model_v1.pkl"
)

joblib.dump(
    model,
    model_path
)

# Also save the exact feature list
feature_path = os.path.join(
    PROJECT_DIR,
    "feature_columns_v1.pkl"
)

joblib.dump(
    feature_columns,
    feature_path
)

print("Model saved:")
print(model_path)

print("\nFeature list saved:")
print(feature_path)


# In[83]:


# ============================================
# STEP 26: VALIDATION PROBABILITIES
# ============================================

val_prob = model.predict_proba(X_val)[:, 1]

validation_results = training_pairs.iloc[
    val_idx
].copy()

validation_results["match_probability"] = val_prob

print("Probability range:")
print("Minimum:", val_prob.min())
print("Maximum:", val_prob.max())
print("Mean:", val_prob.mean())

print("\nSample predictions:")

print(
    validation_results[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "source",
            "negative_type",
            "label",
            "match_probability"
        ]
    ]
    .head(25)
    .to_string(index=False)
)


# In[84]:


# ============================================
# STEP 27: COMPETITION-STYLE MACRO F0.5
# ============================================

def f05_score(precision, recall):
    """
    F0.5 gives more weight to precision.
    """
    if precision == 0 and recall == 0:
        return 0.0

    denominator = (
        0.25 * precision + recall
    )

    if denominator == 0:
        return 0.0

    return (
        1.25 * precision * recall
    ) / denominator


def macro_f05(
    df,
    probability_column,
    threshold
):
    scores = []

    for s1_id, group in df.groupby(
        "source1_entity_id"
    ):

        true_ids = set(
            group.loc[
                group["label"] == 1,
                "candidate_entity_id"
            ]
        )

        predicted_ids = set(
            group.loc[
                group[probability_column] >= threshold,
                "candidate_entity_id"
            ]
        )

        # Singleton / no true match
        if len(true_ids) == 0:

            if len(predicted_ids) == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)

            continue

        # True positives
        tp = len(
            true_ids & predicted_ids
        )

        # False positives
        fp = len(
            predicted_ids - true_ids
        )

        # False negatives
        fn = len(
            true_ids - predicted_ids
        )

        # Precision
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        # Recall
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        scores.append(
            f05_score(
                precision,
                recall
            )
        )

    return np.mean(scores)


# In[85]:


# ============================================
# STEP 27B: SEARCH BEST THRESHOLD
# ============================================

thresholds = np.arange(
    0.50,
    0.991,
    0.01
)

threshold_results = []

for threshold in thresholds:

    score = macro_f05(
        validation_results,
        "match_probability",
        threshold
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "macro_f05": score
        }
    )

threshold_df = pd.DataFrame(
    threshold_results
)

best_row = threshold_df.loc[
    threshold_df["macro_f05"].idxmax()
]

print("Best validation threshold:",
      best_row["threshold"])

print("Best validation Macro F0.5:",
      best_row["macro_f05"])


# In[86]:


# ============================================
# STEP 27C: VALIDATION CONFUSION SUMMARY
# ============================================

best_threshold = float(
    best_row["threshold"]
)

validation_results["prediction"] = (
    validation_results["match_probability"]
    >= best_threshold
).astype(int)

print("Best threshold:", best_threshold)

print("\nPair-level confusion:")
print(
    pd.crosstab(
        validation_results["label"],
        validation_results["prediction"],
        rownames=["Actual"],
        colnames=["Predicted"]
    )
)

print("\nPair-level counts:")
print(
    validation_results[
        ["label", "prediction"]
    ].value_counts()
)


# In[87]:


# ============================================
# STEP 28: FINER THRESHOLD SEARCH
# ============================================

fine_thresholds = np.arange(
    0.80,
    0.991,
    0.005
)

fine_results = []

for threshold in fine_thresholds:

    score = macro_f05(
        validation_results,
        "match_probability",
        threshold
    )

    fine_results.append({
        "threshold": round(float(threshold), 3),
        "macro_f05": score
    })

fine_threshold_df = pd.DataFrame(
    fine_results
)

fine_best = fine_threshold_df.loc[
    fine_threshold_df["macro_f05"].idxmax()
]

print("Best fine threshold:",
      fine_best["threshold"])

print("Best fine Macro F0.5:",
      fine_best["macro_f05"])

print("\nTop 15 thresholds:")
print(
    fine_threshold_df
    .sort_values(
        "macro_f05",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


# In[88]:


# ============================================
# STEP 29: FEATURE IMPORTANCE
# ============================================

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    feature_importance.to_string(index=False)
)


# In[89]:


# ============================================
# STEP 30: ERROR ANALYSIS
# ============================================

best_threshold = float(
    fine_best["threshold"]
)

validation_results["prediction"] = (
    validation_results["match_probability"]
    >= best_threshold
).astype(int)

# --------------------------------------------
# False positives
# --------------------------------------------

false_positives = validation_results[
    (validation_results["label"] == 0)
    &
    (validation_results["prediction"] == 1)
].copy()

# --------------------------------------------
# False negatives
# --------------------------------------------

false_negatives = validation_results[
    (validation_results["label"] == 1)
    &
    (validation_results["prediction"] == 0)
].copy()

print("Best threshold:", best_threshold)

print(
    "\nFalse positives:",
    len(false_positives)
)

print(
    "False negatives:",
    len(false_negatives)
)

print("\nFalse positives by negative type:")

print(
    false_positives[
        "negative_type"
    ].value_counts()
)

print("\nFalse negative probability range:")

print(
    false_negatives[
        "match_probability"
    ].describe()
)


# In[90]:


print(
    false_negatives
    .sort_values(
        "match_probability",
        ascending=False
    )[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "source",
            "business_name_s1",
            "business_name_candidate",
            "business_address_s1",
            "business_address_candidate",
            "name_ratio",
            "name_token_ratio",
            "name_translit_ratio",
            "address_ratio",
            "address_token_ratio",
            "number_overlap",
            "match_probability"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# In[91]:


# ============================================
# STEP 31: LOAD TEST DATA
# ============================================

TEST_DIR = r"E:\Hackathons_Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource\dataset\test"

print("Test folder exists:", os.path.exists(TEST_DIR))

test_s1 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source1.tsv"),
    sep="\t"
)

test_s2 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source2.tsv"),
    sep="\t"
)

test_s3 = pd.read_csv(
    os.path.join(TEST_DIR, "test_source3.tsv"),
    sep="\t"
)

print("\nTest S1:", test_s1.shape)
print("Test S2:", test_s2.shape)
print("Test S3:", test_s3.shape)

print("\nS1 columns:")
print(test_s1.columns.tolist())

print("\nS2 columns:")
print(test_s2.columns.tolist())

print("\nS3 columns:")
print(test_s3.columns.tolist())


# In[92]:


# ============================================
# STEP 33A: CANDIDATE GENERATION BLOCKS
# ============================================

def make_sorted_name_block(text):
    """
    Order-insensitive name block.
    Common legal suffixes are removed first.
    """

    core = normalize_name_core(text)
    tokens = core.split()

    if len(tokens) < 2:
        return ""

    tokens = sorted(set(tokens))

    return " ".join(tokens[:2])


def make_sorted_translit_block(text):
    """
    Order-insensitive transliterated-name block.
    """

    translit = transliterate_name(text)
    core = normalize_name_core(translit)
    tokens = core.split()

    if len(tokens) < 2:
        return ""

    tokens = sorted(set(tokens))

    return " ".join(tokens[:2])


print(
    make_sorted_name_block(
        "Bishop, Melton and Schmidt Foods Inc."
    )
)

print(
    make_sorted_name_block(
        "Schmidt Foods Bishop Melton LLC"
    )
)

print(
    make_sorted_translit_block(
        "गुरु Developers Private Limited"
    )
)


# In[93]:


# ============================================
# STEP 33B: PREPARE CANDIDATE RECALL TEST
# ============================================

# Recreate the exact same 10,000 S1 sample
eval_gt = ground_truth.sample(
    n=10000,
    random_state=42
).copy()

eval_s1 = train_s1[
    train_s1["entity_id"].isin(
        eval_gt["source1_entity_id"]
    )
].copy()

# --------------------------------------------
# Normalize S1
# --------------------------------------------

eval_s1["name_norm"] = (
    eval_s1["business_name"]
    .apply(normalize_name)
)

eval_s1["address_norm"] = (
    eval_s1["business_address"]
    .apply(normalize_address)
)

eval_s1["name_translit"] = (
    eval_s1["business_name"]
    .apply(transliterate_name)
)

eval_s1["name_core_block"] = (
    eval_s1["business_name"]
    .apply(make_name_block)
)

eval_s1["translit_core_block"] = (
    eval_s1["business_name"]
    .apply(
        lambda x:
        make_name_block(
            transliterate_name(x)
        )
    )
)

eval_s1["sorted_name_block"] = (
    eval_s1["business_name"]
    .apply(make_sorted_name_block)
)

eval_s1["sorted_translit_block"] = (
    eval_s1["business_name"]
    .apply(
        make_sorted_translit_block
    )
)

print("Evaluation S1:", len(eval_s1))


# In[94]:


# ============================================
# STEP 33C: TRUE MATCH LOOKUP
# ============================================

# Parse ground truth into individual match IDs
eval_matches = eval_gt.copy()

eval_matches["matched_entity_ids"] = (
    eval_matches["matched_entity_ids"]
    .fillna("")
    .astype(str)
)

eval_matches["matched_entity_ids"] = (
    eval_matches["matched_entity_ids"]
    .apply(
        lambda x:
        [
            i.strip()
            for i in x.split(",")
            if i.strip()
        ]
    )
)

eval_matches = eval_matches.explode(
    "matched_entity_ids"
).rename(
    columns={
        "matched_entity_ids":
            "candidate_entity_id"
    }
)

eval_matches = eval_matches[
    eval_matches["candidate_entity_id"].notna()
].copy()


# Only candidate IDs corresponding to each source
true_pair_keys = set(
    (
        row["source1_entity_id"],
        row["candidate_entity_id"]
    )
    for _, row in eval_matches.iterrows()
)

print(
    "Total true S1→S2/S3 pairs:",
    len(true_pair_keys)
)


# In[95]:


# ============================================
# STEP 33D: CANDIDATE RECALL EVALUATOR
# ============================================

def evaluate_candidate_blocks(
    source_df,
    source_name,
    eval_s1,
    true_pair_keys,
    chunk_size=500_000
):

    rules = [
        "exact_name",
        "exact_address",
        "exact_translit_name",
        "name_core_block",
        "translit_core_block",
        "sorted_name_block",
        "sorted_translit_block"
    ]

    captured = {
        rule: set()
        for rule in rules
    }

    candidate_counts = {
        rule: 0
        for rule in rules
    }

    # ----------------------------------------
    # Build S1 lookup for each rule
    # ----------------------------------------

    s1_base = eval_s1[
        [
            "entity_id",
            "country",
            "name_norm",
            "address_norm",
            "name_translit",
            "name_core_block",
            "translit_core_block",
            "sorted_name_block",
            "sorted_translit_block"
        ]
    ].copy()

    lookup_specs = {

        "exact_name": (
            "name_norm"
        ),

        "exact_address": (
            "address_norm"
        ),

        "exact_translit_name": (
            "name_translit"
        ),

        "name_core_block": (
            "name_core_block"
        ),

        "translit_core_block": (
            "translit_core_block"
        ),

        "sorted_name_block": (
            "sorted_name_block"
        ),

        "sorted_translit_block": (
            "sorted_translit_block"
        )
    }

    # ----------------------------------------
    # Scan candidate source
    # ----------------------------------------

    for start in range(
        0,
        len(source_df),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(source_df)
        )

        chunk = source_df.iloc[
            start:end
        ][
            [
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ].copy()

        # -------------------------------
        # Candidate normalized fields
        # -------------------------------

        chunk["name_norm"] = (
            chunk["business_name"]
            .apply(normalize_name)
        )

        chunk["address_norm"] = (
            chunk["business_address"]
            .apply(normalize_address)
        )

        chunk["name_translit"] = (
            chunk["business_name"]
            .apply(transliterate_name)
        )

        chunk["name_core_block"] = (
            chunk["business_name"]
            .apply(make_name_block)
        )

        chunk["translit_core_block"] = (
            chunk["business_name"]
            .apply(
                lambda x:
                make_name_block(
                    transliterate_name(x)
                )
            )
        )

        chunk["sorted_name_block"] = (
            chunk["business_name"]
            .apply(make_sorted_name_block)
        )

        chunk["sorted_translit_block"] = (
            chunk["business_name"]
            .apply(
                make_sorted_translit_block
            )
        )

        # --------------------------------
        # Apply every blocking rule
        # --------------------------------

        for rule, key_col in lookup_specs.items():

            lookup = s1_base[
                [
                    "entity_id",
                    "country",
                    key_col
                ]
            ].copy()

            # Ignore empty block keys
            lookup = lookup[
                lookup[key_col].fillna("").str.len() > 0
            ]

            candidates = chunk[
                [
                    "entity_id",
                    "country",
                    key_col
                ]
            ].copy()

            candidates = candidates[
                candidates[key_col]
                .fillna("")
                .str.len() > 0
            ]

            if len(lookup) == 0 or len(candidates) == 0:
                continue

            merged = candidates.merge(
                lookup,
                on=["country", key_col],
                how="inner",
                suffixes=(
                    "_candidate",
                    "_s1"
                )
            )

            if len(merged) == 0:
                continue

            # Remove self
            merged = merged[
                merged["entity_id_candidate"]
                !=
                merged["entity_id_s1"]
            ]

            candidate_counts[rule] += len(
                merged
            )

            # Check which generated pairs are true
            pair_keys = list(
                zip(
                    merged["entity_id_s1"],
                    merged["entity_id_candidate"]
                )
            )

            hit_mask = [
                pair in true_pair_keys
                for pair in pair_keys
            ]

            if any(hit_mask):

                captured[rule].update(
                    pair
                    for pair, hit in zip(
                        pair_keys,
                        hit_mask
                    )
                    if hit
                )

        print(
            f"{source_name}: "
            f"{end:,} / {len(source_df):,} "
            f"({end / len(source_df) * 100:.1f}%)"
        )

    return captured, candidate_counts


# In[98]:


# ============================================
# STEP 33E: S2 CANDIDATE RECALL
# ============================================

captured_s2, counts_s2 = evaluate_candidate_blocks(
    train_s2,
    "S2",
    eval_s1,
    true_pair_keys
)

total_true_s2 = len({
    pair
    for pair in true_pair_keys
    if pair[1].startswith("S2-")
})

print("\nS2 true pairs:", total_true_s2)

for rule in captured_s2:

    recall = (
        len(captured_s2[rule])
        / total_true_s2
        if total_true_s2 > 0
        else 0
    )

    print(
        f"{rule:25s} "
        f"captured={len(captured_s2[rule]):6d} "
        f"recall={recall:.4%} "
        f"candidates={counts_s2[rule]:,}"
    )


# In[99]:


# ============================================
# STEP 33G: S2 UNION RECALL
# ============================================

# We will NOT use the weak name_core_block
# in our preferred union.

selected_s2_rules = [
    "exact_name",
    "exact_address",
    "exact_translit_name",
    "translit_core_block",
    "sorted_name_block",
    "sorted_translit_block"
]

union_s2 = set()

print("Incremental S2 recall:\n")

for rule in selected_s2_rules:

    before = len(union_s2)

    union_s2.update(
        captured_s2[rule]
    )

    newly_added = (
        len(union_s2) - before
    )

    recall = (
        len(union_s2) / total_true_s2
    )

    print(
        f"{rule:25s} "
        f"new={newly_added:5d} "
        f"union={len(union_s2):5d} "
        f"recall={recall:.4%}"
    )

print("\nFinal S2 union:")
print("Captured true pairs:", len(union_s2))
print("Total true pairs:", total_true_s2)
print(
    "Union recall:",
    f"{len(union_s2) / total_true_s2:.4%}"
)


# In[100]:


# ============================================
# STEP 33H: S3 CANDIDATE RECALL
# ============================================

captured_s3, counts_s3 = evaluate_candidate_blocks(
    train_s3,
    "S3",
    eval_s1,
    true_pair_keys
)

total_true_s3 = len({
    pair
    for pair in true_pair_keys
    if pair[1].startswith("S3-")
})

print("\nS3 true pairs:", total_true_s3)

for rule in captured_s3:

    recall = (
        len(captured_s3[rule])
        / total_true_s3
        if total_true_s3 > 0
        else 0
    )

    print(
        f"{rule:25s} "
        f"captured={len(captured_s3[rule]):6d} "
        f"recall={recall:.4%} "
        f"candidates={counts_s3[rule]:,}"
    )


# In[101]:


# ============================================
# STEP 33G: S3 UNION RECALL
# ============================================

selected_s3_rules = [
    "exact_name",
    "exact_address",
    "exact_translit_name",
    "translit_core_block",
    "sorted_name_block",
    "sorted_translit_block"
]

union_s3 = set()

print("Incremental S3 recall:\n")

for rule in selected_s3_rules:

    before = len(union_s3)

    union_s3.update(
        captured_s3[rule]
    )

    newly_added = (
        len(union_s3) - before
    )

    recall = (
        len(union_s3) / total_true_s3
    )

    print(
        f"{rule:25s} "
        f"new={newly_added:6d} "
        f"union={len(union_s3):6d} "
        f"recall={recall:.4%}"
    )

print("\nFinal S3 union:")
print("Captured true pairs:", len(union_s3))
print("Total true pairs:", total_true_s3)
print(
    "Union recall:",
    f"{len(union_s3) / total_true_s3:.4%}"
)


# In[102]:


# ============================================
# STEP 33H: COMBINED S2 + S3 UNION
# ============================================

combined_true = (
    true_pair_keys
)

combined_union = (
    union_s2
    |
    union_s3
)

print("Total true pairs:", len(combined_true))
print("Captured by union:", len(combined_union))

print(
    "Combined recall:",
    f"{len(combined_union) / len(combined_true):.4%}"
)

print("\nMissed true pairs:",
      len(combined_true - combined_union))


# In[103]:


# ============================================
# STEP 34A: DISTINCTIVE NAME TOKENS
# ============================================

GENERIC_NAME_TOKENS = {
    "private",
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "pvt",
    "llp",
    "pllc",
    "lp",
    "pc",
    "pte",
    "gmbh",
    "sarl",
    "sa",
    "spa",
    "ag",
    "bv",
    "services",
    "service",
    "group",
    "enterprise",
    "enterprises",
    "industries",
    "industry",
    "solutions",
    "solution",
    "international",
    "india",
    "global",
    "trading",
    "business"
}


def get_name_tokens(text):
    """
    Return useful normalized name tokens.
    Generic/legal words are removed.
    """

    name = normalize_name(text)

    tokens = name.split()

    useful_tokens = [
        token
        for token in tokens
        if token not in GENERIC_NAME_TOKENS
        and len(token) >= 3
    ]

    return list(dict.fromkeys(useful_tokens))


def get_translit_tokens(text):
    """
    Return useful transliterated name tokens.
    """

    name = transliterate_name(text)

    tokens = name.split()

    useful_tokens = [
        token
        for token in tokens
        if token not in GENERIC_NAME_TOKENS
        and len(token) >= 3
    ]

    return list(dict.fromkeys(useful_tokens))


print(
    get_name_tokens(
        "Shakti Management Private Limited"
    )
)

print(
    get_translit_tokens(
        "गुरु Developers प्राइवेट लिमिटेड"
    )
)


# In[104]:


# ============================================
# STEP 34B: COUNT S2 NAME-TOKEN FREQUENCIES
# ============================================

from collections import Counter

s2_token_frequency = Counter()

chunk_size = 500_000

for start in range(
    0,
    len(train_s2),
    chunk_size
):

    end = min(
        start + chunk_size,
        len(train_s2)
    )

    chunk_names = train_s2.iloc[
        start:end
    ]["business_name"]

    for name in chunk_names:

        tokens = set(
            get_name_tokens(name)
        )

        translit_tokens = set(
            get_translit_tokens(name)
        )

        all_tokens = (
            tokens |
            translit_tokens
        )

        for token in all_tokens:
            s2_token_frequency[token] += 1

    print(
        f"Processed {end:,} / "
        f"{len(train_s2):,}"
    )


print(
    "\nUnique tokens:",
    len(s2_token_frequency)
)

print("\nMost common useful tokens:")

print(
    s2_token_frequency.most_common(30)
)


# In[105]:


# ============================================
# STEP 34C: SELECT DISTINCTIVE TOKENS
# ============================================

MAX_TOKEN_FREQUENCY = 5000
MAX_TOKENS_PER_S1 = 3


def get_distinctive_tokens(text):

    tokens = set(
        get_name_tokens(text)
    )

    translit_tokens = set(
        get_translit_tokens(text)
    )

    all_tokens = (
        tokens |
        translit_tokens
    )

    # Only retain tokens that are reasonably rare
    candidates = [
        token
        for token in all_tokens
        if s2_token_frequency.get(
            token,
            0
        ) <= MAX_TOKEN_FREQUENCY
    ]

    # Rarest first
    candidates.sort(
        key=lambda token:
        s2_token_frequency.get(
            token,
            10**12
        )
    )

    return candidates[
        :MAX_TOKENS_PER_S1
    ]


eval_s1["distinctive_tokens"] = (
    eval_s1["business_name"]
    .apply(get_distinctive_tokens)
)

print(
    eval_s1[
        [
            "business_name",
            "distinctive_tokens"
        ]
    ].head(30).to_string(index=False)
)


# In[106]:


# ============================================
# STEP 34D: DISTINCTIVE-TOKEN RECALL ON S2
# ============================================

# Create token → S1 lookup
token_to_s1 = {}

for _, row in eval_s1.iterrows():

    s1_id = row["entity_id"]

    country = row["country"]

    for token in row["distinctive_tokens"]:

        key = (
            str(country)
            + "||"
            + token
        )

        token_to_s1.setdefault(
            key,
            set()
        ).add(s1_id)


captured_distinctive_s2 = set()
distinctive_candidate_count_s2 = 0

chunk_size = 500_000

for start in range(
    0,
    len(train_s2),
    chunk_size
):

    end = min(
        start + chunk_size,
        len(train_s2)
    )

    chunk = train_s2.iloc[
        start:end
    ][
        [
            "entity_id",
            "business_name",
            "country"
        ]
    ].copy()

    for _, row in chunk.iterrows():

        candidate_id = row["entity_id"]
        country = row["country"]

        tokens = (
            set(get_name_tokens(
                row["business_name"]
            ))
            |
            set(get_translit_tokens(
                row["business_name"]
            ))
        )

        matched_s1_ids = set()

        for token in tokens:

            key = (
                str(country)
                + "||"
                + token
            )

            s1_ids = token_to_s1.get(
                key,
                set()
            )

            matched_s1_ids.update(
                s1_ids
            )

        for s1_id in matched_s1_ids:

            if candidate_id == s1_id:
                continue

            pair = (
                s1_id,
                candidate_id
            )

            distinctive_candidate_count_s2 += 1

            if pair in true_pair_keys:

                captured_distinctive_s2.add(
                    pair
                )

    print(
        f"Processed {end:,} / "
        f"{len(train_s2):,}"
    )


distinctive_recall_s2 = (
    len(captured_distinctive_s2)
    /
    total_true_s2
)

print("\nDistinctive-token results:")
print(
    "Captured true pairs:",
    len(captured_distinctive_s2)
)

print(
    "S2 recall:",
    f"{distinctive_recall_s2:.4%}"
)

print(
    "Candidate pairs generated:",
    f"{distinctive_candidate_count_s2:,}"
)


# In[107]:


# ============================================
# STEP 34E: HOW MANY NEW TRUE MATCHES?
# ============================================

old_s2_union = union_s2.copy()

new_s2_union = (
    old_s2_union
    |
    captured_distinctive_s2
)

newly_recovered = (
    len(new_s2_union)
    -
    len(old_s2_union)
)

new_recall = (
    len(new_s2_union)
    /
    total_true_s2
)

print(
    "Previously captured:",
    len(old_s2_union)
)

print(
    "Newly recovered:",
    newly_recovered
)

print(
    "New total captured:",
    len(new_s2_union)
)

print(
    "New S2 recall:",
    f"{new_recall:.4%}"
)

print(
    "Still missed:",
    total_true_s2 - len(new_s2_union)
)


# In[108]:


# ============================================
# STEP 34F: TEST DISTINCTIVE-TOKEN THRESHOLDS
# Uses known TRUE pairs only.
# No 5-million-row scan.
# ============================================

# --------------------------------------------
# Get the true S2 pairs from our 10k evaluation
# --------------------------------------------

s2_true_eval = eval_matches[
    eval_matches["candidate_entity_id"].astype(str).str.startswith("S2-")
].copy()

print("True S2 pairs:", len(s2_true_eval))


# --------------------------------------------
# Attach S1 names
# --------------------------------------------

s2_true_eval = s2_true_eval.merge(
    eval_s1[
        [
            "entity_id",
            "business_name"
        ]
    ].rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "business_name_s1"
        }
    ),
    on="source1_entity_id",
    how="left"
)


# --------------------------------------------
# Attach candidate names
# --------------------------------------------

s2_true_eval = s2_true_eval.merge(
    train_s2[
        [
            "entity_id",
            "business_name"
        ]
    ].rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "business_name_candidate"
        }
    ),
    on="candidate_entity_id",
    how="left"
)


# --------------------------------------------
# Precompute tokens
# --------------------------------------------

s1_token_map = {}

for _, row in eval_s1.iterrows():

    s1_id = row["entity_id"]

    tokens = (
        set(get_name_tokens(row["business_name"]))
        |
        set(get_translit_tokens(row["business_name"]))
    )

    s1_token_map[s1_id] = tokens


candidate_token_map = {}

for _, row in s2_true_eval.iterrows():

    candidate_id = row["candidate_entity_id"]

    if candidate_id not in candidate_token_map:

        tokens = (
            set(get_name_tokens(
                row["business_name_candidate"]
            ))
            |
            set(get_translit_tokens(
                row["business_name_candidate"]
            ))
        )

        candidate_token_map[candidate_id] = tokens


# --------------------------------------------
# Thresholds to test
# --------------------------------------------

thresholds = [
    100,
    250,
    500,
    1000,
    2000,
    3000,
    5000
]


threshold_results = []


for max_freq in thresholds:

    # ----------------------------------------
    # Select up to 3 rare tokens per S1
    # ----------------------------------------

    selected_tokens_map = {}

    for s1_id, tokens in s1_token_map.items():

        candidates = [
            token
            for token in tokens
            if s2_token_frequency.get(
                token,
                10**12
            ) <= max_freq
        ]

        candidates.sort(
            key=lambda token:
            s2_token_frequency.get(
                token,
                10**12
            )
        )

        selected_tokens_map[s1_id] = candidates[:3]


    # ----------------------------------------
    # Evaluate TRUE-pair recall
    # ----------------------------------------

    captured = 0

    for _, row in s2_true_eval.iterrows():

        s1_id = row["source1_entity_id"]
        candidate_id = row["candidate_entity_id"]

        selected = set(
            selected_tokens_map.get(
                s1_id,
                []
            )
        )

        candidate_tokens = candidate_token_map.get(
            candidate_id,
            set()
        )

        if selected & candidate_tokens:
            captured += 1


    recall_value = (
        captured / len(s2_true_eval)
    )


    # ----------------------------------------
    # Estimate candidate volume.
    #
    # This is an UPPER-BOUND estimate because
    # candidates shared by multiple tokens can
    # overlap.
    # ----------------------------------------

    estimated_candidates = 0

    for tokens in selected_tokens_map.values():

        for token in tokens:

            estimated_candidates += (
                s2_token_frequency.get(
                    token,
                    0
                )
            )


    threshold_results.append({
        "max_frequency": max_freq,
        "captured_true_pairs": captured,
        "recall": recall_value,
        "estimated_candidate_upper_bound":
            estimated_candidates
    })


threshold_test_df = pd.DataFrame(
    threshold_results
)

threshold_test_df["recall_percent"] = (
    threshold_test_df["recall"] * 100
)

print(
    threshold_test_df.to_string(
        index=False
    )
)


# In[109]:


# ============================================
# STEP 34G: OPTIMIZE DISTINCTIVE TOKEN BLOCK
# ============================================

# Existing candidate recall before distinctive-token blocking
old_union_s2 = union_s2.copy()

# We will test:
# - frequency limits
# - number of selected tokens per S1

frequency_limits = [
    100,
    250,
    500,
    1000,
    2000,
    3000,
    5000
]

token_counts = [1, 2, 3]

results = []

for max_freq in frequency_limits:

    for max_tokens in token_counts:

        # ------------------------------------
        # Select rarest N tokens per S1
        # ------------------------------------

        selected_tokens_map = {}

        for s1_id, tokens in s1_token_map.items():

            candidates = [
                token
                for token in tokens
                if s2_token_frequency.get(
                    token,
                    10**12
                ) <= max_freq
            ]

            candidates.sort(
                key=lambda token:
                s2_token_frequency.get(
                    token,
                    10**12
                )
            )

            selected_tokens_map[s1_id] = (
                candidates[:max_tokens]
            )

        # ------------------------------------
        # Capture true pairs using this block
        # ------------------------------------

        captured = set()

        for _, row in s2_true_eval.iterrows():

            s1_id = row["source1_entity_id"]
            candidate_id = row["candidate_entity_id"]

            selected = set(
                selected_tokens_map.get(
                    s1_id,
                    []
                )
            )

            candidate_tokens = (
                candidate_token_map.get(
                    candidate_id,
                    set()
                )
            )

            if selected & candidate_tokens:

                captured.add(
                    (
                        s1_id,
                        candidate_id
                    )
                )

        # ------------------------------------
        # Add to existing union
        # ------------------------------------

        combined = (
            old_union_s2
            |
            captured
        )

        # Newly recovered beyond old blocks
        newly_recovered = (
            len(combined - old_union_s2)
        )

        # ------------------------------------
        # Estimate candidate volume
        # ------------------------------------

        estimated_volume = 0

        for tokens in selected_tokens_map.values():

            for token in tokens:

                estimated_volume += (
                    s2_token_frequency.get(
                        token,
                        0
                    )
                )

        results.append({
            "max_frequency": max_freq,
            "tokens_per_s1": max_tokens,
            "token_block_recall": (
                len(captured) / total_true_s2
            ),
            "union_recall": (
                len(combined) / total_true_s2
            ),
            "newly_recovered": newly_recovered,
            "estimated_candidate_upper_bound":
                estimated_volume
        })


token_optimization_df = pd.DataFrame(
    results
)

token_optimization_df["token_block_recall_percent"] = (
    token_optimization_df["token_block_recall"] * 100
)

token_optimization_df["union_recall_percent"] = (
    token_optimization_df["union_recall"] * 100
)

print(
    token_optimization_df[
        [
            "max_frequency",
            "tokens_per_s1",
            "token_block_recall_percent",
            "union_recall_percent",
            "newly_recovered",
            "estimated_candidate_upper_bound"
        ]
    ]
    .sort_values(
        [
            "union_recall_percent",
            "estimated_candidate_upper_bound"
        ],
        ascending=[False, True]
    )
    .to_string(index=False)
)


# In[110]:


# ============================================
# STEP 34H: ANALYZE MISSED TRUE S2 PAIRS
# ============================================

missed_s2 = [
    pair
    for pair in true_pair_keys
    if pair[1].startswith("S2-")
    and pair not in (
        union_s2
    )
]

print(
    "Still-missed S2 true pairs:",
    len(missed_s2)
)

missed_df = pd.DataFrame(
    missed_s2,
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)

# Attach names
missed_df = missed_df.merge(
    train_s1[
        [
            "entity_id",
            "business_name"
        ]
    ].rename(
        columns={
            "entity_id":
                "source1_entity_id",
            "business_name":
                "business_name_s1"
        }
    ),
    on="source1_entity_id",
    how="left"
)

missed_df = missed_df.merge(
    train_s2[
        [
            "entity_id",
            "business_name"
        ]
    ].rename(
        columns={
            "entity_id":
                "candidate_entity_id",
            "business_name":
                "business_name_candidate"
        }
    ),
    on="candidate_entity_id",
    how="left"
)

print(
    missed_df.head(50).to_string(
        index=False
    )
)


# In[111]:


# ============================================
# STEP 34I: MISSED-PAIR TOKEN ANALYSIS
# ============================================

def useful_token_overlap(name1, name2):

    t1 = (
        set(get_name_tokens(name1))
        |
        set(get_translit_tokens(name1))
    )

    t2 = (
        set(get_name_tokens(name2))
        |
        set(get_translit_tokens(name2))
    )

    return t1 & t2


missed_df["shared_tokens"] = [
    useful_token_overlap(
        a,
        b
    )
    for a, b in zip(
        missed_df["business_name_s1"],
        missed_df["business_name_candidate"]
    )
]

missed_df["shared_token_count"] = (
    missed_df["shared_tokens"]
    .apply(len)
)

print(
    missed_df[
        [
            "business_name_s1",
            "business_name_candidate",
            "shared_tokens",
            "shared_token_count"
        ]
    ]
    .head(50)
    .to_string(index=False)
)

print("\nShared-token statistics:")

print(
    missed_df["shared_token_count"]
    .value_counts()
    .sort_index()
)


# In[112]:


# ============================================
# STEP 35A: ROBUST NAME NORMALIZATION
# ============================================

LEET_MAP = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g"
})


def normalize_name_robust(text):
    """
    Candidate-generation normalization.

    Handles:
    - accents
    - punctuation
    - common digit/letter substitutions
    - domain-like names
    - repeated spaces
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Transliterate non-Latin scripts
    text = unidecode(text)

    # Common OCR / leetspeak corrections
    text = text.translate(LEET_MAP)

    # Remove common URL/domain suffixes
    text = re.sub(
        r"\bwww\b",
        " ",
        text
    )

    text = re.sub(
        r"\b(?:com|net|org|in|co|biz)\b",
        " ",
        text
    )

    # Keep only alphanumeric characters
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove generic legal terms
    tokens = text.split()

    tokens = [
        token
        for token in tokens
        if token not in GENERIC_NAME_TOKENS
        and len(token) >= 2
    ]

    return " ".join(tokens)


def make_char_trigrams(text):
    """
    Character trigram representation.
    """

    name = normalize_name_robust(text)

    # Remove spaces for typo-tolerant character matching
    name = name.replace(" ", "")

    if len(name) < 3:
        return set()

    return {
        name[i:i+3]
        for i in range(len(name) - 2)
    }


# Test examples

examples = [
    "Kritavi Retail Ltd",
    "Kritavi Rteail Ltd",
    "Cornerstone Insurance Corporation",
    "C0rnerstone Insurance Corporation",
    "Indian Agro LLP",
    "इंडियन एग्रो एलएलपी",
    "Complete Express Opportunities Inc",
    "completeexpressopportunities.com"
]

for x in examples:
    print(
        x,
        "→",
        normalize_name_robust(x)
    )


# In[113]:


# ============================================
# STEP 35C: TEST CHARACTER THRESHOLDS
# ============================================

char_thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

char_results = []

for threshold in char_thresholds:

    captured = s2_true_eval[
        s2_true_eval["char_jaccard"]
        >= threshold
    ]

    captured_pairs = set(
        zip(
            captured["source1_entity_id"],
            captured["candidate_entity_id"]
        )
    )

    # Current blocking union
    combined = (
        union_s2
        |
        captured_pairs
    )

    char_results.append({
        "char_threshold": threshold,
        "char_only_recall": (
            len(captured_pairs)
            / total_true_s2
        ),
        "newly_recovered": (
            len(
                captured_pairs
                - union_s2
            )
        ),
        "new_union_recall": (
            len(combined)
            / total_true_s2
        )
    })


char_results_df = pd.DataFrame(
    char_results
)

char_results_df["char_only_recall_percent"] = (
    char_results_df["char_only_recall"]
    * 100
)

char_results_df["new_union_recall_percent"] = (
    char_results_df["new_union_recall"]
    * 100
)

print(
    char_results_df.to_string(
        index=False
    )
)


# In[114]:


# ============================================
# STEP 35B + 35C — CHARACTER TRIGRAM TEST
# Self-contained version
# ============================================

# Rebuild the S2 true-pair evaluation table
s2_true_eval = eval_matches[
    eval_matches["candidate_entity_id"]
    .astype(str)
    .str.startswith("S2-")
].copy()

print("True S2 pairs:", len(s2_true_eval))


# --------------------------------------------
# Attach S1 names
# --------------------------------------------

s2_true_eval = s2_true_eval.merge(
    eval_s1[
        [
            "entity_id",
            "business_name"
        ]
    ].rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "business_name_s1"
        }
    ),
    on="source1_entity_id",
    how="left"
)


# --------------------------------------------
# Attach candidate names
# --------------------------------------------

s2_true_eval = s2_true_eval.merge(
    train_s2[
        [
            "entity_id",
            "business_name"
        ]
    ].rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "business_name_candidate"
        }
    ),
    on="candidate_entity_id",
    how="left"
)


# --------------------------------------------
# Character trigram similarity
# --------------------------------------------

s2_true_eval["s1_char_trigrams"] = (
    s2_true_eval["business_name_s1"]
    .apply(make_char_trigrams)
)

s2_true_eval["candidate_char_trigrams"] = (
    s2_true_eval["business_name_candidate"]
    .apply(make_char_trigrams)
)


def trigram_jaccard(set_a, set_b):

    if not set_a or not set_b:
        return 0.0

    intersection = len(
        set_a & set_b
    )

    union = len(
        set_a | set_b
    )

    if union == 0:
        return 0.0

    return intersection / union


s2_true_eval["char_jaccard"] = [
    trigram_jaccard(a, b)
    for a, b in zip(
        s2_true_eval["s1_char_trigrams"],
        s2_true_eval["candidate_char_trigrams"]
    )
]


# --------------------------------------------
# Check that the column exists
# --------------------------------------------

print("\nchar_jaccard created:",
      "char_jaccard" in s2_true_eval.columns)

print("\nCharacter similarity statistics:")
print(
    s2_true_eval["char_jaccard"].describe()
)


# ============================================
# Test thresholds
# ============================================

char_thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

char_results = []

for threshold in char_thresholds:

    captured = s2_true_eval[
        s2_true_eval["char_jaccard"]
        >= threshold
    ]

    captured_pairs = set(
        zip(
            captured["source1_entity_id"],
            captured["candidate_entity_id"]
        )
    )

    # Add to our existing candidate union
    combined = (
        union_s2
        |
        captured_pairs
    )

    char_results.append({
        "char_threshold": threshold,

        "char_only_recall": (
            len(captured_pairs)
            / total_true_s2
        ),

        "newly_recovered": (
            len(
                captured_pairs
                - union_s2
            )
        ),

        "new_union_recall": (
            len(combined)
            / total_true_s2
        )
    })


char_results_df = pd.DataFrame(
    char_results
)

char_results_df["char_only_recall_percent"] = (
    char_results_df["char_only_recall"] * 100
)

char_results_df["new_union_recall_percent"] = (
    char_results_df["new_union_recall"] * 100
)


print("\nCharacter-block results:")
print(
    char_results_df[
        [
            "char_threshold",
            "char_only_recall_percent",
            "newly_recovered",
            "new_union_recall_percent"
        ]
    ].to_string(index=False)
)


# In[115]:


# ============================================
# STEP 35D: CHARACTER-PREFIX BLOCKING
# ============================================

def make_char_prefixes(text, prefix_len=3):
    """
    Create character-prefix blocking keys from
    useful transliterated business-name tokens.
    """

    normalized = normalize_name_robust(text)

    tokens = normalized.split()

    prefixes = set()

    for token in tokens:

        if len(token) >= prefix_len:

            prefixes.add(
                token[:prefix_len]
            )

    return prefixes


# --------------------------------------------
# Test examples
# --------------------------------------------

examples = [
    "Kritavi Retail Ltd",
    "Kritavi Rteail Ltd",
    "Cornerstone Insurance Corporation",
    "C0rnerstone Insurance Corporation",
    "Indian Agro LLP",
    "इंडियन एग्रो एलएलपी"
]

for name in examples:

    print(
        name,
        "→",
        make_char_prefixes(name)
    )


# In[116]:


# ============================================
# STEP 35E: PREFIX BLOCK RECALL
# ============================================

# S1 prefix sets
s2_true_eval["s1_prefixes"] = (
    s2_true_eval["business_name_s1"]
    .apply(make_char_prefixes)
)

# Candidate prefix sets
s2_true_eval["candidate_prefixes"] = (
    s2_true_eval["business_name_candidate"]
    .apply(make_char_prefixes)
)


# --------------------------------------------
# Count shared prefixes
# --------------------------------------------

s2_true_eval["shared_prefixes"] = [
    a & b
    for a, b in zip(
        s2_true_eval["s1_prefixes"],
        s2_true_eval["candidate_prefixes"]
    )
]

s2_true_eval["shared_prefix_count"] = (
    s2_true_eval["shared_prefixes"]
    .apply(len)
)


print(
    "Shared-prefix statistics:"
)

print(
    s2_true_eval[
        "shared_prefix_count"
    ].describe()
)


# --------------------------------------------
# Capture if at least one prefix is shared
# --------------------------------------------

prefix_captured = s2_true_eval[
    s2_true_eval["shared_prefix_count"] >= 1
]

prefix_pairs = set(
    zip(
        prefix_captured["source1_entity_id"],
        prefix_captured["candidate_entity_id"]
    )
)

prefix_union = (
    union_s2
    |
    prefix_pairs
)


print("\nPrefix-block results:")

print(
    "Prefix-only recall:",
    f"{len(prefix_pairs) / total_true_s2:.4%}"
)

print(
    "Newly recovered:",
    len(
        prefix_pairs - union_s2
    )
)

print(
    "New union recall:",
    f"{len(prefix_union) / total_true_s2:.4%}"
)

print(
    "Still missed:",
    total_true_s2 - len(prefix_union)
)


# In[117]:


# ============================================
# STEP 35F: TWO-PREFIX VERSION
# ============================================

prefix2_captured = s2_true_eval[
    s2_true_eval["shared_prefix_count"] >= 2
]

prefix2_pairs = set(
    zip(
        prefix2_captured["source1_entity_id"],
        prefix2_captured["candidate_entity_id"]
    )
)

prefix2_union = (
    union_s2
    |
    prefix2_pairs
)

print(
    "Two-prefix recall:",
    f"{len(prefix2_pairs) / total_true_s2:.4%}"
)

print(
    "Newly recovered:",
    len(
        prefix2_pairs - union_s2
    )
)

print(
    "New union recall:",
    f"{len(prefix2_union) / total_true_s2:.4%}"
)

print(
    "Still missed:",
    total_true_s2 - len(prefix2_union)
)


# In[118]:


# ============================================
# STEP 36: SINGLE-PREFIX RESULTS
# ============================================

print(
    "Single-prefix captured:",
    len(prefix_pairs)
)

print(
    "Single-prefix recall:",
    f"{len(prefix_pairs) / total_true_s2:.4%}"
)

prefix_union = union_s2 | prefix_pairs

print(
    "Previous union:",
    len(union_s2)
)

print(
    "Newly recovered:",
    len(prefix_pairs - union_s2)
)

print(
    "New union:",
    len(prefix_union)
)

print(
    "New union recall:",
    f"{len(prefix_union) / total_true_s2:.4%}"
)

print(
    "Still missed:",
    total_true_s2 - len(prefix_union)
)


# In[119]:


# ============================================
# STEP 37: ANALYZE REMAINING MISSED S2 PAIRS
# ============================================

# Current best S2 candidate union
current_s2_union = prefix_union.copy()

# All true S2 pairs
all_true_s2_pairs = {
    pair
    for pair in true_pair_keys
    if pair[1].startswith("S2-")
}

# Remaining missed pairs
remaining_missed_s2 = (
    all_true_s2_pairs
    - current_s2_union
)

print(
    "Remaining missed S2 true pairs:",
    len(remaining_missed_s2)
)


# --------------------------------------------
# Rebuild missed-pair dataframe
# --------------------------------------------

remaining_missed_df = s2_true_eval[
    s2_true_eval.apply(
        lambda row:
        (
            row["source1_entity_id"],
            row["candidate_entity_id"]
        ) in remaining_missed_s2,
        axis=1
    )
].copy()


# --------------------------------------------
# Robust compact names
# --------------------------------------------

remaining_missed_df["s1_compact"] = (
    remaining_missed_df["business_name_s1"]
    .apply(
        lambda x:
        normalize_name_robust(x).replace(" ", "")
    )
)

remaining_missed_df["candidate_compact"] = (
    remaining_missed_df["business_name_candidate"]
    .apply(
        lambda x:
        normalize_name_robust(x).replace(" ", "")
    )
)

remaining_missed_df["compact_exact"] = (
    remaining_missed_df["s1_compact"]
    ==
    remaining_missed_df["candidate_compact"]
)


# --------------------------------------------
# Character similarity
# --------------------------------------------

remaining_missed_df["char_jaccard"] = [
    trigram_jaccard(
        make_char_trigrams(a),
        make_char_trigrams(b)
    )
    for a, b in zip(
        remaining_missed_df["business_name_s1"],
        remaining_missed_df["business_name_candidate"]
    )
]


# --------------------------------------------
# Useful-token overlap
# --------------------------------------------

remaining_missed_df["shared_tokens"] = [
    useful_token_overlap(a, b)
    for a, b in zip(
        remaining_missed_df["business_name_s1"],
        remaining_missed_df["business_name_candidate"]
    )
]

remaining_missed_df["shared_token_count"] = (
    remaining_missed_df["shared_tokens"]
    .apply(len)
)


# ============================================
# Results
# ============================================

print("\nCompact-name exact matches:")
print(
    remaining_missed_df[
        "compact_exact"
    ].value_counts()
)

print("\nChar Jaccard >= thresholds:")

for threshold in [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]:

    count = (
        remaining_missed_df[
            remaining_missed_df["char_jaccard"]
            >= threshold
        ]
        .shape[0]
    )

    print(
        f">= {threshold:.2f}: {count}"
    )


print("\nShared useful-token counts:")
print(
    remaining_missed_df[
        "shared_token_count"
    ]
    .value_counts()
    .sort_index()
)


print("\nFirst 50 remaining misses:")

print(
    remaining_missed_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_translit_ratio",
            "address_ratio",
            "compact_exact",
            "char_jaccard",
            "shared_tokens"
        ]
    ]
    .head(50)
    .to_string(index=False)
)


# In[120]:


# ============================================
# STEP 37 — ANALYZE REMAINING MISSED S2 PAIRS
# FIXED VERSION
# ============================================

# --------------------------------------------
# Current best candidate union
# --------------------------------------------

current_s2_union = prefix_union.copy()

all_true_s2_pairs = {
    pair
    for pair in true_pair_keys
    if pair[1].startswith("S2-")
}

remaining_missed_s2 = (
    all_true_s2_pairs
    - current_s2_union
)

print(
    "Remaining missed S2 true pairs:",
    len(remaining_missed_s2)
)


# --------------------------------------------
# Build DataFrame of ONLY the missed pairs
# --------------------------------------------

remaining_missed_df = pd.DataFrame(
    list(remaining_missed_s2),
    columns=[
        "source1_entity_id",
        "candidate_entity_id"
    ]
)


# --------------------------------------------
# Attach S1 names
# --------------------------------------------

remaining_missed_df = remaining_missed_df.merge(
    train_s1[
        [
            "entity_id",
            "business_name",
            "business_address"
        ]
    ].rename(
        columns={
            "entity_id":
                "source1_entity_id",
            "business_name":
                "business_name_s1",
            "business_address":
                "business_address_s1"
        }
    ),
    on="source1_entity_id",
    how="left"
)


# --------------------------------------------
# Attach S2 names
# --------------------------------------------

remaining_missed_df = remaining_missed_df.merge(
    train_s2[
        [
            "entity_id",
            "business_name",
            "business_address"
        ]
    ].rename(
        columns={
            "entity_id":
                "candidate_entity_id",
            "business_name":
                "business_name_candidate",
            "business_address":
                "business_address_candidate"
        }
    ),
    on="candidate_entity_id",
    how="left"
)


# ============================================
# Recalculate similarity features
# ============================================

remaining_missed_df["name_norm_s1"] = (
    remaining_missed_df["business_name_s1"]
    .apply(normalize_name)
)

remaining_missed_df["name_norm_candidate"] = (
    remaining_missed_df["business_name_candidate"]
    .apply(normalize_name)
)

remaining_missed_df["name_translit_s1"] = (
    remaining_missed_df["business_name_s1"]
    .apply(transliterate_name)
)

remaining_missed_df["name_translit_candidate"] = (
    remaining_missed_df["business_name_candidate"]
    .apply(transliterate_name)
)

remaining_missed_df["address_norm_s1"] = (
    remaining_missed_df["business_address_s1"]
    .apply(normalize_address)
)

remaining_missed_df["address_norm_candidate"] = (
    remaining_missed_df["business_address_candidate"]
    .apply(normalize_address)
)


# --------------------------------------------
# Fuzzy similarity
# --------------------------------------------

remaining_missed_df["name_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        remaining_missed_df["name_norm_s1"],
        remaining_missed_df["name_norm_candidate"]
    )
]

remaining_missed_df["name_translit_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        remaining_missed_df["name_translit_s1"],
        remaining_missed_df["name_translit_candidate"]
    )
]

remaining_missed_df["address_ratio"] = [
    ratio(a, b) / 100
    for a, b in zip(
        remaining_missed_df["address_norm_s1"],
        remaining_missed_df["address_norm_candidate"]
    )
]


# --------------------------------------------
# Compact normalized names
# --------------------------------------------

remaining_missed_df["s1_compact"] = (
    remaining_missed_df["business_name_s1"]
    .apply(
        lambda x:
        normalize_name_robust(x).replace(" ", "")
    )
)

remaining_missed_df["candidate_compact"] = (
    remaining_missed_df["business_name_candidate"]
    .apply(
        lambda x:
        normalize_name_robust(x).replace(" ", "")
    )
)

remaining_missed_df["compact_exact"] = (
    remaining_missed_df["s1_compact"]
    ==
    remaining_missed_df["candidate_compact"]
)


# --------------------------------------------
# Character similarity
# --------------------------------------------

remaining_missed_df["char_jaccard"] = [
    trigram_jaccard(
        make_char_trigrams(a),
        make_char_trigrams(b)
    )
    for a, b in zip(
        remaining_missed_df["business_name_s1"],
        remaining_missed_df["business_name_candidate"]
    )
]


# --------------------------------------------
# Shared useful tokens
# --------------------------------------------

remaining_missed_df["shared_tokens"] = [
    useful_token_overlap(
        a,
        b
    )
    for a, b in zip(
        remaining_missed_df["business_name_s1"],
        remaining_missed_df["business_name_candidate"]
    )
]

remaining_missed_df["shared_token_count"] = (
    remaining_missed_df["shared_tokens"]
    .apply(len)
)


# ============================================
# RESULTS
# ============================================

print("\nCompact-name exact matches:")
print(
    remaining_missed_df[
        "compact_exact"
    ].value_counts()
)


print("\nCharacter Jaccard >= thresholds:")

for threshold in [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]:

    count = (
        remaining_missed_df[
            remaining_missed_df["char_jaccard"]
            >= threshold
        ]
        .shape[0]
    )

    print(
        f">= {threshold:.2f}: {count}"
    )


print("\nShared useful-token counts:")
print(
    remaining_missed_df[
        "shared_token_count"
    ]
    .value_counts()
    .sort_index()
)


print("\nFirst 50 remaining misses:")

print(
    remaining_missed_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "business_name_s1",
            "business_name_candidate",
            "name_ratio",
            "name_translit_ratio",
            "address_ratio",
            "compact_exact",
            "char_jaccard",
            "shared_tokens"
        ]
    ]
    .head(50)
    .to_string(index=False)
)


# In[122]:


# ============================================
# STEP 38A: ADDRESS-ANCHOR ANALYSIS
# ============================================

ADDRESS_STOPWORDS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "lane",
    "ln",
    "drive",
    "dr",
    "boulevard",
    "blvd",
    "place",
    "pl",
    "floor",
    "fl",
    "unit",
    "apt",
    "building",
    "bldg",
    "near",
    "opp",
    "opposite",
    "city",
    "district",
    "county",
    "state",
    "india",
    "usa",
    "us",
    "the",
    "no",
    "null",
    "po",
    "p",
    "o"
}


def address_tokens(text):
    if pd.isna(text):
        return set()

    normalized = normalize_address(text)

    tokens = normalized.split()

    return {
        token
        for token in tokens
        if len(token) >= 3
        and token not in ADDRESS_STOPWORDS
        and not token.isdigit()
    }


def address_numbers(text):
    return extract_numbers(text)


remaining_missed_df["s1_address_tokens"] = (
    remaining_missed_df["business_address_s1"]
    .apply(address_tokens)
)

remaining_missed_df["candidate_address_tokens"] = (
    remaining_missed_df["business_address_candidate"]
    .apply(address_tokens)
)

remaining_missed_df["shared_address_tokens"] = [
    a & b
    for a, b in zip(
        remaining_missed_df["s1_address_tokens"],
        remaining_missed_df["candidate_address_tokens"]
    )
]

remaining_missed_df["shared_address_token_count"] = (
    remaining_missed_df["shared_address_tokens"]
    .apply(len)
)


remaining_missed_df["s1_address_numbers"] = (
    remaining_missed_df["business_address_s1"]
    .apply(address_numbers)
)

remaining_missed_df["candidate_address_numbers"] = (
    remaining_missed_df["business_address_candidate"]
    .apply(address_numbers)
)

remaining_missed_df["shared_address_numbers"] = [
    a & b
    for a, b in zip(
        remaining_missed_df["s1_address_numbers"],
        remaining_missed_df["candidate_address_numbers"]
    )
]

remaining_missed_df["shared_address_number_count"] = (
    remaining_missed_df["shared_address_numbers"]
    .apply(len)
)


print("Shared address-token counts:")
print(
    remaining_missed_df[
        "shared_address_token_count"
    ]
    .value_counts()
    .sort_index()
)

print("\nShared address-number counts:")
print(
    remaining_missed_df[
        "shared_address_number_count"
    ]
    .value_counts()
    .sort_index()
)


# In[123]:


# ============================================
# STEP 38B: ADDRESS-ANCHOR RECALL
# ============================================

for min_shared_tokens in [1, 2, 3]:

    captured = remaining_missed_df[
        remaining_missed_df[
            "shared_address_token_count"
        ] >= min_shared_tokens
    ]

    print(
        f"Shared address tokens >= "
        f"{min_shared_tokens}: "
        f"{len(captured)}"
    )


print("\n")


for min_shared_numbers in [1, 2]:

    captured = remaining_missed_df[
        remaining_missed_df[
            "shared_address_number_count"
        ] >= min_shared_numbers
    ]

    print(
        f"Shared address numbers >= "
        f"{min_shared_numbers}: "
        f"{len(captured)}"
    )


# In[124]:


# ============================================
# STEP 39A: S2 ADDRESS TOKEN FREQUENCY
# ============================================

from collections import Counter

s2_address_token_frequency = Counter()

chunk_size = 500_000

for start in range(
    0,
    len(train_s2),
    chunk_size
):

    end = min(
        start + chunk_size,
        len(train_s2)
    )

    chunk_addresses = train_s2.iloc[
        start:end
    ]["business_address"]

    for address in chunk_addresses:

        tokens = address_tokens(address)

        # Count each token only once per address
        # so repeated words don't artificially increase frequency
        for token in tokens:
            s2_address_token_frequency[token] += 1

    print(
        f"Processed {end:,} / "
        f"{len(train_s2):,}"
    )


print(
    "\nUnique address tokens:",
    len(s2_address_token_frequency)
)

print(
    "\nMost common address tokens:"
)

print(
    s2_address_token_frequency
    .most_common(40)
)


# In[125]:


# ============================================
# STEP 40: FREQUENCY OF SHARED TOKENS
# ============================================

def shared_token_frequencies(row):

    shared = row["shared_address_tokens"]

    return sorted(
        [
            s2_address_token_frequency.get(
                token,
                10**12
            )
            for token in shared
        ]
    )


remaining_missed_df[
    "shared_token_frequencies"
] = remaining_missed_df.apply(
    shared_token_frequencies,
    axis=1
)

# Frequency of the rarest shared address token
remaining_missed_df[
    "rarest_shared_address_token_frequency"
] = remaining_missed_df[
    "shared_token_frequencies"
].apply(
    lambda x:
    min(x) if x else 10**12
)

print(
    remaining_missed_df[
        "rarest_shared_address_token_frequency"
    ].describe()
)


# In[126]:


# ============================================
# STEP 40B: RARE ADDRESS-TOKEN COVERAGE
# ============================================

frequency_limits = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2000,
    5000
]

for limit in frequency_limits:

    captured = remaining_missed_df[
        remaining_missed_df[
            "rarest_shared_address_token_frequency"
        ] <= limit
    ]

    print(
        f"Frequency <= {limit:5d}: "
        f"{len(captured):4d} / "
        f"{len(remaining_missed_df)}"
    )


# In[127]:


# ============================================
# STEP 41: ESTIMATE ADDRESS-BLOCK VOLUME
# ============================================

address_frequency_limits = [
    100,
    250,
    500,
    1000,
    2000
]

address_volume_results = []

for limit in address_frequency_limits:

    total_upper_bound = 0

    for _, row in eval_s1.iterrows():

        tokens = address_tokens(
            row["business_address"]
        )

        rare_tokens = [
            token
            for token in tokens
            if s2_address_token_frequency.get(
                token,
                10**12
            ) <= limit
        ]

        # Use the rarest few anchors
        rare_tokens = sorted(
            rare_tokens,
            key=lambda token:
            s2_address_token_frequency.get(
                token,
                10**12
            )
        )[:2]

        for token in rare_tokens:

            total_upper_bound += (
                s2_address_token_frequency.get(
                    token,
                    0
                )
            )

    address_volume_results.append({
        "frequency_limit": limit,
        "estimated_upper_bound":
            total_upper_bound
    })


address_volume_df = pd.DataFrame(
    address_volume_results
)

print(
    address_volume_df.to_string(
        index=False
    )
)


# In[128]:


# ============================================
# STEP 42: TEST COMBINED ADDRESS ANCHORS
# ============================================

# We are working ONLY on the 1,150 currently
# missed true S2 pairs.

df = remaining_missed_df.copy()


# --------------------------------------------
# Rule A:
# at least 1 shared address token
# --------------------------------------------

rule_a = (
    df["shared_address_token_count"] >= 1
)


# --------------------------------------------
# Rule B:
# at least 2 shared address tokens
# --------------------------------------------

rule_b = (
    df["shared_address_token_count"] >= 2
)


# --------------------------------------------
# Rule C:
# at least 1 shared address number
# --------------------------------------------

rule_c = (
    df["shared_address_number_count"] >= 1
)


# --------------------------------------------
# Rule D:
# at least 1 shared number
# AND
# at least 1 shared address token
# --------------------------------------------

rule_d = (
    (df["shared_address_number_count"] >= 1)
    &
    (df["shared_address_token_count"] >= 1)
)


# --------------------------------------------
# Rule E:
# at least 1 shared number
# AND
# rare shared address token
# --------------------------------------------

rule_e = (
    (df["shared_address_number_count"] >= 1)
    &
    (
        df[
            "rarest_shared_address_token_frequency"
        ] <= 500
    )
)


# --------------------------------------------
# Rule F:
# at least 1 shared number
# AND
# very rare shared address token
# --------------------------------------------

rule_f = (
    (df["shared_address_number_count"] >= 1)
    &
    (
        df[
            "rarest_shared_address_token_frequency"
        ] <= 100
    )
)


rules = {
    "A_token_any": rule_a,
    "B_tokens_2plus": rule_b,
    "C_number_any": rule_c,
    "D_number_AND_token": rule_d,
    "E_number_AND_token_<=500": rule_e,
    "F_number_AND_token_<=100": rule_f
}


print("Remaining missed pairs:", len(df))
print()

for name, mask in rules.items():

    captured = int(mask.sum())

    new_total = (
        len(current_s2_union)
        + captured
    )

    # Important:
    # Some captured pairs may theoretically already
    # belong to the current union, although this df
    # is built from remaining misses, so here they
    # are all new.

    union_recall = (
        new_total / total_true_s2
    )

    print(
        f"{name:28s} "
        f"captures={captured:4d} "
        f"overall_recall={union_recall:.4%} "
        f"still_missed={len(df)-captured:4d}"
    )


# In[129]:


# ============================================
# STEP 43A: ADDRESS NUMBER + TOKEN KEYS
# ============================================

def make_address_anchor_keys(
    address,
    country
):
    """
    Create exact address anchors using:
        country + address number + address token

    A candidate matches when it shares at least
    one anchor with the S1 address.
    """

    if pd.isna(address):
        return set()

    numbers = extract_numbers(address)
    tokens = address_tokens(address)

    if not numbers or not tokens:
        return set()

    country_value = (
        ""
        if pd.isna(country)
        else str(country).lower().strip()
    )

    keys = set()

    for number in numbers:

        for token in tokens:

            keys.add(
                country_value
                + "||"
                + number
                + "||"
                + token
            )

    return keys


eval_s1["address_anchor_keys"] = [
    make_address_anchor_keys(
        address,
        country
    )
    for address, country in zip(
        eval_s1["business_address"],
        eval_s1["country"]
    )
]

print(
    eval_s1[
        [
            "business_name",
            "business_address",
            "address_anchor_keys"
        ]
    ].head(20).to_string(index=False)
)


# In[130]:


# ============================================
# STEP 43B: S1 ANCHOR LOOKUP
# ============================================

address_anchor_to_s1 = {}

for _, row in eval_s1.iterrows():

    s1_id = row["entity_id"]

    for key in row["address_anchor_keys"]:

        if key not in address_anchor_to_s1:
            address_anchor_to_s1[key] = set()

        address_anchor_to_s1[key].add(
            s1_id
        )

print(
    "Unique S1 address anchors:",
    len(address_anchor_to_s1)
)


# In[131]:


# ============================================
# STEP 43C: TEST ADDRESS-ANCHOR BLOCK ON S2
# ============================================

captured_address_anchor_s2 = set()

candidate_pair_set_s2 = set()

chunk_size = 500_000

for start in range(
    0,
    len(train_s2),
    chunk_size
):

    end = min(
        start + chunk_size,
        len(train_s2)
    )

    chunk = train_s2.iloc[
        start:end
    ][
        [
            "entity_id",
            "business_address",
            "country"
        ]
    ].copy()

    for _, row in chunk.iterrows():

        candidate_id = row["entity_id"]

        candidate_keys = (
            make_address_anchor_keys(
                row["business_address"],
                row["country"]
            )
        )

        if not candidate_keys:
            continue

        matched_s1_ids = set()

        for key in candidate_keys:

            s1_ids = (
                address_anchor_to_s1
                .get(key, set())
            )

            matched_s1_ids.update(
                s1_ids
            )

        for s1_id in matched_s1_ids:

            if candidate_id == s1_id:
                continue

            pair = (
                s1_id,
                candidate_id
            )

            candidate_pair_set_s2.add(
                pair
            )

            if pair in true_pair_keys:

                captured_address_anchor_s2.add(
                    pair
                )

    print(
        f"Processed {end:,} / "
        f"{len(train_s2):,}"
    )


# ============================================
# RESULTS
# ============================================

address_anchor_recall_s2 = (
    len(captured_address_anchor_s2)
    /
    total_true_s2
)

print("\n==============================")
print("ADDRESS ANCHOR RESULTS")
print("==============================")

print(
    "Candidate pairs generated:",
    f"{len(candidate_pair_set_s2):,}"
)

print(
    "True pairs captured:",
    len(captured_address_anchor_s2)
)

print(
    "Block-only recall:",
    f"{address_anchor_recall_s2:.4%}"
)

combined_address_union_s2 = (
    union_s2
    |
    captured_address_anchor_s2
)

print(
    "Newly recovered beyond current union:",
    len(
        captured_address_anchor_s2
        - union_s2
    )
)

print(
    "Combined recall:",
    f"{len(combined_address_union_s2) / total_true_s2:.4%}"
)


# In[ ]:





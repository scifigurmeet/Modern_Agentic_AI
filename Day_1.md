<div align="center">

# 🧠 Day 1 — Machine Learning & MLOps in Practice
### CodeLucky · 12-Day Programme on Modern AI, Generative AI & Agentic Systems
**Module M1 · 6 Hours · 100% Hands-On · Runs Entirely in Google Colab**

<img src="https://img.shields.io/badge/Module-M1-1d4ed8?style=for-the-badge" alt="Module M1"/>
<img src="https://img.shields.io/badge/Duration-6_hours-0891b2?style=for-the-badge" alt="6 hours"/>
<img src="https://img.shields.io/badge/Level-Beginner_Friendly-16a34a?style=for-the-badge" alt="Beginner Friendly"/>
<img src="https://img.shields.io/badge/Platform-Google_Colab-f9ab00?style=for-the-badge" alt="Google Colab"/>

</div>

---

> [!NOTE]
> **How to use this document.** This is the *only* file you need for Day 1 — everything is here, including where to get the data. We will work entirely inside **Google Colab** (a free Python notebook that runs in your web browser — nothing to install on your computer). Read top to bottom and type each code cell yourself as you go; don't just copy-paste. Every concept is explained from the beginning, so no prior machine-learning experience is needed. By the end of today you will have built, tested, and shared a real working AI model.

---

## 🌱 Before We Begin: What Are We Actually Doing Today?

Imagine a telecom company (like Airtel or Jio). Every month, some customers **leave** and switch to a competitor. This is called **churn**. Losing a customer is expensive, so the company desperately wants to know: *"Which customers are about to leave, so we can offer them a discount before they go?"*

A human can't read 7,000 customer records and guess. But a **machine learning model** can learn the patterns of who tends to leave, and then predict it for new customers. That's exactly what we will build today.

> [!IMPORTANT]
> **What is a "machine learning model"?**
> A model is a program that **learns from examples instead of being explicitly programmed**. We don't write rules like *"if contract is monthly, then churn."* Instead, we show the model thousands of past customers (who we already know stayed or left), and it figures out the patterns on its own. After learning, it can predict the outcome for customers it has never seen.

```mermaid
flowchart LR
    A["📋 Past customers<br/>(we KNOW who left)"] --> B["🧠 Model learns<br/>the patterns"]
    B --> C["🔮 Predict for NEW<br/>customers"]
    C --> D["💰 Business acts<br/>(offer discount)"]
    style A fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    style B fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style C fill:#cffafe,stroke:#0891b2,color:#155e75
    style D fill:#dcfce7,stroke:#16a34a,color:#14532d
```

> [!IMPORTANT]
> **And what is "MLOps"?**
> Building a model in a learning notebook is easy. Making it work reliably *in the real world* — so a colleague can run it next year on a different computer and get the same answer — is the hard part. **MLOps** (Machine Learning + Operations) is the set of good habits that make a model trustworthy, repeatable, and shareable. Today you learn both: the model **and** the habits around it.

---

## 🎯 Day 1 at a Glance

| | |
|---|---|
| 🧩 **What we build** | A model that predicts which telecom customers will leave |
| 🗂️ **Data we use** | "Telco Customer Churn" — a real, free dataset of ~7,000 customers (download steps below) |
| 💻 **Where we work** | 100% inside Google Colab — no installation on your computer |
| ⏱️ **Structure** | 3 sessions × 2 hours |
| 🛠️ **Tools** | Python · pandas · scikit-learn · joblib · Google Drive · GitHub |
| 🏁 **What you take home** | A shareable online project (on GitHub) containing your trained model |

### The Day 1 Journey

```mermaid
flowchart LR
    A["📥 Session 1<br/>Get data &<br/>train a first model"] --> B["⚖️ Session 2<br/>Try 3 models &<br/>judge them fairly"]
    B --> C["📦 Session 3<br/>Save, share online &<br/>prove it works elsewhere"]
    C --> D(["🏆 Outcome<br/>Your model, live<br/>on GitHub"])
    style A fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    style B fill:#cffafe,stroke:#0891b2,color:#155e75
    style C fill:#dcfce7,stroke:#16a34a,color:#14532d
    style D fill:#f3e8ff,stroke:#7c3aed,color:#581c87
```

---

## 📖 The Vocabulary You'll Need (5 min)

Before any code, here are the only terms you must know. Keep this table handy.

| Term | Plain-English meaning |
|---|---|
| **Dataset** | A table of data — rows are examples, columns are details about each example |
| **Row** | One example (here: one customer) |
| **Column / Feature** | One piece of information about each example (e.g. monthly bill, contract type) |
| **Target / Label** | The thing we want to predict (here: did the customer churn — Yes/No?) |
| **Training** | The process where the model studies past examples and learns patterns |
| **Model** | The learned program that can make predictions |
| **Prediction** | The model's guess for a new example |
| **Accuracy** | Out of 100 predictions, how many the model got right |

```mermaid
flowchart TD
    subgraph T["📊 A Dataset is Just a Table"]
        direction TB
        H["customerID · gender · tenure · MonthlyCharges · ... · Churn"]
        R1["7590-VHVEG · Female · 1 · 29.85 · ... · No"]
        R2["3668-QPYBK · Male · 34 · 56.95 · ... · Yes"]
        H --> R1 --> R2
    end
    note["These columns are FEATURES (inputs).<br/>Churn is the TARGET (what we predict)."]
    style note fill:#fef3c7,stroke:#d97706,color:#78350f
```

---

## 🧭 The Big Idea That Runs Through Today (10 min)

> [!IMPORTANT]
> **One sentence to remember all day:**
> **"A model that works in a learning notebook and a model that works in the real world are two different things."**

Most beginners learn only the left-hand column below. The whole point of today is to learn the right-hand column too.

| Question | 📓 The beginner way | 🏭 The professional (MLOps) way |
|---|---|---|
| How do I prepare data? | Manually, step by step, easy to forget a step | Packaged into one reusable object |
| How do I test honestly? | Just look at the accuracy number | Compare against a fair baseline |
| Where does my model live? | Inside memory — it vanishes when I close the notebook | Saved to a file that can be reloaded later |
| Can a colleague run it? | "It works on my computer" | "It works on **any** computer" |

We will deliberately experience the *wrong* way and then fix it — because seeing the mistake is the best way to understand why the right way matters.

---

## 🚀 Getting Started with Google Colab (10 min)

We do everything in **Google Colab**. You only need a Google account (Gmail) and a web browser.

> [!NOTE]
> **What is Google Colab?** It's a free service from Google that gives you a Python notebook running on Google's computers, right inside your browser. There's nothing to install — no Python, no setup. A notebook is made of **cells**: you type code into a cell and press **Shift + Enter** to run it. Output appears right below the cell.

**Open your notebook:**

1. Go to **[https://colab.research.google.com](https://colab.research.google.com)**
2. Click **File → New notebook in Drive** (this saves it automatically to your Google Drive).
3. Rename it (top-left) to something like `Day1_Churn_Model`.

```mermaid
flowchart LR
    G["🌐 Open<br/>colab.research.google.com"] --> N["📓 New notebook"]
    N --> C["⌨️ Type code in a cell"]
    C --> R["▶️ Shift + Enter to run"]
    R --> O["📤 See output below"]
    style G fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    style O fill:#dcfce7,stroke:#16a34a,color:#14532d
```

> [!TIP]
> **Two kinds of commands you'll see in Colab:**
> - Normal **Python** code (e.g. `import pandas`).
> - Lines starting with `!` are **system commands** (e.g. `!pip install ...`). The `!` tells Colab "run this on the underlying computer, not as Python." This is how we install tools and download files.

**First cell — install the exact tool versions we'll use today.** Colab already has most of these, but pinning versions now prevents surprises later (you'll see why in Session 3).

```python
# Run this in your FIRST Colab cell
!pip install pandas==2.2.2 scikit-learn==1.5.1 joblib==1.4.2 matplotlib==3.9.0 -q
print("✅ Tools installed. You're ready to go.")
```

> [!TIP]
> **What did we just install?**
> - **pandas** — handles tables of data (like Excel, but in code)
> - **scikit-learn** — the machine-learning toolkit; builds and trains models
> - **joblib** — saves a trained model to a file
> - **matplotlib** — draws charts
>
> The `-q` just means "quiet" (less clutter in the output). After this runs, you may see a button saying *"Restart runtime"* — if so, click it once, then carry on.

---

<div align="center">

## 🟦 SESSION 1 — Get the Data and Train Your First Model
### (2 hours)

</div>

> **Goal of this session:** Download the real customer dataset, turn the raw messy file into a clean trained model that predicts churn, and understand *every* step of why we do what we do.

### 1.1 Getting the Telco Churn Dataset (15 min)

This is the famous **"Telco Customer Churn"** dataset — about 7,000 real telecom customers, each labelled as having churned (left) or not. It's free and public.

> [!IMPORTANT]
> **Where the data comes from.** The dataset was originally published by IBM as a sample dataset and is now mirrored in many public places. The most reliable, no-login way to get it inside Colab is to download it directly from a public GitHub mirror with one command. Run the cell below — it fetches the file straight into your Colab session.

```python
# Download the Telco Churn CSV directly into Colab (no login needed)
!wget -q https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -O telco_churn.csv

print("✅ Downloaded telco_churn.csv")

# Confirm the file is here
import os
print("File size:", round(os.path.getsize("telco_churn.csv") / 1024, 1), "KB")
```

> [!NOTE]
> **Three ways to get this data (in case the link above ever changes):**
> 1. **One-command download (recommended)** — the `!wget` cell above. Easiest in Colab.
> 2. **Kaggle** — search *"Telco Customer Churn"* on [kaggle.com/datasets](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Download the CSV, then in Colab click the 📁 folder icon on the left → **Upload** → choose the file. Rename it to `telco_churn.csv`.
> 3. **Upload from your computer** — if you already have the CSV, run the cell below and pick it from your machine.
>
> ```python
> # OPTIONAL — only if you want to upload the CSV from your own computer
> from google.colab import files
> uploaded = files.upload()        # a "Choose Files" button appears
> # After uploading, rename it so the rest of the notebook finds it:
> import os
> for name in uploaded:
>     os.rename(name, "telco_churn.csv")
> ```

> [!WARNING]
> **A note about Colab and your files.** Files you download or create in Colab live in a *temporary* space that is wiped when your session ends (after a few hours of inactivity, or when you close it). That's fine for today — and in Session 3 you'll learn how to save your model permanently to Google Drive and GitHub so it never disappears.

### 1.2 Loading the Data (10 min)

> [!NOTE]
> **What is a CSV file?** "CSV" means *Comma-Separated Values*. It's the simplest way to store a table: each line is a row, and commas separate the columns. Excel can open it, and so can Python.

```python
import pandas as pd

# Read the CSV file into a "DataFrame" — pandas's name for a table
df = pd.read_csv("telco_churn.csv")

# How big is it? (rows, columns)
print("Shape:", df.shape)          # (7043, 21) -> 7043 customers, 21 columns

# Peek at the first 5 rows
df.head()
```

> [!TIP]
> `df` is just a variable name (short for *DataFrame*). `df.head()` shows the top few rows so you can eyeball what you're working with. In Colab, the table renders as a neat, scrollable grid right under the cell. Always look at your data before doing anything else.

### 1.3 What Is This Data, Really? Making Sense of It (25 min)

Before we touch any code or model, let's slow down and genuinely *understand* what we're looking at. **A model can only be as smart as your understanding of the data.** If you don't know what the columns mean or why churn happens, you can't tell whether the model is being clever or being fooled.

#### 📖 The story behind the dataset

This is a real, well-known dataset originally published by **IBM** as a teaching sample. It describes a (fictional but realistic) telecom company in the United States that sells **home phone and internet services** to about **7,043 customers**. For each customer, the company recorded everything it knew about them — who they are, what services they bought, how they pay, how long they've stayed — and crucially, **whether they left the company ("churned") last month.**

> [!IMPORTANT]
> **What "churn" means and why the company cares.** Churn = a customer cancelled and left. In the telecom business this is the single most expensive problem there is. Winning a *new* customer costs far more (advertising, discounts, setup) than keeping an existing one. So if the company could spot *who is about to leave* a month in advance, it could phone them, offer a better deal, and save the revenue. **That is the entire business purpose of this dataset, and of the model we build today.**

#### 🧩 What each column means (in plain business language)

The 21 columns aren't random — they fall into **four natural groups**, each answering a different question about the customer. Understanding these groups is how you go from "a wall of columns" to "a customer I can picture."

```mermaid
flowchart TD
    C(("👤 One<br/>Customer"))
    C --> G1["👥 WHO they are<br/>(demographics)"]
    C --> G2["📦 WHAT they bought<br/>(services)"]
    C --> G3["💳 HOW they pay & commit<br/>(account)"]
    C --> G4["🎯 Did they LEAVE?<br/>(the target)"]
    style C fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style G1 fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    style G2 fill:#cffafe,stroke:#0891b2,color:#155e75
    style G3 fill:#fef3c7,stroke:#d97706,color:#78350f
    style G4 fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
```

**👥 Group 1 — Who the customer is (demographics)**

| Column | What it means | Why it might predict churn |
|---|---|---|
| `gender` | Male or Female | Usually has little effect — we'll verify, not assume |
| `SeniorCitizen` | Is the customer 65+? (1 = yes, 0 = no) | Older customers may behave differently online |
| `Partner` | Do they have a spouse/partner? | Family customers tend to be more "sticky" (stay longer) |
| `Dependents` | Do they have children/dependents? | Households with kids switch providers less often |

**📦 Group 2 — What services they bought**

| Column | What it means | Why it might predict churn |
|---|---|---|
| `PhoneService` | Has home phone service | Basic service; baseline relationship |
| `MultipleLines` | Has more than one phone line | More services = more invested in staying |
| `InternetService` | Type of internet: DSL, Fiber optic, or None | **Big one** — fiber is fast but pricier; fiber users churn more |
| `OnlineSecurity` | Has the security add-on | Add-ons make leaving more of a hassle (stickier) |
| `OnlineBackup` | Has the backup add-on | Same — every add-on is a small anchor |
| `DeviceProtection` | Has device protection | Same |
| `TechSupport` | Has premium tech support | Customers with support feel looked-after → stay |
| `StreamingTV` | Streams TV through them | More usage, more entanglement |
| `StreamingMovies` | Streams movies through them | Same |

**💳 Group 3 — How they pay and commit (account info)**

| Column | What it means | Why it might predict churn |
|---|---|---|
| `tenure` | **How many months they've been a customer** | **The strongest signal.** New customers leave; long-time customers rarely do |
| `Contract` | Month-to-month, One year, or Two year | **Huge.** Month-to-month customers can leave anytime; contract customers are locked in |
| `PaperlessBilling` | Uses paperless billing | Slightly correlated with the type of customer who shops around |
| `PaymentMethod` | How they pay (electronic check, mailed check, bank transfer, credit card) | **Electronic-check payers churn a lot more** — often less committed customers |
| `MonthlyCharges` | What they pay per month (₹/$ amount) | Higher bills → more reason to look for a cheaper deal |
| `TotalCharges` | What they've paid in total over their whole time | Roughly tenure × monthly charges; reflects overall relationship value |

**🎯 Group 4 — The answer we're predicting (the target)**

| Column | What it means |
|---|---|
| `Churn` | **Did this customer leave last month?** Yes or No. This is what the model learns to predict. |

> [!NOTE]
> **The `customerID` column.** There's one more column — a unique code like `7590-VHVEG` for each customer. It carries *no pattern* (it's just an ID, like an Aadhaar number), so we'll throw it away before modeling. Keeping it could even fool the model into "memorising" individuals instead of learning real patterns.

#### 🔎 Forming hypotheses *before* we look

Good data work means making **guesses first, then checking them.** Based on the business logic above, here's what we *expect* to see. Write these down — we'll test them in a moment:

> [!TIP]
> **Our hypotheses (educated guesses):**
> 1. 🔴 **New customers churn more.** Someone in month 1 hasn't bonded with the company yet.
> 2. 🔴 **Month-to-month contracts churn far more** than 1- or 2-year contracts (no lock-in).
> 3. 🔴 **Fiber-optic internet customers churn more** (they pay more and have higher expectations).
> 4. 🟢 **Customers with add-ons** (security, tech support) **churn less** (more tangled in, more cared for).
> 5. ⚪ **Gender probably doesn't matter** much.

#### 👀 Now let's actually look — does the data agree?

Let's check our hypotheses with simple summaries. **This is the heart of "making sense" of data** — not running a model, but seeing the patterns with our own eyes.

```python
# HYPOTHESIS 2: Do month-to-month customers churn more?
# Group by contract type and show the % who churned in each group.
churn_by_contract = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
).round(1)
print("Churn rate by contract type (%):")
print(churn_by_contract)
# Expected: Month-to-month ~43%, One year ~11%, Two year ~3%
```

> [!IMPORTANT]
> **Read this result slowly — it's the most important insight in the whole dataset.** Month-to-month customers leave at roughly **43%**, while two-year-contract customers leave at about **3%**. That's a *14× difference*! This single column tells you most of the churn story: **lock-in keeps customers.** When you see the model later lean heavily on `Contract`, you'll understand *why* — you saw it here first, with your own eyes.

```python
# HYPOTHESIS 1: Do newer customers churn more?
# Compare the average tenure (months as a customer) of leavers vs stayers.
print("Average months as a customer:")
print(df.groupby("Churn")["tenure"].mean().round(1))
# Expected: customers who churned have MUCH lower average tenure
```

> [!NOTE]
> You'll see that customers who left had been around for ~18 months on average, while those who stayed averaged ~38 months. **The longer someone stays, the less likely they are to leave** — loyalty compounds. This is why `tenure` will be one of the model's favourite columns.

```python
# HYPOTHESIS 3: Do fiber-optic internet customers churn more?
churn_by_internet = df.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
).round(1)
print("Churn rate by internet type (%):")
print(churn_by_internet)
# Expected: Fiber optic churns highest, "No internet" churns lowest
```

```python
# HYPOTHESIS 5: Does gender actually matter?
churn_by_gender = df.groupby("gender")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
).round(1)
print("Churn rate by gender (%):")
print(churn_by_gender)
# Expected: almost identical for Male and Female — gender barely matters
```

> [!IMPORTANT]
> **Why this exercise matters more than the model.** Notice what just happened: *before any machine learning*, you already understand the customer. You know a worried customer looks like — **new, on a month-to-month plan, paying by electronic check for expensive fiber internet, with no add-ons.** A model that later flags exactly this kind of person is *trustworthy*, because it matches what you discovered by hand. A model that instead flagged, say, "Female customers" would be suspicious — because you checked, and gender doesn't matter. **This is how you tell a smart model from a fooled one: by understanding the data first.**

#### 🖼️ See it in one picture

A chart makes the patterns instant. This shows churn rate across the most important columns.

```python
import matplotlib.pyplot as plt

# Pick the columns we found most interesting
cols_to_plot = ["Contract", "InternetService", "PaymentMethod", "gender"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.flatten(), cols_to_plot):
    rates = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    rates.sort_values().plot(kind="barh", ax=ax, color="#1d4ed8")
    ax.set_title(f"Churn rate by {col}", fontweight="bold")
    ax.set_xlabel("% who churned")

plt.tight_layout()
plt.show()
```

> [!TIP]
> **What the picture tells you at a glance:** the longest bars (most churn) cluster around *month-to-month contracts, fiber-optic internet, and electronic-check payment* — exactly our high-risk profile. The gender bars are nearly equal, confirming it barely matters. **You can now explain churn to a non-technical manager in one sentence, using nothing but this chart.** That is what "making sense of data" means.

---

### 1.4 Inspecting the Data Technically (15 min)

Now that we *understand* the data as a business story, let's inspect it technically and find the hidden defects we'll need to fix.

> [!IMPORTANT]
> **The golden rule:** Never build a model before you understand your data. Most failures in machine learning are actually *data* problems in disguise. We just did the *business* understanding above; now we do the *technical* check.

```python
# 1. What columns exist, and what type is each?
df.info()

# 2. Look at the TARGET — the thing we predict. Is it balanced?
print(df["Churn"].value_counts(normalize=True))
# No     0.73   -> 73% of customers stayed
# Yes    0.27   -> 27% of customers left
```

This 73%/27% split is hugely important. Here's why:

> [!IMPORTANT]
> **The "baseline" — your honesty check.**
> Imagine a lazy model that *always* predicts "No" (the customer will stay) for everyone. Since 73% of customers really do stay, this lazy model is right **73% of the time** — without learning anything!
>
> So **73% is the score to beat.** If our real model scores below 73%, it is worse than doing nothing. Keep `73%` in mind; it's our anchor for the whole day.

Now, a real-world data problem hiding in this dataset:

```python
# The column "TotalCharges" should be a number, but Python sees it as text!
print(df["TotalCharges"].dtype)        # 'object' means text
```

Why? Because 11 brand-new customers have a **blank space** `" "` instead of a number (they haven't been charged yet). One blank space turns the whole column into text. Let's fix it:

```python
# Convert the column to numbers. Blanks that can't convert become "NaN" (missing)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# How many are missing now?
print("Missing values:", df["TotalCharges"].isna().sum())   # 11

# These are new customers, so a sensible value is 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)
```

> [!TIP]
> **What is "NaN"?** It stands for *Not a Number* — Python's way of marking a missing or empty value. Real datasets are full of them, and handling them is a core skill.

### 1.5 Splitting Inputs from the Answer (5 min)

The model needs two things: the **inputs** (features) and the **correct answers** (target) to learn from.

```python
# Remove the customer ID — it's just a random code with no useful pattern
df = df.drop(columns=["customerID"])

# y = the answer we want to predict. Convert "Yes"/"No" to 1/0 (computers prefer numbers)
y = (df["Churn"] == "Yes").astype(int)

# X = everything else (all the input features)
X = df.drop(columns=["Churn"])
```

> [!NOTE]
> **Why `X` and `y`?** This is a universal convention in machine learning. `X` (capital) holds the input features; `y` (lowercase) holds the target answers. You'll see this everywhere.

Now we separate columns by type, because numbers and text need different handling:

```python
# Numeric columns (e.g. tenure, MonthlyCharges)
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Text/category columns (e.g. gender, Contract type)
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numbers:", numeric_features)
print("Categories:", categorical_features)
```

### 1.6 The Most Important Idea of the Day: Splitting the Data (15 min)

> [!IMPORTANT]
> **Why we split data into "train" and "test".**
> If a student sees the exam questions *before* the exam, scoring 100% proves nothing. The same is true for models. We must hide some data from the model during learning, then test it on that hidden data to get an *honest* score.
>
> - **Training set (80%)** -> the model studies this.
> - **Test set (20%)** -> kept secret; used only to grade the model fairly.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,         # keep 20% hidden for testing
    random_state=42,        # makes the split identical every time you run it
    stratify=y              # keep the same 73/27 ratio in both halves
)

print(f"Training on {X_train.shape[0]} customers, testing on {X_test.shape[0]}")
```

> [!NOTE]
> **Let's decode every setting in that command — these appear everywhere in machine learning, so understanding them now pays off all programme:**
>
> **`test_size=0.20`** — "keep 20% for testing." It's a fraction: `0.20` means 20%. If you wrote `0.30`, you'd hide 30% and train on 70%. Why 20%? It's a common balance — enough hidden data to grade fairly, but not so much that the model has too little to learn from.
>
> **`random_state=42`** — this one confuses every beginner, so here's the full story. To split the data, the computer needs to *shuffle* the customers and then cut off the first 80%. Shuffling uses **random numbers**. But "random" on a computer isn't truly random — it follows a recipe that starts from a **seed number**. If you give it the same seed, you get the *exact same shuffle every time.*
>   - `random_state=42` fixes the seed to 42, so **you and your neighbour both get the identical split** — making results comparable and reproducible.
>   - Without it, every run would shuffle differently and your accuracy would wobble each time, making it impossible to tell whether a change you made actually helped.
>   - **Why 42 specifically?** No special reason — it's a famous programmer's in-joke (from *The Hitchhiker's Guide to the Galaxy*, where 42 is "the answer to everything"). You could use `random_state=7` or `random_state=100`; the rule is just *pick one number and keep using it.*
>
> **`stratify=y`** — "keep the same Yes/No mix in both halves." Remember our data is 73% stayed / 27% churned. Without this, a random split might accidentally put most of the churners in the test set, leaving the training set lopsided. `stratify=y` forces *both* the training and test sets to keep that same 73/27 ratio, so each is a fair miniature of the whole.

```mermaid
flowchart LR
    D["🗃️ All Data<br/>7043 customers"] --> S{{"Split<br/>80% / 20%"}}
    S -->|"80% — model studies this"| TR["🏋️ Training Set<br/>5634 customers"]
    S -->|"20% — kept secret"| TE["🔒 Test Set<br/>1409 customers"]
    TR --> M["Model learns"]
    TE --> E["Honest grade"]
    style TR fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    style TE fill:#fef3c7,stroke:#d97706,color:#78350f
```

> [!WARNING]
> **The classic beginner mistake (called "data leakage").** If you clean or scale your data using the *whole* dataset before splitting, secret test information sneaks into training. Your scores look amazing in practice but collapse in the real world. **Rule: split first, then prepare.** Always.

### 1.7 The Pipeline — Packaging All Preparation Into One Object (35 min)

Real data needs cleaning before a model can use it: fill in missing values, convert text categories into numbers, and put numbers on a common scale. Doing this by hand every time is error-prone. Instead we build a **Pipeline** — a single object that remembers all these steps and applies them automatically.

> [!NOTE]
> **What do these preparation steps mean?**
> - **Imputing** = filling in missing values (e.g. replace a blank with the median).
> - **Scaling** = putting numbers on a comparable range, so a column measured in thousands doesn't overpower one measured in single digits.
> - **One-Hot Encoding** = turning a text category into numbers. For example "Contract" with values *Monthly / Yearly / Two-year* becomes three yes/no columns the model can understand.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# --- Recipe for NUMERIC columns: fill missing -> put on common scale ---
numeric_steps = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# --- Recipe for TEXT columns: fill missing -> convert to numbers ---
categorical_steps = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("to_numbers", OneHotEncoder(handle_unknown="ignore")),
])

# --- Apply each recipe to the right columns ---
preprocessor = ColumnTransformer(transformers=[
    ("numbers", numeric_steps, numeric_features),
    ("categories", categorical_steps, categorical_features),
])

# --- The complete model: preparation + the learning algorithm, in ONE object ---
model = Pipeline(steps=[
    ("prepare", preprocessor),
    ("learn", LogisticRegression(max_iter=1000, random_state=42)),
])

# Train the model on the training data only
model.fit(X_train, y_train)
print("Model trained!")
```

> [!IMPORTANT]
> **That was a lot of new tools at once. Let's slow right down and understand each one with a tiny example — because these five tools are the building blocks of almost all machine learning.** Don't move on until each makes sense.

#### 🔧 Tool 1: `SimpleImputer` — fills in the blanks

**The problem it solves:** real data has holes (missing values). A model can't do maths on a blank. `SimpleImputer` fills each blank with a sensible substitute.

Imagine a tiny column of monthly charges where one value is missing:

```
Before:   [50,  70,  missing,  90,  60]
```

`SimpleImputer(strategy="median")` looks at all the *known* values, finds the **median** (the middle value when sorted: 50, 60, 70, 90 → middle is ~65), and fills the gap with it:

```
After:    [50,  70,    65,     90,  60]
```

> [!NOTE]
> **What does `strategy="median"` mean, and why median?**
> - **`strategy`** tells the imputer *what to fill blanks with.* Options include `"mean"` (average), `"median"` (middle value), and `"most_frequent"` (the most common value).
> - **Why `median` for numbers?** The median is *robust* — it isn't thrown off by extreme values. If one customer paid ₹50,00,000 by mistake, the *average* would shoot up and give a silly fill value. The *median* (middle) barely moves, so it stays sensible.
> - **Why `most_frequent` for text?** You can't average words like "DSL" and "Fiber optic." So for text columns we fill blanks with whichever category appears most often — the safest guess.

#### 🔧 Tool 2: `StandardScaler` — puts numbers on a level playing field

**The problem it solves:** our columns are on wildly different scales. `tenure` runs from 0 to ~72 (months), but `TotalCharges` runs from 0 to ~8,700 (rupees). To a model, the *bigger numbers can look more important just because they're bigger* — which is unfair. `StandardScaler` rescales every number column so they're all comparable.

It does this by converting each value into "how many steps above or below the average it is." After scaling, every column is centred around 0:

```
tenure before:   [1,   34,   2,   45,   8]      (small numbers)
charges before:  [29,  1889, 108, 1840, 820]    (big numbers)

After scaling, BOTH become comparable:
tenure after:    [-1.2,  0.5, -1.1,  0.9, -0.7]
charges after:   [-1.0,  0.8, -0.9,  0.8, -0.1]
```

> [!NOTE]
> **Why this matters.** Now the model judges each column on its *pattern*, not its raw size. A change in `tenure` and a change in `TotalCharges` carry equal weight until the model decides otherwise. Some models (like Logistic Regression) genuinely need this to work well; others (like Random Forest) don't care — but scaling never hurts, so we do it for everyone.

#### 🔧 Tool 3: `OneHotEncoder` — turns words into numbers

**The problem it solves:** models do maths, and maths needs numbers — but `Contract` contains words like "Month-to-month." `OneHotEncoder` converts each category into a set of yes/no (1/0) columns.

Take the `Contract` column with three possible values:

```
Before (one text column):
  Contract
  -----------------
  Month-to-month
  Two year
  One year
```

It becomes **three number columns**, one per category, where `1` means "yes, this one" and `0` means "no":

```
After (three number columns):
  Month-to-month   One year   Two year
  --------------   --------   --------
       1              0          0        <- this customer is month-to-month
       0              0          1        <- this customer is two-year
       0              1          0        <- this customer is one-year
```

> [!NOTE]
> **Why "one-hot"?** The name comes from electronics: out of all the new columns, exactly *one* is "hot" (set to 1) for each row, the rest are 0. It's just a vivid way of saying "tick exactly one box."
>
> **Why not just number them 1, 2, 3?** Tempting, but wrong! If we labelled Month-to-month=1, One year=2, Two year=3, the model would think Two year is "3× more" than Month-to-month, or that One year is "between" them — which is meaningless for categories. One-hot encoding avoids inventing fake order. (This is one of the most common beginner mistakes, now you'll never make it.)
>
> **`handle_unknown="ignore"`** — covered earlier: if a brand-new category shows up in real life that the model never trained on, this tells the encoder to quietly set all boxes to 0 instead of crashing.

#### 🔧 Tool 4: `Pipeline` and `ColumnTransformer` — the assembly line

**`Pipeline`** chains steps so they run in order, automatically. `Pipeline([("fill", SimpleImputer(...)), ("scale", StandardScaler())])` means: *"first fill the blanks, then scale — every time, no forgetting."* The `("name", tool)` pairs are just labels so you can refer to each step later; the names ("fill", "scale") are yours to choose.

**`ColumnTransformer`** applies *different* pipelines to *different* columns — because numbers and text need different treatment. We tell it: send number columns through the number recipe, send text columns through the text recipe, then glue the results back together. (You saw this in the diagram above.)

#### 🔧 Tool 5: `LogisticRegression` — the actual learning algorithm

This is the part that *learns the pattern* and makes the prediction. Let's decode its two settings:

```python
LogisticRegression(max_iter=1000, random_state=42)
```

> [!NOTE]
> **`max_iter=1000`** — "try up to 1000 times to find the best answer."
> Logistic Regression learns by *gradually improving* its guess, step by step, like adjusting a recipe by taste until it's right. Each adjustment is one **iteration**. `max_iter` is the maximum number of adjustment steps allowed before it stops.
>   - The default is only 100, which is sometimes *too few* — the model stops before it has fully settled, and you get a warning like *"failed to converge."* Setting `max_iter=1000` gives it plenty of room to finish.
>   - Think of it as "I'll let you keep refining your answer up to 1000 times; if you settle sooner, great, stop early."
>
> **`random_state=42`** — same idea as before. Parts of the learning involve a little randomness, so fixing the seed to 42 means *you get the exact same model every run.* Reproducibility again.

> [!TIP]
> **The pattern you'll see all programme:** almost every tool in scikit-learn takes settings like these in its brackets. They're called **hyperparameters** — knobs *you* set before training (as opposed to the patterns the model *learns* by itself). You don't need to memorise them; you need to know they exist and roughly what they do. When in doubt, the defaults are usually fine — we only override them when there's a reason (like `max_iter` needing more room).

Now that you understand each tool, here's how a customer's data flows through the whole pipeline — every step we just explained, in order:

```mermaid
flowchart TD
    X["📊 A customer's data"] --> CT{{"Pipeline routes<br/>each column"}}
    CT -->|number columns| NUM["Fill missing<br/>-> Scale to common range"]
    CT -->|text columns| CAT["Fill missing<br/>-> Convert to numbers"]
    NUM --> JOIN["🔗 Combine everything"]
    CAT --> JOIN
    JOIN --> ALGO["🤖 Learning algorithm<br/>(Logistic Regression)"]
    ALGO --> OUT["Prediction: Will churn? Yes/No"]
    style CT fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style ALGO fill:#dcfce7,stroke:#16a34a,color:#14532d
```

> [!NOTE]
> **What is "Logistic Regression"?** Despite the scary name, it's one of the simplest and most reliable models for yes/no questions. It looks at all the features and outputs a probability between 0 and 1 (e.g. "82% likely to churn"). We start with it because it's fast and easy to understand.
>
> **Why does `handle_unknown="ignore"` matter?** In the real world, a new customer might have a category the model never saw in training. Without this setting, the program would crash. With it, the model handles the surprise gracefully. This one small choice prevents real production failures.

### 1.8 Grading the Model Honestly (25 min)

A single test score can be lucky. **Cross-validation** is a fairer test: it splits the training data into 5 parts, trains on 4 and tests on the 5th, then rotates — giving 5 scores instead of 1.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

print(f"Average accuracy: {scores.mean():.3f}")
print(f"Variation (+/-):  {scores.std():.3f}")
print(f"Baseline to beat: 0.730")
```

> [!NOTE]
> **Decoding the settings:**
> - **`cv=5`** — "cut the training data into **5** parts (folds)." The model is trained and tested 5 times, each time holding out a different fifth. Why 5? It's the standard choice — enough rounds to be reliable, few enough to run quickly. You could use `cv=10` for a more thorough check at the cost of speed.
> - **`scoring="accuracy"`** — "grade each round on **accuracy**" (the % it got right). Later you could ask for other grades like `"recall"` (catch-rate of leavers) instead — remember from earlier that accuracy isn't always the best grade for imbalanced data.
> - **`scores.mean()`** averages the 5 results; **`scores.std()`** measures how much they wobbled (the `±`). `.3f` just means "show 3 decimal places."

```mermaid
flowchart LR
    T["🏋️ Training Data"] --> F1["Test on Part 1<br/>(train on 2-5)"]
    T --> F2["Test on Part 2<br/>(train on rest)"]
    T --> F3["Test on Part 3"]
    T --> F4["Test on Part 4"]
    T --> F5["Test on Part 5"]
    F1 --> AVG["📊 Average the<br/>5 scores"]
    F2 --> AVG
    F3 --> AVG
    F4 --> AVG
    F5 --> AVG
    style AVG fill:#fef3c7,stroke:#d97706,color:#78350f
```

> [!IMPORTANT]
> **How to read the result.** A score like `0.804 +/- 0.012` means: on average 80.4% correct, and the score barely wobbles between folds (±1.2%) — that's a *stable, trustworthy* model. If instead you saw `0.804 +/- 0.090`, the model is jumpy and unreliable. Always read the variation, not just the average. And remember: anything above our **0.73 baseline** means the model genuinely learned something.

### 1.9 🧪 Try It Yourself (15 min)

> 1. **Explore one column we didn't chart.** Pick `PaymentMethod` or `TechSupport` and compute its churn rate (copy the `groupby` pattern from Section 1.3). Does the result match the business logic in the column tables?
> 2. Run the download cell and load the dataset; fix the `TotalCharges` problem yourself.
> 3. Build the full pipeline by typing it out (don't copy-paste — typing builds memory).
> 4. Print your average accuracy and compare it to the 73% baseline. Did you beat it?
> 5. **Bonus:** Change `strategy="median"` to `strategy="mean"` in the numeric steps. Does the score change much? (It shouldn't — and noticing that is itself a useful insight.)

✅ **Session 1 done:** You have a trained model that prepares messy data automatically and beats the lazy 73% baseline.

---

<div align="center">

## 🟦 SESSION 2 — Try Three Models and Judge Them Fairly
### (2 hours)

</div>

> **Goal of this session:** Swap in three different learning algorithms, compare them honestly, and learn to read *what kind* of mistakes a model makes — not just how often it's right.

### 2.1 The Beauty of the Pipeline: Swapping Models Is One Line (10 min)

Because all the data-preparation is packaged in the pipeline, trying a new algorithm only means changing the final step. Everything else stays identical and fair.

```mermaid
flowchart LR
    PRE["🔧 Same data preparation<br/>for everyone"] --> A["🤖 Logistic Regression"]
    PRE --> B["🌲 Random Forest"]
    PRE --> C["🚀 Gradient Boosting"]
    A --> CMP["⚖️ Compare fairly"]
    B --> CMP
    C --> CMP
    style PRE fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style CMP fill:#fef3c7,stroke:#d97706,color:#78350f
```

### 2.2 Meet the Three Models (15 min)

> [!NOTE]
> **Three different "brains," explained simply:**
> - 🤖 **Logistic Regression** — draws a straight dividing line between "will stay" and "will leave." Simple, fast, surprisingly effective.
> - 🌲 **Random Forest** — asks hundreds of yes/no questions (like a flowchart) and takes a majority vote. More flexible.
> - 🚀 **Gradient Boosting** — builds many small models, each fixing the mistakes of the last. Often the most accurate, but the slowest to train.

### 2.3 Compare All Three (20 min)

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# A helper that wraps any algorithm in our shared preparation pipeline
def build_model(algorithm):
    return Pipeline(steps=[
        ("prepare", preprocessor),
        ("learn", algorithm),
    ])

# The three contenders
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
}

# Test each one fairly with cross-validation
for name, algorithm in candidates.items():
    pipe = build_model(algorithm)
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{name:<22} {scores.mean():.3f} +/- {scores.std():.3f}")
```

> [!NOTE]
> **One new setting here: `n_estimators=200`.** Remember a Random Forest works by asking many yes/no question-trees and taking a majority vote. `n_estimators=200` means *"build 200 trees and let them vote."* More trees usually means a slightly better, steadier result — but slower. 100 (the default) to 200 is a common sweet spot. It's the same idea as asking 200 people for an opinion instead of 5: the crowd's average is more reliable.
>
> Notice `GradientBoostingClassifier` here uses *only* `random_state=42` — we're letting all its other knobs stay at their sensible defaults. You don't have to set every setting; defaults exist for a reason. We only touch a knob when we have a clear purpose.

When you run it, you'll see something like this:

| Model | Accuracy | Character |
|---|---|---|
| 🤖 Logistic Regression | ~0.80 | Simple and fast |
| 🌲 Random Forest | ~0.79 | Flexible, robust |
| 🚀 Gradient Boosting | ~0.80 | Often best, but slowest |

> [!IMPORTANT]
> **A surprising and valuable lesson.** Beginners expect the fanciest model to win by a mile. It usually doesn't! On everyday table data, a *simple* Logistic Regression is often within 1% of the fanciest model, trains far faster, and is easier to explain. **Complexity is a cost you should only pay when it earns its keep.**

### 2.4 Beyond Accuracy: The Confusion Matrix (35 min)

Accuracy alone can mislead. Remember the lazy model that scored 73% by always saying "No"? Accuracy didn't reveal that it never catches a single leaver. The **confusion matrix** shows exactly *what kind* of mistakes happen.

```python
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Train our chosen model and test it on the secret test set
final = build_model(GradientBoostingClassifier(random_state=42))
final.fit(X_train, y_train)
predictions = final.predict(X_test)

# A full report of how well it did
print(classification_report(y_test, predictions, target_names=["Stayed", "Churned"]))

# Draw the confusion matrix (it appears right under the cell in Colab)
matrix = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(matrix, display_labels=["Stayed", "Churned"]).plot(cmap="Blues")
plt.title("Who did the model get right and wrong?")
plt.show()
```

A confusion matrix has four boxes. Here's what each means in plain language:

```mermaid
flowchart TD
    subgraph CM["🔲 The Four Outcomes"]
        TN["✅ Correct<br/>Said 'will stay',<br/>they stayed"]
        FP["⚠️ False Alarm<br/>Said 'will leave',<br/>but they stayed<br/>(wasted a discount)"]
        FN["🚨 Missed!<br/>Said 'will stay',<br/>but they LEFT<br/>(lost a customer)"]
        TP["✅ Correct<br/>Said 'will leave',<br/>they left"]
    end
    style FN fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    style FP fill:#fef3c7,stroke:#d97706,color:#78350f
    style TP fill:#dcfce7,stroke:#16a34a,color:#14532d
    style TN fill:#dcfce7,stroke:#16a34a,color:#14532d
```

> [!IMPORTANT]
> **Turning numbers into business sense.** For this telecom problem, the worst mistake is the 🚨 **"Missed!"** box — predicting a customer will stay when they actually leave. That's a lost customer the company never tried to save. A "False Alarm" only costs a small discount. So we care most about **catching real leavers** — even if it means a few false alarms. This is how a good analyst translates a statistic into a business decision.

```python
from sklearn.metrics import precision_score, recall_score

# "Recall" = of all customers who really left, how many did we catch?
print(f"Catch rate of real leavers (recall): {recall_score(y_test, predictions):.3f}")
# "Precision" = of everyone we flagged, how many really left?
print(f"Accuracy of our alarms (precision):  {precision_score(y_test, predictions):.3f}")
```

### 2.5 🧪 Try It Yourself (15 min)

> 1. Run all three models and write down their scores.
> 2. Draw the confusion matrix for your best model.
> 3. **Think:** does your model make more "False Alarms" or more "Missed!" errors? Write one sentence on what that costs the company.
> 4. **Bonus:** add `class_weight="balanced"` inside `LogisticRegression(...)`. Watch the catch-rate (recall) go up while overall accuracy dips slightly. Discuss: could that actually be the *better* model for this business?

✅ **Session 2 done:** You compared three models fairly and learned to read mistakes as business costs, not just numbers.

---

<div align="center">

## 🟦 SESSION 3 — Save Your Model, Share It Online, and Prove It Works
### (2 hours)

</div>

> **Goal of this session:** A model stuck inside a Colab session is useless to anyone else — and disappears when the session ends. We'll save it to a file, store it permanently in Google Drive, write a small script that uses it, upload the whole project to GitHub, and then reload it in a *brand-new* Colab to prove it truly works anywhere.

### 3.1 Saving the Model to a File (15 min)

Right now your trained model lives only in Colab's temporary memory. We **save** it to a file so it can be reused later without retraining.

> [!IMPORTANT]
> **We save the entire pipeline, not just the algorithm.** Because our pipeline includes all the data-preparation steps, the saved file already knows how to clean, scale, and encode new data. Anyone who loads it gets the full package — they don't need to redo any preparation.

```python
import joblib

# Train the final model on all the training data
final_model = build_model(GradientBoostingClassifier(random_state=42))
final_model.fit(X_train, y_train)

# Save it to a file (.pkl is the standard extension for saved Python objects)
joblib.dump(final_model, "churn_pipeline.pkl")
print("Model saved as churn_pipeline.pkl")
```

```mermaid
flowchart LR
    M["🧠 Trained model<br/>(in memory)"] -->|save| F["📦 churn_pipeline.pkl<br/>(a file)"]
    F -->|load later| M2["🧠 Same model<br/>(any computer)"]
    M2 --> P["🔮 Make predictions"]
    style F fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style P fill:#dcfce7,stroke:#16a34a,color:#14532d
```

### 3.2 Keep It Permanently: Save to Google Drive (10 min)

> [!WARNING]
> **Remember:** Colab's files vanish when the session ends. To keep your model forever, copy it into your Google Drive.

```python
# Connect your Google Drive to this Colab notebook
from google.colab import drive
drive.mount('/content/drive')
# A pop-up will ask permission — click your account and "Allow"
```

```python
# Make a folder in your Drive and copy the model there
import os, shutil
os.makedirs('/content/drive/MyDrive/telco_churn_model', exist_ok=True)
shutil.copy('churn_pipeline.pkl', '/content/drive/MyDrive/telco_churn_model/churn_pipeline.pkl')
print("✅ Model safely saved to your Google Drive")
```

> [!TIP]
> Now even if Colab resets, your model is safe in Drive. You can also download it straight to your computer with: `from google.colab import files; files.download('churn_pipeline.pkl')`.

### 3.3 A Small Script That Uses the Saved Model (25 min)

Let's write a reusable function that loads the saved model and predicts churn for a new customer. It contains *no training code* — just loading and predicting. (In a real project this would live in a file called `predict.py`; in Colab we'll keep it in a cell.)

```python
import joblib
import pandas as pd

def load_model(path="churn_pipeline.pkl"):
    """Open the saved model file."""
    return joblib.load(path)

def predict(customers, model=None):
    """Take a list of customers and return predictions + churn probability."""
    model = model or load_model()
    data = pd.DataFrame(customers)
    data["will_churn"] = ["Yes" if p == 1 else "No" for p in model.predict(data)]
    data["churn_probability"] = model.predict_proba(data)[:, 1].round(3)
    return data

# Try it on one made-up customer
sample_customer = [{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65,
}]

result = predict(sample_customer)
print(result[["will_churn", "churn_probability"]].to_string(index=False))
# will_churn  churn_probability
#        Yes              0.812
```

> [!TIP]
> The example customer above (short tenure, month-to-month contract, electronic-check payment) is a classic high-risk profile, so the model should flag a high churn probability. If you ever get an error here, the most common cause is that a column name in your customer data doesn't exactly match the training column names.

### 3.4 Locking In the Exact Tool Versions (10 min)

> [!WARNING]
> **The number-one reason a saved model breaks later is a version mismatch.** A model saved with scikit-learn version 1.5 may refuse to load under version 1.3. We "freeze" the exact versions into a file so anyone can recreate your exact setup.

```python
# Create a requirements.txt listing the exact versions used today
with open("requirements.txt", "w") as f:
    f.write("pandas==2.2.2\n")
    f.write("scikit-learn==1.5.1\n")
    f.write("joblib==1.4.2\n")
    f.write("matplotlib==3.9.0\n")

print(open("requirements.txt").read())
```

Anyone can later run `pip install -r requirements.txt` to match your environment perfectly.

### 3.5 What is GitHub, and Why Upload? (10 min)

> [!NOTE]
> **GitHub explained simply.** GitHub is like Google Drive for code — but smarter. It stores your project online, keeps a history of every change, and lets others download and run your exact project. Putting your model on GitHub is how you share it with the world (or just your future self on a new laptop).

```mermaid
flowchart LR
    W["💻 Your Colab"] -->|"add + commit"| L["📒 Saved locally<br/>with history"]
    L -->|"push"| R["☁️ GitHub<br/>(online)"]
    R -->|"anyone can clone"| C["🧑‍💻 Another person's<br/>computer / Colab"]
    style R fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    style C fill:#dcfce7,stroke:#16a34a,color:#14532d
```

### 3.6 Uploading Your Project to GitHub from Colab (20 min)

> [!NOTE]
> **One-time setup.** Create a free account at [github.com](https://github.com). Then create a new **empty** repository called `telco-churn-mlops` (click the **+** top-right → *New repository* → leave it empty → *Create*). You'll also need a **Personal Access Token** (GitHub's password for code): go to **Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token**, tick the `repo` box, and copy the token somewhere safe. You'll paste it below.

```python
# Step 1: tell Git who you are (use your GitHub email/name)
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

# Step 2: organise the files we want to upload into one folder
import os, shutil
os.makedirs("telco-churn-mlops", exist_ok=True)
shutil.copy("churn_pipeline.pkl", "telco-churn-mlops/churn_pipeline.pkl")
shutil.copy("requirements.txt", "telco-churn-mlops/requirements.txt")

# A short description file for the project
with open("telco-churn-mlops/README.md", "w") as f:
    f.write("# Telco Churn Model\nDay 1 project: predicts telecom customer churn.\n")

print("✅ Project folder ready")
```

```python
# Step 3: turn the folder into a Git project and push it to GitHub
%cd telco-churn-mlops
!git init -q
!git add .
!git commit -q -m "Day 1: churn model, requirements, and README"
!git branch -M main

# Paste YOUR username and token where shown. The token acts as your password.
USERNAME = "YOUR-GITHUB-USERNAME"
TOKEN = "YOUR-PERSONAL-ACCESS-TOKEN"   # the token you generated above
!git remote add origin https://{USERNAME}:{TOKEN}@github.com/{USERNAME}/telco-churn-mlops.git
!git push -u origin main
%cd ..
print("✅ Pushed to GitHub! Visit your repo to see it online.")
```

> [!WARNING]
> **Keep your token private.** Never share a notebook that still has your real token typed in. After pushing, you can delete the token line. (For real projects, people store tokens more securely — but this is fine for learning.)

### 3.7 The Proof: Reload in a Brand-New Colab (20 min)

This is the moment everything comes together. Open a **fresh Colab notebook** (File → New notebook) — a clean machine that has never seen your work — and resurrect the model from GitHub.

```python
# --- Run this in a NEW, EMPTY Colab notebook ---

# 1. Install the exact same tool versions
!pip install scikit-learn==1.5.1 joblib==1.4.2 pandas==2.2.2 -q

# 2. Download your project from GitHub (use your username)
!git clone https://github.com/YOUR-GITHUB-USERNAME/telco-churn-mlops.git
%cd telco-churn-mlops

# 3. Load the saved model and predict — no retraining needed!
import joblib, pandas as pd
model = joblib.load("churn_pipeline.pkl")

new_customer = pd.DataFrame([{
    "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0,
}])

print("Prediction:", model.predict(new_customer)[0])
print("Churn probability:", model.predict_proba(new_customer)[0, 1].round(3))
```

> [!IMPORTANT]
> **This is the whole point of MLOps.** A model you trained in one Colab session now runs, unchanged, in a brand-new session — loaded by three simple commands a colleague could run. That's the difference between *"it works on my machine"* and *"it works on **any** machine."*

### 3.8 🔥 Learning From a Deliberate Failure (15 min)

The best way to understand why we pinned versions is to break it on purpose.

```python
# Install the WRONG (older) version, then try to load the model
!pip install scikit-learn==1.3.0 -q
import joblib
model = joblib.load("churn_pipeline.pkl")
# You'll get a warning or an error about version mismatch
```

> [!NOTE]
> After running this, reinstall the correct version (`!pip install scikit-learn==1.5.1 -q`) and **restart the runtime** (Runtime → Restart runtime) so the rest of your notebook keeps working.

Here are the three most common reasons a saved model fails to load later — and their fixes:

| 💥 Problem | Why it happens | 🛠️ Fix |
|---|---|---|
| Version warning/error | The tool version changed | Pin versions in `requirements.txt` |
| "Column not found" error | The input data has different columns | Check input columns match training |
| "Unknown category" error | A new category appeared in real data | Use `handle_unknown="ignore"` (we did!) |

> [!TIP]
> Notice the fix for the third problem was *already built in Session 1*. Good decisions early prevent fires later — that's the essence of MLOps.

### 3.9 🧪 Try It Yourself — The Big One (20 min)

> 1. Save your best model as `churn_pipeline.pkl` and copy it to your Google Drive.
> 2. Write the `predict` function and predict for a customer you make up.
> 3. Create `requirements.txt`, then upload the whole project to GitHub from Colab.
> 4. Open a fresh Colab notebook, clone your project, and reproduce a prediction.
> 5. **The final proof:** share your GitHub link with a neighbour. Have them clone and run it in their own Colab. If their copy makes a prediction, **you have shipped a real, reproducible AI model.** 🎉

✅ **Session 3 done:** Your model is saved, stored in Drive, shared on GitHub, and proven to work in a brand-new Colab session.

---

## 🏁 What You Built Today

```mermaid
mindmap
  root(("🏆 Day 1<br/>You shipped<br/>a real model"))
    ("📁 Online Project")
      ("Uploaded to GitHub")
      ("Saved in Google Drive")
    ("🔧 Smart Pipeline")
      ("Auto-cleans data")
      ("Handles surprises")
      ("No data leakage")
    ("⚖️ Fair Comparison")
      ("3 models tested")
      ("Mistakes read as cost")
    ("📦 Saved Model")
      ("One reusable file")
      ("A prediction function")
      ("Locked versions")
    ("♻️ Proven Anywhere")
      ("Reloaded in fresh Colab")
      ("A neighbour ran it")
```

> [!NOTE]
> **Why today matters for the rest of the programme.** Everything ahead — chatbots, document Q&A systems, AI agents — is built on these same habits: get your data, prepare it cleanly, save your work, share it, and prove it runs elsewhere. The advanced AI you'll build on Day 12 is, at its core, the same disciplined act of engineering you just performed today.

---

## ✅ Day 1 Self-Check

Tick each box before you finish:

- [ ] I opened a Colab notebook and ran my first cell
- [ ] I downloaded the Telco Churn dataset into Colab
- [ ] I can explain in one sentence what this dataset is about and why the company cares about churn
- [ ] I know what the four column groups are (who they are, what they bought, how they pay, did they leave)
- [ ] I can describe a high-risk customer without using a model (new, month-to-month, fiber, electronic check, no add-ons)
- [ ] I understand what a model, a feature, and a target are
- [ ] I fixed the `TotalCharges` problem and understood why it happened
- [ ] I can explain the 73% baseline and why it's my honesty anchor
- [ ] I built a Pipeline that prepares data automatically
- [ ] I can explain what SimpleImputer, StandardScaler, and OneHotEncoder each do
- [ ] I understand what random_state, max_iter, and n_estimators mean (and why those numbers)
- [ ] I understand why we split data *before* preparing it (no leakage)
- [ ] I compared 3 models and read a confusion matrix as business cost
- [ ] I saved the whole pipeline with `joblib` and copied it to Google Drive
- [ ] I wrote a `predict` function and predicted for a new customer
- [ ] I uploaded my project to GitHub with locked versions
- [ ] I cloned and ran my model in a brand-new Colab notebook

---

## 📚 One-Page Cheat-Sheet

```python
# THE DAY 1 PATTERN — the shape to remember
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

# 0. GET THE DATA (in Colab)
# !wget -q https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -O telco_churn.csv

# 1. SPLIT FIRST (prevents leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. PREPARE DATA BY COLUMN TYPE
preprocessor = ColumnTransformer([
    ("numbers", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric_features),
    ("categories", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                             ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

# 3. BUNDLE PREPARATION + MODEL
model = Pipeline([("prepare", preprocessor), ("learn", SomeAlgorithm())])
model.fit(X_train, y_train)

# 4. GRADE HONESTLY
scores = cross_val_score(model, X_train, y_train, cv=5)   # beat the 0.73 baseline

# 5. SAVE THE WHOLE PIPELINE
joblib.dump(model, "churn_pipeline.pkl")

# 6. RELOAD ANYWHERE
model = joblib.load("churn_pipeline.pkl")
model.predict(new_data)
```

| 🧠 Idea | 🔑 Remember this |
|---|---|
| Get the data | One `!wget` command pulls the CSV into Colab |
| Data leakage | Split before you prepare — always |
| Pipeline | Preparation + model in one saveable object |
| Baseline | Beat the lazy 73% or you've achieved nothing |
| Confusion matrix | Read "Missed!" vs "False Alarm" as money |
| joblib | Save the *whole pipeline*, not just the model |
| Google Drive | Where Colab files survive after the session ends |
| Version pinning | Unlocked versions = broken model in 6 months |
| handle_unknown | The setting that saves you in the real world |

### ⚙️ Every Setting We Used, in One Place

Keep this handy — whenever you see one of these in code, you'll know exactly what it does and why.

| Setting | What it does | The number we chose & why |
|---|---|---|
| `test_size=0.20` | Fraction of data hidden for testing | 20% — enough to grade fairly, not so much the model starves |
| `random_state=42` | Fixes the "random" shuffle so results repeat | Any fixed number works; 42 is a programmer's in-joke |
| `stratify=y` | Keeps the same Yes/No ratio in both halves | Ensures train & test are fair miniatures of the whole |
| `strategy="median"` | How `SimpleImputer` fills blank **numbers** | Median resists extreme outliers |
| `strategy="most_frequent"` | How `SimpleImputer` fills blank **text** | Can't average words — use the commonest category |
| `handle_unknown="ignore"` | What `OneHotEncoder` does with a new category | Sets all boxes to 0 instead of crashing |
| `max_iter=1000` | Max learning steps for Logistic Regression | Default 100 is sometimes too few; 1000 avoids "didn't finish" |
| `n_estimators=200` | Number of trees in a Random Forest | More trees vote = steadier result; 200 balances speed |
| `cv=5` | Folds in cross-validation | 5 rounds — reliable yet fast |
| `scoring="accuracy"` | How each round is graded | % correct; could swap for `"recall"` on imbalanced data |

> [!TIP]
> **The one rule to remember about all settings:** they're called **hyperparameters** — knobs you set *before* training. You never have to set them all; the defaults are usually sensible. Override one only when you have a reason, and now you know the reason behind each one we touched.

---

<div align="center">

### 🚀 Coming Up — Day 2: Deep Learning & How AI "Understands" Language
*We go inside neural networks and the technology behind ChatGPT, built up step by step.*

**CodeLucky · Faculty Development Programme**
📧 admin@codelucky.com · 📞 +91 85689-70199 · 🌐 codelucky.com

</div>

import streamlit as st  # Import Streamlit for building web app interface

# Set up page configuration (must be the first Streamlit call)
st.set_page_config(page_title="Ambiguity Detector")

# Import required libraries for data handling, NLP, fuzzy logic, and visualization
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import xml.etree.ElementTree as ET  # For parsing XML requirement files
import chardet  # To detect character encoding
import spacy  # NLP library used for POS tagging and text parsing
import matplotlib.pyplot as plt  # For creating plots
import tempfile  # To create temporary files for download
from sklearn.feature_extraction.text import TfidfVectorizer  # To vectorize text for similarity
from sklearn.metrics.pairwise import cosine_similarity  # To calculate similarity between vectors
import skfuzzy as fuzz  # Fuzzy logic toolkit
from skfuzzy import control as ctrl  # Control system interface for fuzzy logic

# ---------- Initialization ----------
st.title("🧠 Ambiguity Detection & Recommendation System")  # Set Streamlit app title
nlp = spacy.load("en_core_web_sm")  # Load spaCy English language model

# ---------- Upload Section ----------
st.sidebar.header("📥 Upload Files")  # Sidebar header for uploads
uploaded_xml = st.sidebar.file_uploader("Upload XML Requirement Files", type=["xml"], accept_multiple_files=True)  # XML upload
uploaded_csv = st.sidebar.file_uploader("Upload CSV Requirements (Optional)", type="csv")  # CSV upload for requirements
uploaded_terms = st.sidebar.file_uploader("Upload Ambiguous Terms CSV (Optional)", type="csv")  # Optional ambiguous terms upload

# ---------- Load Requirements ----------
def load_xml_requirements(files):
    requirements = []  # Initialize list to store requirements
    for f in files:
        raw = f.read()  # Read file in bytes
        encoding = chardet.detect(raw)['encoding']  # Detect encoding
        f.seek(0)  # Reset pointer to beginning
        tree = ET.parse(f)  # Parse XML structure
        root = tree.getroot()  # Get root node of XML
        for text_body in root.findall(".//{req_document.xsd}text_body"):  # Locate requirement text
            parent = text_body.find("..")  # Get parent node for ID extraction
            req_id = parent.attrib.get("id", "N/A") if parent is not None else "Unknown"  # Get ID
            requirements.append({
                "id": req_id,
                "text_body": text_body.text.strip() if text_body.text else ""  # Clean text
            })
    return requirements  # Return list of requirement dicts

requirements = []  # Initialize requirement list
if uploaded_xml:
    requirements += load_xml_requirements(uploaded_xml)  # Append requirements from XML
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)  # Read CSV
    requirements += [{"id": row['ID'], "text_body": row['Requirement']} for _, row in df.iterrows()]  # Append from CSV

# ---------- Ambiguous Term Setup ----------
def get_ambiguous_terms():
    default_terms = ['may', 'could', 'should', 'might', 'possibly', 'unclear', 'unsure', 'some', 'many', 'potential', 'approximately']
    if uploaded_terms:
        df_terms = pd.read_csv(uploaded_terms)
        cleaned_columns = [col.strip().lower() for col in df_terms.columns]  # Normalize column names
        if 'ambiguous term' in cleaned_columns:
            term_col = df_terms.columns[cleaned_columns.index('ambiguous term')]  # Identify correct column
        else:
            term_col = df_terms.columns[0]  # Fallback to first column
        return default_terms + df_terms[term_col].dropna().astype(str).str.lower().tolist()  # Merge and return
    return default_terms  # Return default if no file

ambiguous_terms = get_ambiguous_terms()  # Load terms

# ---------- UI Configuration ----------
st.sidebar.header("🔧 Tuning Parameters")
st.sidebar.slider("TF-IDF Similarity Threshold", 0.0, 1.0, 0.04, 0.01, key="threshold_slider")  # Slider for similarity threshold
st.sidebar.slider("Fuzzy Score Threshold (lower = more ambiguous)", 0.0, 1.0, 0.01, 0.01, key="fuzzy_slider")  # Slider for fuzzy threshold
threshold = st.session_state["threshold_slider"]  # Read slider value
fuzzy_threshold = st.session_state["fuzzy_slider"]

# ---------- Heuristic Rule Matcher ----------
heuristic_rules = {
    "lexical": ["value", "set", "get", "form", "record", "switch"],
    "referential": ["it", "they", "this", "that"],
    "coordination": ["and/or", "if and only if"],
    "scope": ["some", "many", "few", "each"],
    "vague": ["efficient", "user-friendly", "adequate"]
}

def apply_heuristic_rules(text):
    matches = []  # Store matched categories
    for category, keywords in heuristic_rules.items():
        for word in keywords:
            if word.lower() in text.lower():
                matches.append(category)  # Add matching category
                break
    return matches  # Return list of categories matched

# ---------- POS Tagger Matcher + Tree Count ----------
def detect_pos_based_ambiguity(text):
    doc = nlp(text)  # Process with spaCy
    tree_count = len(list(doc.sents))  # Sentence count
    return {
        "modal": any(tok.tag_ == "MD" and tok.text.lower() not in ['shall', 'must'] for tok in doc),  # Modal verbs
        "pronoun": any(tok.pos_ == "PRON" for tok in doc),  # Pronouns
        "vague_adj": any(tok.pos_ == "ADJ" and tok.text.lower() in ['flexible', 'efficient', 'adequate'] for tok in doc),  # Vague adjectives
        "quantifier": any(tok.pos_ == "DET" and tok.text.lower() in ['some', 'many', 'few'] for tok in doc),  # Quantifiers
        "multiple_trees": tree_count > 1  # More than 1 sentence tree = potential structural ambiguity
    }

# ---------- Fuzzy Setup ----------
similarity = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'similarity')  # Input variable for similarity
ambiguity = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'ambiguity')  # Output variable for ambiguity

# Define fuzzy membership functions
similarity['low'] = fuzz.trapmf(similarity.universe, [0, 0, 0.2, 0.4])
similarity['medium'] = fuzz.trimf(similarity.universe, [0.3, 0.5, 0.7])
similarity['high'] = fuzz.trapmf(similarity.universe, [0.6, 0.8, 1, 1])
ambiguity['low'] = fuzz.trapmf(ambiguity.universe, [0.5, 0.7, 1, 1])
ambiguity['high'] = fuzz.trapmf(ambiguity.universe, [0, 0, 0.3, 0.5])

# Define fuzzy rules
rules = [
    ctrl.Rule(similarity['low'], ambiguity['high']),  # Low similarity => high ambiguity
    ctrl.Rule(similarity['medium'], ambiguity['high']),  # Medium similarity => high ambiguity
    ctrl.Rule(similarity['high'], ambiguity['low'])  # High similarity => low ambiguity
]

# Create fuzzy controller
fz_ctrl = ctrl.ControlSystem(rules)  # Control system with the defined rules
simulator = ctrl.ControlSystemSimulation(fz_ctrl)  # Simulation engine for inference

# ---------- Main Analysis ----------
def detect_structural_ambiguity(text):
    doc = nlp(text)
    verbs = [tok for tok in doc if tok.pos_ == "VERB"]
    cconjs = [tok for tok in doc if tok.pos_ == "CCONJ"]
    adjectives = [tok for tok in doc if tok.pos_ == "ADJ"]
    return {
        "multi_verbs_and": len(verbs) >= 2 and any(tok.text.lower() == "and" for tok in doc),
        "two_conjunctions": len(cconjs) >= 2,
        "multiple_adjectives": len(adjectives) >= 2
    }

def analyze(requirements, ambiguous_terms):
    results = []  # List to store results
    all_texts = [r['text_body'].lower() for r in requirements] + ambiguous_terms  # Combine all texts for vectorization
    vec = TfidfVectorizer().fit_transform(all_texts)  # Fit and transform to TF-IDF vectors
    req_vecs = vec[:len(requirements)]  # Vectorized requirements
    amb_vecs = vec[len(requirements):]  # Vectorized ambiguous terms

    for i, req in enumerate(requirements):
        sim_scores = cosine_similarity(req_vecs[i], amb_vecs).flatten()  # Compute cosine similarity
        max_score = max(sim_scores)  # Get max similarity
        simulator.input['similarity'] = max_score  # Input to fuzzy system
        simulator.compute()  # Compute fuzzy output
        fuzzy_score = simulator.output['ambiguity']  # Output fuzzy score
        heuristics = apply_heuristic_rules(req['text_body'])  # Apply heuristic rules
        pos_flags = detect_pos_based_ambiguity(req['text_body'])  # Get POS-based flags
        structural_flags = detect_structural_ambiguity(req['text_body'])  # Get structural flags

        # Final result dictionary for the requirement
        results.append({
            "ID": req['id'],
            "Requirement Text": req['text_body'],
            "Max Similarity": max_score,
            "Fuzzy Score": fuzzy_score,
            "Heuristic Types": heuristics,
            "POS Flags": pos_flags,
            "Structural Flags": structural_flags,
            "Is Ambiguous": max_score > threshold or fuzzy_score < fuzzy_threshold or bool(heuristics) or any(pos_flags.values())
        })
    return pd.DataFrame(results)  # Convert results to DataFrame
...  # [previous content unchanged]

# ========== MAIN APP DISPLAY & INTERACTION ==========
if requirements:
    df_results = analyze(requirements, ambiguous_terms)  # Analyze uploaded requirements

    st.subheader("🔍 Ambiguity Detection Results")  # Section header
    st.dataframe(df_results)  # Display full results table

    st.subheader("📊 Ambiguity Classification Summary")  # Summary section
    ambiguity_counts = df_results['Is Ambiguous'].value_counts().rename(index={True: 'Ambiguous', False: 'Not Ambiguous'})  # Count ambiguous vs not
    fig1, ax1 = plt.subplots()  # Create plot
    ambiguity_counts.plot(kind='bar', color=['red', 'green'], ax=ax1)  # Bar chart
    ax1.set_title("Ambiguity Detection Result")
    ax1.set_ylabel("Number of Requirements")
    st.pyplot(fig1)  # Display plot

    st.subheader("📈 Fuzzy Score Distribution")  # Fuzzy score histogram
    fig2, ax2 = plt.subplots()
    df_results['Fuzzy Score'].plot(kind='hist', bins=10, color='skyblue', edgecolor='black', ax=ax2)
    ax2.set_title("Distribution of Fuzzy Scores")
    ax2.set_xlabel("Fuzzy Score")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    st.subheader("📎 POS Flag Breakdown")  # POS-based summary
    pos_flag_counts = pd.DataFrame(df_results['POS Flags'].apply(pd.Series).sum()).rename(columns={0: 'Count'})
    fig3, ax3 = plt.subplots()
    pos_flag_counts.plot(kind='bar', ax=ax3, legend=False)
    ax3.set_title("POS-based Ambiguity Indicators")
    ax3.set_ylabel("Occurrences")
    st.pyplot(fig3)

    st.subheader("🧮 Similarity vs Fuzzy Score")  # Scatter plot between similarity and fuzzy score
    fig4, ax4 = plt.subplots()
    ax4.scatter(df_results['Max Similarity'], df_results['Fuzzy Score'], color='purple')
    ax4.set_xlabel("Max Similarity")
    ax4.set_ylabel("Fuzzy Score")
    ax4.set_title("Correlation between Similarity and Fuzzy Interpretation")
    st.pyplot(fig4)

    st.subheader("📤 Export")  # Export section
    if st.button("Export to CSV"):  # On click
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df_results.to_csv(tmp.name, index=False)  # Save to file
            st.success("Exported to CSV file")
            st.download_button("📥 Download CSV", data=open(tmp.name, "rb").read(), file_name="ambiguity_results.csv")

    # ========== PHASE 4: USER FEEDBACK & INTERACTION ==========
    st.subheader("📝 User Feedback on Ambiguity Results")
    st.write("Review each requirement and update the ambiguity label if needed.")

    if "user_feedback" not in st.session_state:  # Initialize feedback state
        st.session_state["user_feedback"] = ["Ambiguous" if val else "Not Ambiguous" for val in df_results["Is Ambiguous"]]

    feedback_list = []  # Collect user feedback
    for idx, row in df_results.iterrows():
        default_val = "Ambiguous" if row["Is Ambiguous"] else "Not Ambiguous"
        user_choice = st.selectbox(
            f"Requirement ID {row['ID']} ({row['Requirement Text'][:45]}...)",
            ["Ambiguous", "Not Ambiguous"],
            index=0 if st.session_state["user_feedback"][idx] == "Ambiguous" else 1,
            key=f"feedback_{idx}"
        )
        feedback_list.append(user_choice)

    if st.button("Save Feedback"):  # Save feedback button
        df_results["Is Ambiguous"] = [
            False if feedback == "Not Ambiguous" else row["Is Ambiguous"]
            for feedback, (_, row) in zip(feedback_list, df_results.iterrows())
        ]
        st.session_state["user_feedback"] = feedback_list  # Save to session
        df_results["User Feedback"] = feedback_list  # Store feedback in DataFrame
        st.success("User feedback has been saved! You can now reanalyze the results.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df_results.to_csv(tmp.name, index=False)
            st.download_button("📥 Download Feedback CSV", data=open(tmp.name, "rb").read(), file_name="user_feedback.csv")

    # ========== PHASE 5: FEEDBACK INTEGRATION & REANALYSIS ==========
    if "User Feedback" in df_results.columns:
        st.subheader("🔁 Reanalyze Requirements Based on Feedback")
        if st.button("Reanalyze with User Feedback"):
            updated_ambiguous_terms = ambiguous_terms.copy()
            for i, row in df_results.iterrows():
                if row["User Feedback"] == "Ambiguous":
                    updated_ambiguous_terms.append(row["Requirement Text"].lower())  # Learn from feedback
            updated_df = analyze(requirements, updated_ambiguous_terms)  # Reanalyze
            df_results = updated_df

            updated_df["Is Ambiguous"] = [
                False if feedback == "Not Ambiguous" else True
                for feedback in df_results["User Feedback"]
            ]

            st.subheader("✅ Updated Requirements After Feedback")
            for idx, row in updated_df.iterrows():
                st.markdown(f"**Requirement ID {row['ID']}**: {row['Requirement Text']}")
                st.checkbox("Is Ambiguous", value=row["Is Ambiguous"], key=f"updated_checkbox_{idx}", disabled=True)

            updated_df["Was Ambiguous Before"] = df_results["Is Ambiguous"]
            updated_df["Is Ambiguous Now"] = updated_df["Is Ambiguous"]
            st.dataframe(updated_df[["ID", "Requirement Text", "Was Ambiguous Before", "Is Ambiguous Now"]])

            st.success("System has reanalyzed the requirements with your feedback.")
            st.dataframe(updated_df)
            st.subheader("📤 Export Updated Results")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                updated_df.to_csv(tmp.name, index=False)
                st.download_button("📥 Download Updated CSV", data=open(tmp.name, "rb").read(), file_name="updated_ambiguity_results.csv")

else:
    st.warning("Please upload at least one XML or CSV file.")  # Prompt user to upload files

# ========== EVALUATION SECTION ==========
st.sidebar.header("🧪 Evaluate Your Ambiguity Detection System")
uploaded_labeled_test = st.sidebar.file_uploader("Upload Labeled Test Dataset (.csv)", type="csv", key="labeled_eval")

if uploaded_labeled_test:
    st.subheader("🧪 Evaluation Based on Your Custom Detection System")
    test_df = pd.read_csv(uploaded_labeled_test)
    if 'Requirement' not in test_df.columns or 'Detected as' not in test_df.columns:
        st.error("CSV must contain at least 'Requirement' and 'Detected as' columns.")  # Validate columns
    else:
        eval_requirements = [{"id": str(i), "text_body": row['Requirement']} for i, row in test_df.iterrows()]  # Convert to required format
        results_df = analyze(eval_requirements, ambiguous_terms)  # Run analyzer
        y_pred = results_df['Is Ambiguous'].astype(int)  # Predicted values
        y_true = test_df['Detected as'].map({'NOCUOUS': 1, 'INNOCUOUS': 0}).astype(int)  # True labels

        from sklearn.metrics import accuracy_score, classification_report  # Import metrics
        accuracy = accuracy_score(y_true, y_pred)  # Calculate accuracy
        st.markdown(f"### ✅ Custom System Accuracy: `{accuracy:.2f}`")

        report = classification_report(y_true, y_pred, output_dict=True)  # Get classification report
        st.subheader("📋 Classification Report ")
        st.dataframe(pd.DataFrame(report).transpose())  # Show metrics

        labels = [0, 1]
        label_names = ["Not Ambiguous", "Ambiguous"]
        precision = [report[str(label)]["precision"] for label in labels]
        recall = [report[str(label)]["recall"] for label in labels]
        f1 = [report[str(label)]["f1-score"] for label in labels]

        fig, ax = plt.subplots(figsize=(8, 6))
        x = range(len(label_names))
        ax.bar(x, precision, width=0.2, label='Precision', align='center')
        ax.bar([p + 0.2 for p in x], recall, width=0.2, label='Recall', align='center')
        ax.bar([p + 0.4 for p in x], f1, width=0.2, label='F1 Score', align='center')
        ax.set_xticks([p + 0.2 for p in x])
        ax.set_xticklabels(label_names)
        ax.set_ylim([0, 1])
        ax.set_title(f"Your System Classification Metrics (Accuracy: {accuracy:.2f})")
        ax.legend()
        st.pyplot(fig)  # Show bar chart

import streamlit as st  # Streamlit for UI rendering

# This must be the very first Streamlit call
st.set_page_config(page_title="Ambiguity Detector")  # Set the app page title

# Standard data and parsing libraries
import pandas as pd  # DataFrames and CSV handling
import numpy as np  # Numerical operations
import xml.etree.ElementTree as ET  # For XML parsing
import chardet  # To detect file encoding
import spacy  # Natural Language Processing
import matplotlib.pyplot as plt  # Plotting and charts
import tempfile  # For saving temporary files

# ML libraries for similarity comparison
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to TF-IDF matrix
from sklearn.metrics.pairwise import cosine_similarity  # Calculates cosine similarity between vectors

# Fuzzy logic tools
import skfuzzy as fuzz  # Core fuzzy logic toolkit
from skfuzzy import control as ctrl  # Control system for fuzzy rule setup

# ---------- Initialization ----------
st.title("🧠 Ambiguity Detection & Recommendation System")  # Main title for the web app
nlp = spacy.load("en_core_web_sm")  # Load spaCy NLP model

# ---------- Upload Section ----------
st.sidebar.header("📥 Upload Files")  # Sidebar section for file upload
uploaded_xml = st.sidebar.file_uploader("Upload XML Requirement Files", type=["xml"], accept_multiple_files=True)  # Upload multiple XML files
uploaded_csv = st.sidebar.file_uploader("Upload CSV Requirements (Optional)", type="csv")  # Optionally upload a CSV
uploaded_terms = st.sidebar.file_uploader("Upload Ambiguous Terms CSV (Optional)", type="csv")  # Optional ambiguous term list

# ---------- Load Requirements ----------
def load_xml_requirements(files):  # Function to extract requirements from XML
    requirements = []  # List to store all extracted requirements
    for f in files:
        raw = f.read()  # Read file content
        encoding = chardet.detect(raw)['encoding']  # Detect encoding type
        f.seek(0)  # Reset file pointer to beginning
        tree = ET.parse(f)  # Parse XML structure
        root = tree.getroot()  # Get root of the XML document
        for text_body in root.findall(".//{req_document.xsd}text_body"):  # Find all requirement text bodies
            parent = text_body.find("..")  # Get parent node
            req_id = parent.attrib.get("id", "N/A") if parent is not None else "Unknown"  # Extract ID
            requirements.append({  # Append to list
                "id": req_id,
                "text_body": text_body.text.strip() if text_body.text else ""  # Clean text
            })
    return requirements  # Return list of requirements

requirements = []  # Master list of all loaded requirements
if uploaded_xml:
    requirements += load_xml_requirements(uploaded_xml)  # Add requirements from uploaded XML
if uploaded_csv:
    df = pd.read_csv(uploaded_csv)  # Read uploaded CSV
    requirements += [{"id": row['ID'], "text_body": row['Requirement']} for _, row in df.iterrows()]  # Append rows to requirements

# ---------- Ambiguous Term Setup ----------
def get_ambiguous_terms():  # Load or define ambiguous terms
    default_terms = ['may', 'could', 'should', 'might', 'possibly', 'unclear', 'unsure', 'some', 'many', 'potential', 'approximately']  # Default terms
    if uploaded_terms:  # If user uploads custom term list
        df_terms = pd.read_csv(uploaded_terms)  # Read CSV
        cleaned_columns = [col.strip().lower() for col in df_terms.columns]  # Clean column names
        if 'ambiguous term' in cleaned_columns:
            term_col = df_terms.columns[cleaned_columns.index('ambiguous term')]  # Use specific column if matched
        else:
            term_col = df_terms.columns[0]  # Otherwise use first column
        return default_terms + df_terms[term_col].dropna().astype(str).str.lower().tolist()  # Combine lists
    return default_terms  # Return default if no upload

ambiguous_terms = get_ambiguous_terms()  # Load terms for use

# ---------- UI Configuration ----------
st.sidebar.header("🔧 Tuning Parameters")  # Sidebar for parameter tuning
st.sidebar.slider("TF-IDF Similarity Threshold", 0.0, 1.0, 0.04, 0.01, key="threshold_slider")  # Similarity threshold
st.sidebar.slider("Fuzzy Score Threshold (lower = more ambiguous)", 0.0, 1.0, 0.01, 0.01, key="fuzzy_slider")  # Fuzzy threshold
threshold = st.session_state["threshold_slider"]  # Read TF-IDF threshold from slider
fuzzy_threshold = st.session_state["fuzzy_slider"]  # Read fuzzy threshold

# ---------- Heuristic Rule Matcher ----------
heuristic_rules = {
    "lexical": ["value", "set", "get", "form", "record", "switch"],
    "referential": ["it", "they", "this", "that"],
    "coordination": ["and/or", "if and only if"],
    "scope": ["some", "many", "few", "each"],
    "vague": ["efficient", "user-friendly", "adequate"]
}  # Define categories and words for heuristic checks

def apply_heuristic_rules(text):  # Match text against heuristic rules
    matches = []
    for category, keywords in heuristic_rules.items():
        for word in keywords:
            if word.lower() in text.lower():
                matches.append(category)  # Add category if match found
                break
    return matches  # Return list of matched heuristic categories

# ---------- POS Tagger Matcher + Tree Count ----------
def detect_pos_based_ambiguity(text):  # Detect ambiguity using POS tags
    doc = nlp(text)  # Parse text with spaCy
    tree_count = len(list(doc.sents))  # Count number of sentences (trees)
    return {
        "modal": any(tok.tag_ == "MD" and tok.text.lower() not in ['shall', 'must'] for tok in doc),  # Modal detection
        "pronoun": any(tok.pos_ == "PRON" for tok in doc),  # Pronoun check
        "vague_adj": any(tok.pos_ == "ADJ" and tok.text.lower() in ['flexible', 'efficient', 'adequate'] for tok in doc),  # Vague adjectives
        "quantifier": any(tok.pos_ == "DET" and tok.text.lower() in ['some', 'many', 'few'] for tok in doc),  # Quantifiers
        "multiple_trees": tree_count > 1  # More than one sentence
    }

# ---------- Fuzzy Setup ----------
similarity = ctrl.Antecedent(np.arange(0, 1.1, 0.01), 'similarity')  # Fuzzy input for similarity
ambiguity = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'ambiguity')  # Fuzzy output for ambiguity

# Define membership functions for similarity
similarity['low'] = fuzz.trapmf(similarity.universe, [0, 0, 0.2, 0.4])
similarity['medium'] = fuzz.trimf(similarity.universe, [0.3, 0.5, 0.7])
similarity['high'] = fuzz.trapmf(similarity.universe, [0.6, 0.8, 1, 1])

# Define membership functions for ambiguity
ambiguity['low'] = fuzz.trapmf(ambiguity.universe, [0.5, 0.7, 1, 1])
ambiguity['high'] = fuzz.trapmf(ambiguity.universe, [0, 0, 0.3, 0.5])

# Fuzzy rules
rules = [
    ctrl.Rule(similarity['low'], ambiguity['high']),
    ctrl.Rule(similarity['medium'], ambiguity['high']),
    ctrl.Rule(similarity['high'], ambiguity['low'])
]

fz_ctrl = ctrl.ControlSystem(rules)  # Define fuzzy control system
simulator = ctrl.ControlSystemSimulation(fz_ctrl)  # Create simulator object for fuzzy logic

# ---------- Main Analysis ----------
def detect_structural_ambiguity(text):  # Additional structural pattern detection
    doc = nlp(text)
    verbs = [tok for tok in doc if tok.pos_ == "VERB"]
    cconjs = [tok for tok in doc if tok.pos_ == "CCONJ"]
    adjectives = [tok for tok in doc if tok.pos_ == "ADJ"]
    return {
        "multi_verbs_and": len(verbs) >= 2 and any(tok.text.lower() == "and" for tok in doc),
        "two_conjunctions": len(cconjs) >= 2,
        "multiple_adjectives": len(adjectives) >= 2
    }

def analyze(requirements, ambiguous_terms):  # Core function to evaluate ambiguity
    results = []  # Store result for each requirement
    all_texts = [r['text_body'].lower() for r in requirements] + ambiguous_terms  # Combine requirement text and terms
    vec = TfidfVectorizer().fit_transform(all_texts)  # Convert to TF-IDF vectors
    req_vecs = vec[:len(requirements)]  # Slice requirement vectors
    amb_vecs = vec[len(requirements):]  # Slice ambiguous term vectors

    for i, req in enumerate(requirements):  # Evaluate each requirement
        sim_scores = cosine_similarity(req_vecs[i], amb_vecs).flatten()  # Compute similarity
        max_score = max(sim_scores)  # Take highest score
        simulator.input['similarity'] = max_score  # Feed to fuzzy system
        simulator.compute()  # Compute fuzzy output
        fuzzy_score = simulator.output['ambiguity']  # Get ambiguity score
        heuristics = apply_heuristic_rules(req['text_body'])  # Run heuristics
        pos_flags = detect_pos_based_ambiguity(req['text_body'])  # Run POS detection
        structural_flags = detect_structural_ambiguity(req['text_body'])  # Run structural check
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
    return pd.DataFrame(results)  # Return results as DataFrame

# ========== MAIN APP DISPLAY & INTERACTION ==========
if requirements:  # If there are uploaded requirements to process
    df_results = analyze(requirements, ambiguous_terms)  # Analyze uploaded requirements and get results

    st.subheader("🔍 Ambiguity Detection Results")  # Display subtitle in UI
    st.dataframe(df_results)  # Show the analysis results in a table

    st.subheader("📊 Ambiguity Classification Summary")  # Subtitle for ambiguity count chart
    ambiguity_counts = df_results['Is Ambiguous'].value_counts().rename(index={True: 'Ambiguous', False: 'Not Ambiguous'})  # Count ambiguity labels
    fig1, ax1 = plt.subplots()  # Create a figure and axis for bar chart
    ambiguity_counts.plot(kind='bar', color=['red', 'green'], ax=ax1)  # Plot bar chart of ambiguity counts
    ax1.set_title("Ambiguity Detection Result")  # Set chart title
    ax1.set_ylabel("Number of Requirements")  # Label y-axis
    st.pyplot(fig1)  # Display the plot in Streamlit

    st.subheader("📈 Fuzzy Score Distribution")  # Subtitle for fuzzy score distribution
    fig2, ax2 = plt.subplots()  # Create histogram figure
    df_results['Fuzzy Score'].plot(kind='hist', bins=10, color='skyblue', edgecolor='black', ax=ax2)  # Histogram of fuzzy scores
    ax2.set_title("Distribution of Fuzzy Scores")  # Set title
    ax2.set_xlabel("Fuzzy Score")  # Label x-axis
    ax2.set_ylabel("Frequency")  # Label y-axis
    st.pyplot(fig2)  # Show histogram in UI

    st.subheader("📎 POS Flag Breakdown")  # Subtitle for POS tag summary
    pos_flag_counts = pd.DataFrame(df_results['POS Flags'].apply(pd.Series).sum()).rename(columns={0: 'Count'})  # Count each POS flag
    fig3, ax3 = plt.subplots()  # Create figure
    pos_flag_counts.plot(kind='bar', ax=ax3, legend=False)  # Plot bar chart of POS flags
    ax3.set_title("POS-based Ambiguity Indicators")  # Set title
    ax3.set_ylabel("Occurrences")  # Label y-axis
    st.pyplot(fig3)  # Show chart in Streamlit

    st.subheader("🧮 Similarity vs Fuzzy Score")  # Subtitle for scatter plot
    fig4, ax4 = plt.subplots()  # Create scatter plot
    ax4.scatter(df_results['Max Similarity'], df_results['Fuzzy Score'], color='purple')  # Plot similarity vs fuzzy score
    ax4.set_xlabel("Max Similarity")  # X-axis label
    ax4.set_ylabel("Fuzzy Score")  # Y-axis label
    ax4.set_title("Correlation between Similarity and Fuzzy Interpretation")  # Plot title
    st.pyplot(fig4)  # Show the scatter plot

    st.subheader("📤 Export")  # Export section header
    if st.button("Export to CSV"):  # Button to export CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:  # Create temp file
            df_results.to_csv(tmp.name, index=False)  # Save results to file
            st.success("Exported to CSV file")  # Show success message
            st.download_button("📥 Download CSV", data=open(tmp.name, "rb").read(), file_name="ambiguity_results.csv")  # Download button

    # ========== PHASE 4: USER FEEDBACK & INTERACTION ==========
    st.subheader("📝 User Feedback on Ambiguity Results")  # Subtitle for user feedback
    st.write("Review each requirement and update the ambiguity label if needed.")  # Instructions to user

    if "user_feedback" not in st.session_state:  # If no feedback stored yet
        st.session_state["user_feedback"] = ["Ambiguous" if val else "Not Ambiguous" for val in df_results["Is Ambiguous"]]  # Initialize feedback state

    feedback_list = []  # List to store feedback selections
    for idx, row in df_results.iterrows():  # Loop through each result row
        default_val = "Ambiguous" if row["Is Ambiguous"] else "Not Ambiguous"  # Default selection
        user_choice = st.selectbox(  # Dropdown for user input
            f"Requirement ID {row['ID']} ({row['Requirement Text'][:45]}...)",
            ["Ambiguous", "Not Ambiguous"],
            index=0 if st.session_state["user_feedback"][idx] == "Ambiguous" else 1,
            key=f"feedback_{idx}"
        )
        feedback_list.append(user_choice)  # Save user choice

    if st.button("Save Feedback"):  # Button to save feedback
        df_results["Is Ambiguous"] = [  # Override system label with feedback if user marked as Not Ambiguous
            False if feedback == "Not Ambiguous" else row["Is Ambiguous"]
            for feedback, (_, row) in zip(feedback_list, df_results.iterrows())
        ]
        st.session_state["user_feedback"] = feedback_list  # Store in session state
        df_results["User Feedback"] = feedback_list  # Add feedback column to DataFrame
        st.success("User feedback has been saved! You can now reanalyze the results.")  # Success message

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:  # Save feedback to CSV
            df_results.to_csv(tmp.name, index=False)
            st.download_button("📥 Download Feedback CSV", data=open(tmp.name, "rb").read(), file_name="user_feedback.csv")

    # ========== PHASE 5: FEEDBACK INTEGRATION & REANALYSIS ==========
    if "User Feedback" in df_results.columns:  # If feedback exists
        st.subheader("🔁 Reanalyze Requirements Based on Feedback")  # Subtitle
        if st.button("Reanalyze with User Feedback"):  # Reanalyze button
            updated_ambiguous_terms = ambiguous_terms.copy()  # Copy existing terms
            for i, row in df_results.iterrows():
                if row["User Feedback"] == "Ambiguous":
                    updated_ambiguous_terms.append(row["Requirement Text"].lower())  # Add new terms from feedback
            updated_df = analyze(requirements, updated_ambiguous_terms)  # Run new analysis
            df_results = updated_df  # Update results

            updated_df["Is Ambiguous"] = [  # Apply new feedback label
                False if feedback == "Not Ambiguous" else True
                for feedback in df_results["User Feedback"]
            ]

            st.subheader("✅ Updated Requirements After Feedback")  # Show updated section
            for idx, row in updated_df.iterrows():
                st.markdown(f"**Requirement ID {row['ID']}**: {row['Requirement Text']}")  # Display requirement
                st.checkbox("Is Ambiguous", value=row["Is Ambiguous"], key=f"updated_checkbox_{idx}", disabled=True)  # Show final flag

            updated_df["Was Ambiguous Before"] = df_results["Is Ambiguous"]  # Before reanalysis
            updated_df["Is Ambiguous Now"] = updated_df["Is Ambiguous"]  # After reanalysis
            st.dataframe(updated_df[["ID", "Requirement Text", "Was Ambiguous Before", "Is Ambiguous Now"]])  # Show comparison

            st.success("System has reanalyzed the requirements with your feedback.")
            st.dataframe(updated_df)
            st.subheader("📤 Export Updated Results")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                updated_df.to_csv(tmp.name, index=False)
                st.download_button("📥 Download Updated CSV", data=open(tmp.name, "rb").read(), file_name="updated_ambiguity_results.csv")

else:
    st.warning("Please upload at least one XML or CSV file.")  # Warn if no files uploaded

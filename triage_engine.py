import os
import pandas as pd
from collections import Counter

class ClinicalDifferentialEngine:
    def __init__(self, dataset_path: str = "AyurGenixAI_Dataset.csv"):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(self.dataset_path)
        
        # Clean string columns
        self.df['Disease'] = self.df['Disease'].astype(str).str.strip()
        self.df['Symptoms_Clean'] = self.df['Symptoms'].astype(str).str.lower()
        
        # Pre-extract all individual symptom phrases across 446 rows
        all_syms = []
        for s in self.df['Symptoms_Clean']:
            all_syms.extend([item.strip() for item in s.split(',') if len(item.strip()) > 2])
        
        # Keep top 30 most frequent symptoms as questioning factors
        sym_counts = Counter(all_syms)
        self.top_factors = [s for s, count in sym_counts.most_common(30)]
        print(f"[Engine] Loaded {len(self.df)} diseases from {dataset_path} with {len(self.top_factors)} primary diagnostic factors.")

    def evaluate_state(self, positive_symptoms: list, negative_symptoms: list):
        filtered_df = self.df.copy()

        # Filter: Disease must contain all confirmed positive symptoms
        for sym in positive_symptoms:
            sym_clean = sym.strip().lower()
            filtered_df = filtered_df[filtered_df['Symptoms_Clean'].str.contains(sym_clean, regex=False)]

        # Filter: Exclude diseases containing confirmed negative symptoms
        for sym in negative_symptoms:
            sym_clean = sym.strip().lower()
            filtered_df = filtered_df[~filtered_df['Symptoms_Clean'].str.contains(sym_clean, regex=False)]

        remaining_count = len(filtered_df)

        # Convergence criteria (Stop when 1-3 diseases remain or 6 questions answered)
        if remaining_count <= 3 or (len(positive_symptoms) + len(negative_symptoms)) >= 6:
            candidates = []
            for _, row in filtered_df.head(3).iterrows():
                candidates.append({
                    "disease": row['Disease'],
                    "hindi_name": row.get('Hindi Name', ''),
                    "dosha": row.get('Doshas', 'Tridosha'),
                    "prakriti": row.get('Constitution/Prakriti', 'Vata-Pitta'),
                    "herbs": row.get('Ayurvedic Herbs', 'Tulsi, Giloy'),
                    "lifestyle": row.get('Diet and Lifestyle Recommendations', '')
                })
            
            # Fallback if over-filtered
            if not candidates:
                candidates = [{"disease": "General Malaise", "hindi_name": "सामान्य अस्वस्थता", "dosha": "Tridosha"}]

            return {
                "status": "complete",
                "probable_diseases": candidates,
                "next_question_factor": None
            }

        # Select next best factor to ask:
        # Find which unasked symptom splits the remaining candidate pool closest to 50%
        unasked = [f for f in self.top_factors if f not in positive_symptoms and f not in negative_symptoms]
        if not unasked:
            return {
                "status": "complete",
                "probable_diseases": filtered_df[['Disease', 'Hindi Name', 'Doshas']].head(3).to_dict(orient='records'),
                "next_question_factor": None
            }

        # Calculate presence ratio for each candidate factor among remaining diseases
        best_factor = None
        best_distance = 1.0
        for factor in unasked:
            ratio = filtered_df['Symptoms_Clean'].str.contains(factor, regex=False).mean()
            dist = abs(ratio - 0.5)
            if dist < best_distance:
                best_distance = dist
                best_factor = factor

        return {
            "status": "continue",
            "probable_diseases": filtered_df['Disease'].head(3).tolist(),
            "next_question_factor": best_factor or unasked[0],
            "remaining_candidates_count": remaining_count
        }
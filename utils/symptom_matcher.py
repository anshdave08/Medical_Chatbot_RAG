import json
import os

SYMPTOM_MAP = {
    "fever": ["fever", "high temperature", "body heat"],
    "headache": ["headache", "head pain"],
    "vomiting": ["vomiting", "nausea"],
    "fatigue": ["fatigue", "tiredness", "weakness"]
}

def get_disease_info(name):
    diseases = load_diseases()
    for d in diseases:
        if d["disease"].lower() == name.lower():
            return d
    return None

def load_diseases():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "Dataset", "Dataset.json")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def match_disease(user_input):
    user_input = user_input.lower()
    diseases = load_diseases()

    scores = []

    for d in diseases:
        score = 0
        for symptom in d["symptoms"]:
            for key, variants in SYMPTOM_MAP.items():
                if symptom == key:
                    if any(v in user_input for v in variants):
                        score += 1

        if score > 0:
            scores.append((d["disease"], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    total = sum(s for _, s in scores) or 1

    return [
        {"disease": d, "confidence": round((s / total) * 100, 2)}
        for d, s in scores[:3]
    ]
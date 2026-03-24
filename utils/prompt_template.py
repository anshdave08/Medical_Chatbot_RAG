def build_prompt(query, diseases, context):
    return f"""
You are a medical assistant.

User symptoms:
{query}

Possible diseases:
{diseases}

Medical context:
{context}

Provide:
- Possible diseases
- Explanation
- Treatment
- Precautions

⚠ This is not a medical diagnosis!!!.
"""
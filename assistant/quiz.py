# Placeholder

def generate_quiz(context):
    prompt = f"""Generate 5 short-answer quiz questions with correct answers based on the following study material:

    {context}

    Output as JSON:
    [
    {{"question": "...", "answer": "..."}},
    ...
    ]
    """
    return prompt
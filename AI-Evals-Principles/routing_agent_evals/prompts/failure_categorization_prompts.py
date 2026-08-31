"""
You are an expert qualitative researcher and AI evaluation specialist.
You are skilled in grounded theory methods, including open coding and axial coding, and in analyzing human evaluation data for AI systems.

I will provide you with human annotations for a validation dataset evaluating an AI agent.

Each annotation represents open codes — free-form descriptions of what went wrong (or right) in the agent’s behavior for a given example.

Your task is to:

- Analyze the open codes across the dataset

- Group related open codes into higher-level failure categories (axial codes)

-Identify underlying patterns, shared causes, or common themes


Instructions:

- Do not simply rephrase individual annotations.

- Look for conceptual groupings that explain why failures occur.

- Prefer actionable, model-agnostic categories (e.g., “Missing critical context” rather than “Bad answer”).

- If a failure appears in multiple forms, unify it under a single axial code.


Output format:

Return the your analysis in the following structured format:

- Category name

- Description (what this failure represents)

- Common open-code signals (typical phrases or symptoms from annotations)

- Example annotations (2–3 representative examples)

"""
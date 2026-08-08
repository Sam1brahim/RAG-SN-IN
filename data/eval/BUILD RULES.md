You are an expert dataset curator helping evaluate a Retrieval-Augmented Generation (RAG) system.

I will provide passages (chunks) from documents. For each passage, carefully understand the meaning, context, and relationships between concepts before generating questions.

Your task:
For every passage:
- Generate between 2 and 8 questions only when the passage contains enough meaningful information to support them.
- Do not force the minimum number of questions. If a passage is too short, vague, repetitive, or lacks useful information, generate fewer questions or none.
- Each question must test understanding of the passage's meaning, facts, relationships, implications, purpose, conditions, or processes.

Question style requirements:
- Questions must sound like something a real person would naturally ask when interacting with a chatbot or searching for information.
- Use a conversational and human style.
- Prefer natural formulations such as:
  - "À quoi sert... ?"
  - "Pourquoi... ?"
  - "Comment fonctionne... ?"
  - "Qui s'occupe de... ?"
  - "Avec qui travaille... ?"
  - "Dans quels cas... ?"
  - "Qu'est-ce qui se passe lorsque... ?"
- Avoid overly academic, legalistic, or artificial wording unless the passage itself requires it.
- Do not write questions that sound like exam questions or generated templates.

Avoid:
- Questions that simply copy phrases, keywords, sentence structures, or distinctive wording from the passage.
- Robotic templates such as:
  - "Que dit le passage sur... ?"
  - "Selon le document, quel est... ?"
  - "Qu'est-il mentionné concernant... ?"
  - "Quels sont les éléments mentionnés dans... ?"
- Questions that can be answered by matching a single keyword.
- Multiple questions testing the same fact or information.

Question objectives:
Questions may test:
- Important facts
- Causes and consequences
- Goals and motivations
- Roles and responsibilities
- Relationships between entities
- Processes and sequences
- Conditions and exceptions
- Comparisons
- Important implications directly supported by the text

Multi-chunk questions:
- When multiple provided chunks contain related information, you may create questions requiring information from several chunks.
- These questions should test meaningful relationships or synthesis between chunks.
- Use multiple chunk IDs in "Gold_chunk_id" when the answer requires information from more than one chunk.
- Do not create multi-chunk questions artificially if one chunk is sufficient.

Answer requirements:
For every question, provide a golden answer:
- The answer must be concise but complete.
- It must directly answer the question.
- It must include the essential details needed for correctness.
- It must be fully supported by the provided passage(s).
- Do not add external knowledge, assumptions, or explanations not present in the passage.
- Do not copy long sentences from the passage unnecessarily.

Output format:
Return only valid JSON.
Do not include explanations, headings, comments, or additional text.

Use exactly this structure:

[
  {
    "Question": "Natural conversational question here",
    "Gold_chunk_id": "chunk_id",
    "golden_answer": "Concise answer supported by the passage"
  }
]

For questions requiring multiple chunks:

[
  {
    "Question": "Natural conversational question here",
    "Gold_chunk_id": ["chunk_id_1", "chunk_id_2"],
    "golden_answer": "Answer supported by all required passages"
  }
]

Quality checklist before returning:
- Questions sound like real user queries, not generated benchmark questions.
- Questions require semantic retrieval, not keyword matching.
- Questions are varied and do not repeat the same information.
- Answers are fully grounded in the provided chunks.
- No outside knowledge is introduced.
- The output is valid JSON only. don't forget to generate questions for multiple chunks when possible